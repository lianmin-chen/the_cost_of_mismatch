import contextlib
import functools
import inspect
import json
import os
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Union

from datasets import Dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel, Trainer

import torch
import numpy as np
from torch import nn

from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_callback import (
    ExportableState,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    get_model_param_count,
)
from transformers.trainer_utils import (
    SaveStrategy,
    TrainOutput,
    speed_metrics,
)
from transformers.utils import (
    is_accelerate_available,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
    logging,
)
from transformers.trainer import _is_peft_model, TRAINER_STATE_NAME
from trl.trainer.utils import selective_log_softmax
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate.utils import (
        DistributedType,
    )

from .zo_bolt_config import ZOBOLTConfig

logger = logging.get_logger(__name__)


class ZOBOLTTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        args: ZOBOLTConfig,
        reward_funcs: List[Callable],
        processing_class: AutoTokenizer,
        pregenerated_data_file: str,
        peft_config: Optional[Dict] = None,
        beta: float = 0.1,
        **kwargs,
    ):
        
        self.tokenizer = processing_class
        self.reward_funcs = reward_funcs
        self.beta = beta
        self.gfol_args = args

        if args.trainer not in {"regular", "zo"}:
            raise ValueError(
                f"Unknown trainer mode {args.trainer}. "
                "Please use one of [regular, zo]."
            )

        print("--- Loading and processing pre-generated data ---")
        
        offline_dataset = None
        pregenerated_data_dir = os.path.dirname(os.path.abspath(pregenerated_data_file))
        source_dataset_name = os.path.splitext(os.path.basename(pregenerated_data_file))[0]
        processed_dataset_path = os.path.join(
            pregenerated_data_dir,
            f"weighted_{source_dataset_name}",
        )
        processed_dataset_ready = processed_dataset_path + ".done"

        dataset_is_ready = os.path.isdir(processed_dataset_path) and os.path.exists(processed_dataset_ready)

        if dataset_is_ready:
            offline_dataset = Dataset.load_from_disk(processed_dataset_path)
            print(f"--- Loaded cached processed dataset: {len(offline_dataset)} samples ---")
        elif self.gfol_args.local_rank in [-1, 0]:
            offline_dataset = self._load_and_process_data(pregenerated_data_file)
            print(f"--- Processed {len(offline_dataset)} samples for training ---")

            os.makedirs(self.gfol_args.output_dir, exist_ok=True)
            offline_dataset.save_to_disk(processed_dataset_path)
            with open(processed_dataset_ready, "w", encoding="utf-8") as f:
                f.write("ready\n")
        else:
            print(f"--- Rank {self.gfol_args.local_rank} waiting for processed dataset to be ready... ---")
            while not os.path.exists(processed_dataset_ready):
                time.sleep(5)
            offline_dataset = Dataset.load_from_disk(processed_dataset_path)
            print(f"--- Rank {self.gfol_args.local_rank} loaded {len(offline_dataset)} samples for training ---")

        if peft_config:
            model = PeftModel(model, peft_config)
            model.enable_input_require_grads()

        # Some models support `logits_to_keep`; detect once and reuse in loss computation.
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        super().__init__(
            model=model,
            args=args,
            train_dataset=offline_dataset,
            data_collator=self.collate_fn,
            processing_class=processing_class,
            **kwargs,
        )

        # Our custom compute_loss does not use num_items_in_batch.
        # Force Trainer to normalize loss by current_gradient_accumulation_steps
        # in training_step to keep loss scale consistent across GA settings.
        self.model_accepts_loss_kwargs = False

        distributed_type = str(getattr(self.accelerator, "distributed_type", "")).upper()
        if args.trainer == "zo" and any(
            name in distributed_type for name in ("DEEPSPEED", "FSDP")
        ):
            raise NotImplementedError(
                "ZO training supports replicated data parallelism, but not "
                "parameter-sharded DeepSpeed/FSDP models."
            )

    def _load_and_process_data(self, data_file: str) -> Dataset:
        new_dataset_rows = []
        
        if self.processing_class.pad_token is None:
            self.processing_class.pad_token = self.processing_class.eos_token

        print(f"Loading pre-generated data from {data_file}...")
        with open(data_file, 'r', encoding='utf-8') as f:
            pregenerated_data = json.load(f)

        for sample in tqdm(pregenerated_data, desc="Processing pre-generated data"):
            structured_prompt = sample.get('prompt')
            responses_text = sample.get('generated_texts')
            answer = sample.get('ground_truth_answer')

            if not all([structured_prompt, responses_text, answer]):
                continue

            num_generations = len(responses_text)
            if num_generations == 0:
                continue


            completions_for_reward = responses_text
            prompts_for_reward = [structured_prompt] * num_generations
            answers_for_reward = [answer] * num_generations

            rewards = torch.zeros(num_generations, device="cpu")
            for func in self.reward_funcs:
                r_values = func(prompts=prompts_for_reward, completions=completions_for_reward, answer=answers_for_reward)
                rewards += torch.tensor(r_values, device="cpu")
            
            # BOLT
            exp_rewards = torch.exp(rewards / self.beta)
            z_hat = torch.mean(exp_rewards)
            weights = exp_rewards / z_hat
            
            for j in range(num_generations):
                new_dataset_rows.append({
                    "prompt": structured_prompt,
                    "response": responses_text[j],
                    "weight": weights[j].item()
                })
        
        return Dataset.from_list(new_dataset_rows)

    def collate_fn(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = getattr(self.gfol_args, "max_length", None)
        prompt_max_length = getattr(self.gfol_args, "max_prompt_length", None)
        completion_max_length = getattr(self.gfol_args, "max_completion_length", None)
        if max_length is None and prompt_max_length is not None and completion_max_length is not None:
            max_length = prompt_max_length + completion_max_length
        if prompt_max_length is None:
            prompt_max_length = max_length

        prompts_responses = [f["prompt"] + f["response"] + self.processing_class.eos_token for f in features]
        weights = torch.tensor([f["weight"] for f in features])

        inputs = self.processing_class(
            prompts_responses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()

        prompt_texts = [f["prompt"] for f in features]
        prompt_tokenized = self.processing_class(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        # Mask the prompt length actually retained in the concatenated input.
        prompt_lengths = prompt_tokenized.attention_mask.sum(dim=1)
        sequence_lengths = inputs["attention_mask"].sum(dim=1)
        total_length = inputs["input_ids"].size(1)

        for i in range(len(features)):
            prompt_start = total_length - sequence_lengths[i].item()
            prompt_end = prompt_start + prompt_lengths[i].item()
            labels[i, prompt_start:prompt_end] = -100
        
        labels[inputs["attention_mask"] == 0] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "weights": weights,
        }

    def compute_loss_original(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weights").to(model.device)
        labels = inputs.pop("labels").to(model.device)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=False, use_cache=False)
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss_per_sample = loss.view(shift_labels.size())

        mask = (shift_labels != -100).float()
        
        sum_loss_per_sample = (loss_per_sample * mask).sum(dim=1)
        num_valid_tokens = mask.sum(dim=1).clamp(min=1.0)
        avg_loss_per_sample = sum_loss_per_sample / num_valid_tokens

        weighted_loss = (avg_loss_per_sample * weights).mean()
        return (weighted_loss, outputs) if return_outputs else weighted_loss
    
    # def compute_loss_original(self, model, inputs, return_outputs=False, **kwargs):
    #     # memory reduced version: with GRPOTrainer style
    #     weights = inputs.pop("weights").to(model.device)
    #     labels = inputs.pop("labels").to(model.device)
    #     inputs = {k: v.to(model.device) for k, v in inputs.items()}
    #     batch_size = labels.size(0)

    #     # Reduce peak memory by processing the batch in smaller chunks.
    #     loss_chunk_size = getattr(self.gfol_args, "loss_chunk_size", None)
    #     if loss_chunk_size is None:
    #         loss_chunk_size = max(1, batch_size // 2) if batch_size > 1 else 1
    #     loss_chunk_size = max(1, min(batch_size, int(loss_chunk_size)))

    #     def _forward_and_score(chunk_inputs, chunk_labels, chunk_weights):
    #         completion_mask = chunk_labels != -100
    #         logits_to_keep = int(completion_mask.sum(dim=1).max().item())

    #         model_kwargs = {
    #             **chunk_inputs,
    #             "output_hidden_states": False,
    #             "use_cache": False,
    #         }
    #         if logits_to_keep > 0 and "logits_to_keep" in self.model_kwarg_keys:
    #             # Add 1 because the last logit is dropped before token loss computation.
    #             model_kwargs["logits_to_keep"] = logits_to_keep + 1

    #         outputs = model(**model_kwargs)
    #         logits = outputs.logits

    #         # Keep completion-side logits/labels to reduce compute and memory.
    #         if logits_to_keep > 0:
    #             logits = logits[:, :-1, :]
    #             logits = logits[:, -logits_to_keep:, :]
    #             chunk_labels = chunk_labels[:, -logits_to_keep:]
    #         else:
    #             logits = logits[:, :-1, :]
    #             chunk_labels = chunk_labels[:, 1:]

    #         valid_mask = chunk_labels != -100
    #         if not valid_mask.any():
    #             zero = logits.sum() * 0.0
    #             return zero, outputs

    #         valid_logits = logits[valid_mask]
    #         valid_labels = chunk_labels[valid_mask]
    #         token_nll = -selective_log_softmax(valid_logits, valid_labels)

    #         sample_indices = valid_mask.nonzero(as_tuple=False)[:, 0]
    #         per_sample_loss_sum = torch.zeros(
    #             chunk_labels.size(0), device=token_nll.device, dtype=token_nll.dtype
    #         )
    #         per_sample_loss_sum.scatter_add_(0, sample_indices, token_nll)

    #         num_valid_tokens = valid_mask.sum(dim=1).to(token_nll.dtype).clamp(min=1.0)
    #         avg_loss_per_sample = per_sample_loss_sum / num_valid_tokens
    #         weighted_loss_sum = (avg_loss_per_sample * chunk_weights).sum()
    #         return weighted_loss_sum, outputs

    #     # Keep full-batch path when outputs are requested; chunked outputs are not merged.
    #     if return_outputs or batch_size <= loss_chunk_size:
    #         weighted_loss_sum, outputs = _forward_and_score(inputs, labels, weights)
    #         weighted_loss = weighted_loss_sum / batch_size
    #         return (weighted_loss, outputs) if return_outputs else weighted_loss

    #     weighted_loss_sum = torch.zeros((), device=model.device, dtype=torch.float32)
    #     total_samples = 0
    #     for start in range(0, batch_size, loss_chunk_size):
    #         end = min(start + loss_chunk_size, batch_size)
    #         chunk_inputs = {k: v[start:end] for k, v in inputs.items()}
    #         chunk_labels = labels[start:end]
    #         chunk_weights = weights[start:end]
    #         chunk_loss_sum, _ = _forward_and_score(chunk_inputs, chunk_labels, chunk_weights)
    #         weighted_loss_sum = weighted_loss_sum + chunk_loss_sum
    #         total_samples += end - start

    #     weighted_loss = weighted_loss_sum / total_samples
    #     return weighted_loss

    def compute_loss_flce(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weights").to(model.device)
        labels = inputs.pop("labels").to(model.device)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
        outputs = model(**inputs, output_hidden_states=False, use_cache=False)
        logits = outputs.get("logits")
    
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
    
        batch_size = shift_labels.size(0)
        loss_dtype = shift_logits.dtype
        per_sample_loss_sum = torch.zeros(batch_size, device=shift_logits.device, dtype=loss_dtype)
        num_valid_tokens = torch.zeros(batch_size, device=shift_logits.device, dtype=loss_dtype)
    
        token_chunk_size = int(getattr(self.args, "loss_token_chunk_size", 128))
        if token_chunk_size <= 0:
            token_chunk_size = 128
    
        seq_len_minus_one = shift_labels.size(1)
        for start in range(0, seq_len_minus_one, token_chunk_size):
            end = min(start + token_chunk_size, seq_len_minus_one)
            chunk_logits = shift_logits[:, start:end, :]
            chunk_labels = shift_labels[:, start:end]
    
            chunk_valid_mask = chunk_labels.ne(-100)
            if not chunk_valid_mask.any():
                continue
    
            valid_logits = chunk_logits[chunk_valid_mask]
            valid_labels = chunk_labels[chunk_valid_mask]
    
            token_loss = torch.nn.functional.cross_entropy(
                valid_logits,
                valid_labels,
                reduction="none",
            )
    
            sample_indices = chunk_valid_mask.nonzero(as_tuple=False)[:, 0]
            per_sample_loss_sum.scatter_add_(0, sample_indices, token_loss)
            num_valid_tokens += chunk_valid_mask.sum(dim=1).to(loss_dtype)
    
        num_valid_tokens = num_valid_tokens.clamp(min=1.0)
        avg_loss_per_sample = per_sample_loss_sum / num_valid_tokens
        weighted_loss = (avg_loss_per_sample * weights.to(loss_dtype)).mean()
    
        return (weighted_loss, outputs) if return_outputs else weighted_loss

    # def _get_lm_backbone_and_head(self, model):
    #     lm_model = model
    #     if hasattr(lm_model, "get_base_model"):
    #         try:
    #             base = lm_model.get_base_model()
    #             if hasattr(base, "model") and hasattr(base, "lm_head"):
    #                 lm_model = base
    #         except Exception:
    #             pass

    #     if not hasattr(lm_model, "model") or not hasattr(lm_model, "lm_head"):
    #         raise ValueError("Current model does not expose `.model` and `.lm_head`, cannot run CCE/FLCE path.")

    #     return lm_model.model, lm_model.lm_head
    
    def compute_loss_cce(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weights").to(model.device)
        labels = inputs.pop("labels").to(model.device)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # === forward backbone（不产生 logits）===
        backbone, lm_head = self._get_lm_backbone_and_head(model)
        outputs = backbone(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)

        hidden_states = outputs.last_hidden_state  # [B, T, H]

        # === shift 对齐（完全等价）===
        hidden_states = hidden_states[:, :-1, :]      # [B, T-1, H]
        shift_labels = labels[:, 1:]                  # [B, T-1]

        # === mask ===
        valid_mask = shift_labels.ne(-100)

        if not valid_mask.any():
            zero = hidden_states.sum() * 0.0
            return (zero, outputs) if return_outputs else zero

        # === flatten valid token ===
        valid_hidden = hidden_states[valid_mask]      # [N, H]
        valid_labels = shift_labels[valid_mask].long()  # [N]

        # === lm_head 参数 ===
        lm_weight = lm_head.weight                   # [V, H]
        lm_bias = getattr(lm_head, "bias", None)     # [V] or None

        # =========================================================
        # 1️⃣ 计算 target logits（完全等价于 gather）
        # =========================================================
        target_w = lm_weight.index_select(0, valid_labels)   # [N, H]
        target_logits = (valid_hidden * target_w).sum(dim=-1)  # [N]

        if lm_bias is not None:
            target_b = lm_bias.index_select(0, valid_labels)
            target_logits = target_logits + target_b

        # =========================================================
        # 2️⃣ 分块计算 logsumexp（核心 CCE）
        # =========================================================
        vocab_size = lm_weight.size(0)
        chunk_size = int(getattr(self.args, "cce_vocab_chunk_size", 4096))
        if chunk_size <= 0:
            chunk_size = 4096

        running_max = torch.full_like(target_logits, -float("inf"))
        running_sumexp = torch.zeros_like(target_logits)

        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)

            weight_chunk = lm_weight[start:end]  # [chunk, H]
            bias_chunk = lm_bias[start:end] if lm_bias is not None else None

            # [N, chunk]
            chunk_logits = torch.nn.functional.linear(valid_hidden, weight_chunk, bias_chunk)

            chunk_max = chunk_logits.max(dim=-1).values
            new_max = torch.maximum(running_max, chunk_max)

            running_sumexp = (
                torch.exp(running_max - new_max) * running_sumexp
                + torch.exp(chunk_logits - new_max.unsqueeze(-1)).sum(dim=-1)
            )

            running_max = new_max

        logsumexp = running_max + torch.log(running_sumexp.clamp_min(1e-20))

        # =========================================================
        # 3️⃣ token-level NLL（完全等价）
        # =========================================================
        token_loss = logsumexp - target_logits  # [N]

        # =========================================================
        # 4️⃣ reshape 回 per-sample（完全等价）
        # =========================================================
        B = shift_labels.size(0)
        loss_dtype = token_loss.dtype

        per_sample_loss_sum = torch.zeros(B, device=token_loss.device, dtype=loss_dtype)

        sample_indices = valid_mask.nonzero(as_tuple=False)[:, 0]  # [N]
        per_sample_loss_sum.scatter_add_(0, sample_indices, token_loss)

        num_valid_tokens = valid_mask.sum(dim=1).to(loss_dtype).clamp(min=1.0)
        avg_loss_per_sample = per_sample_loss_sum / num_valid_tokens

        # =========================================================
        # 5️⃣ weighted mean（完全等价）
        # =========================================================
        weighted_loss = (avg_loss_per_sample * weights.to(loss_dtype)).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss

    def _get_lm_backbone_and_head(self, model):
        lm_model = model
        if hasattr(lm_model, "module"):
            lm_model = lm_model.module

        if hasattr(lm_model, "get_base_model"):
            try:
                base = lm_model.get_base_model()
                if hasattr(base, "model") and hasattr(base, "lm_head"):
                    lm_model = base
            except Exception:
                pass

        if not hasattr(lm_model, "model") or not hasattr(lm_model, "lm_head"):
            raise ValueError("Current model does not expose `.model` and `.lm_head`, cannot run FLCE path.")

        return lm_model.model, lm_model.lm_head

    def compute_loss_liger(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weights").to(model.device)
        labels = inputs.pop("labels").to(model.device)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        backbone, lm_head = self._get_lm_backbone_and_head(model)
        outputs = backbone(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)

        hidden_states = outputs.last_hidden_state[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        batch_size, seq_len_minus_one = shift_labels.shape
        flat_hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        flat_labels = shift_labels.view(-1)

        if not hasattr(self, "_liger_flce"):
            self._liger_flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="none")

        token_loss = self._liger_flce(
            lm_head.weight,
            flat_hidden_states,
            flat_labels,
            bias=getattr(lm_head, "bias", None),
        )

        loss_per_sample = token_loss.view(batch_size, seq_len_minus_one)
        mask = shift_labels.ne(-100).to(loss_per_sample.dtype)

        sum_loss_per_sample = (loss_per_sample * mask).sum(dim=1)
        num_valid_tokens = mask.sum(dim=1).clamp(min=1.0)
        avg_loss_per_sample = sum_loss_per_sample / num_valid_tokens
        weighted_loss = (avg_loss_per_sample * weights.to(avg_loss_per_sample.dtype)).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_imp = str(getattr(self.args, "loss_imp", "original")).lower()

        if loss_imp == "original":
            return self.compute_loss_original(model, inputs, return_outputs=return_outputs, **kwargs)
        if loss_imp == "flce":
            return self.compute_loss_flce(model, inputs, return_outputs=return_outputs, **kwargs)
        if loss_imp == "cce":
            return self.compute_loss_cce(model, inputs, return_outputs=return_outputs, **kwargs)
        if loss_imp == "liger":
            return self.compute_loss_liger(model, inputs, return_outputs=return_outputs, **kwargs)

        raise ValueError(f"Unknown loss_imp {loss_imp}. Please use one of [original, flce, cce, liger].")

    def compare_loss_implementations(self, model, inputs):
        def clone_inputs(batch):
            cloned = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    cloned[key] = value.detach().clone()
                else:
                    cloned[key] = value
            return cloned

        loss_fns = {
            "original": self.compute_loss_original,
            "flce": self.compute_loss_flce,
            "cce": self.compute_loss_cce,
            "liger": self.compute_loss_liger,
        }

        was_training = model.training
        model.eval()
        results = {}

        try:
            with torch.inference_mode():
                for name, loss_fn in loss_fns.items():
                    batch_inputs = clone_inputs(inputs)
                    loss = loss_fn(model, batch_inputs, return_outputs=False)
                    results[name] = float(loss.detach().item())
        finally:
            if was_training:
                model.train()

        base = results["original"]
        print("[loss compare] values:", {k: round(v, 8) for k, v in results.items()})
        print(
            "[loss compare] deltas_vs_original:",
            {k: round(v - base, 8) for k, v in results.items() if k != "original"},
        )
        return results


    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self.get_total_train_batch_size(args)

        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                DebugUnderflowOverflow(self.model)

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = model is self.model

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    # We should avoid accelerate preparing the model in TP case since we dont need it as it is handled by transformers from_pretrained and also it goes into DDP based preparation.
                    if self.is_tp_enabled:
                        self.optimizer = self.accelerator.prepare(self.optimizer)
                    else:
                        model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1
            rng_to_sync = False

            # Handle resumption from checkpoint
            if epoch == epochs_trained and resume_from_checkpoint is not None:
                if steps_trained_in_current_epoch > 0 and not args.ignore_data_skip:
                    epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                    step = steps_trained_in_current_epoch - 1
                    rng_to_sync = True
                elif steps_trained_in_current_epoch == 0:
                    self._load_rng_state(resume_from_checkpoint)

            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                # Store the number of batches for current gradient accumulation
                # This is used to correctly scale the loss when the last accumulation step has fewer batches
                self.current_gradient_accumulation_steps = len(batch_samples)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen not in ["no", False]:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            if self.args.include_num_input_tokens_seen == "non_padding":
                                if "attention_mask" in inputs:
                                    input_tokens = inputs["attention_mask"].sum()
                                elif (
                                    self.processing_class is not None
                                    and hasattr(self.processing_class, "pad_token_id")
                                    and self.processing_class.pad_token_id is not None
                                ):
                                    input_tokens = (
                                        inputs[main_input_name] != self.processing_class.pad_token_id
                                    ).sum()
                                else:
                                    logger.warning(
                                        "Could not determine method to count non-padding tokens, falling back to counting all tokens."
                                    )
                                    input_tokens = inputs[main_input_name].numel()
                            else:
                                input_tokens = inputs[main_input_name].numel()

                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()

                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # Estimate the ZoAR gradient for zeroth-order training.
                    if args.trainer == "zo":
                        tr_loss_step = self.do_step(model, inputs)
                    else:
                        # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                        context = (
                            functools.partial(self.accelerator.no_sync, model=model)
                            if i != len(batch_samples) - 1
                            and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                            else contextlib.nullcontext
                        )
                        with context():
                            tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Apply the R-AdaZO update for zeroth-order training.
                        if args.trainer == "zo":
                            self.do_update(model)
                        else:
                            # Gradient clipping
                            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                                if is_sagemaker_mp_enabled() and args.fp16:
                                    _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                                elif self.use_apex:
                                    from apex import amp

                                    # Revert to normal clipping otherwise, handling Apex or full precision
                                    _grad_norm = nn.utils.clip_grad_norm_(
                                        amp.master_params(self.optimizer),
                                        args.max_grad_norm,
                                    )
                                else:
                                    grad_norm_context = contextlib.nullcontext
                                    if self.is_tp_enabled:
                                        from torch.distributed._tensor.experimental import implicit_replication

                                        grad_norm_context = implicit_replication
                                    with grad_norm_context():
                                        _grad_norm = self.accelerator.clip_grad_norm_(
                                            model.parameters(),
                                            args.max_grad_norm,
                                        )

                                if (
                                    is_accelerate_available()
                                    and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                                ):
                                    grad_norm = model.get_global_grad_norm()
                                    # In some cases the grad norm may not return a float
                                    if hasattr(grad_norm, "item"):
                                        grad_norm = grad_norm.item()
                                else:
                                    grad_norm = _grad_norm

                            self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                            context = contextlib.nullcontext
                            if self.is_tp_enabled:
                                from torch.distributed._tensor.experimental import implicit_replication

                                context = implicit_replication

                            with context():
                                self.optimizer.step()

                            self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                            # get leaning rate before update
                            learning_rate = self._get_learning_rate()

                            if not self.accelerator.optimizer_step_was_skipped:
                                # Delay optimizer scheduling until metrics are generated
                                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    self.lr_scheduler.step()

                            model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                            learning_rate=learning_rate,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()
            if hasattr(self, "projected_grad") and self.projected_grad is not None:
                logs["zo/projected_grad"] = self.projected_grad.mean().item()
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    ############## Zeroth-order finite differences ##############

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: seed for in-place perturbation (if None, use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """
        
        for name, param in self.named_parameters_to_optim:
            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.add_(scaling_factor * z * self._get_zo_eps())

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    def _get_zo_minus_inputs(self, inputs):
        """Return the data used by the minus/base finite-difference evaluation.

        Standard ZO-BOLT is matched: both evaluations use the dataloader batch.
        Subclasses can override this hook without duplicating the training loop
        or optimizer implementations.
        """
        return inputs

    def _get_current_gradient_accumulation_steps(self):
        """Return the number of micro-batches in the current optimizer window."""
        accum_steps = getattr(self, "current_gradient_accumulation_steps", None)
        if accum_steps is None or accum_steps <= 0:
            accum_steps = self.args.gradient_accumulation_steps
        return max(1, int(accum_steps))

    @staticmethod
    def _get_zo_batch_size(inputs):
        """Infer the micro-batch size used to weight its mean loss."""
        for key in ("input_ids", "labels", "weights"):
            value = inputs.get(key)
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.size(0))
        raise ValueError("Cannot infer ZO micro-batch size from inputs.")

    def _new_zo_seed(self):
        """Create one perturbation seed shared by all replicated ranks."""
        # Advance NumPy on every rank, then let rank 0 define the shared direction.
        seed = int(np.random.randint(1000000000))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            seed_tensor = torch.tensor(
                seed,
                device=self.accelerator.device,
                dtype=torch.int64,
            )
            torch.distributed.broadcast(seed_tensor, src=0)
            seed = int(seed_tensor.item())
        return seed

    @staticmethod
    def _global_zo_loss_average(loss_pos_sum, loss_neg_sum, num_samples):
        """Return sample-weighted two-sided losses over all replicated ranks."""
        stats = torch.stack(
            [
                loss_pos_sum.float(),
                loss_neg_sum.float(),
                loss_pos_sum.new_tensor(float(num_samples), dtype=torch.float32),
            ]
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

        if stats[2].item() <= 0:
            raise RuntimeError("Cannot finalize an empty ZO accumulation window.")
        return stats[0] / stats[2], stats[1] / stats[2]

    @torch.no_grad()
    def _zo_forward_pair(self, model, plus_inputs, minus_inputs, random_seed):
        """Evaluate a two-sided ZO pair and always restore the center parameters."""
        position = 0
        try:
            self.zo_perturb_parameters(random_seed=random_seed, scaling_factor=1)
            position = 1
            loss_pos = self.zo_forward(model, plus_inputs)

            self.zo_perturb_parameters(random_seed=random_seed, scaling_factor=-2)
            position = -1
            loss_neg = self.zo_forward(model, minus_inputs)
            return loss_pos, loss_neg
        finally:
            if position != 0:
                self.zo_perturb_parameters(random_seed=random_seed, scaling_factor=-position)

    def _reset_zoar_accum_state(self):
        self.zoar_accum_steps = 0
        self.zoar_accum_samples = 0
        self.zoar_accum_seed = None
        self.zoar_accum_reward_pos = None
        self.zoar_accum_reward_neg = None

    ############## ZoAR ##############

    def zoar_step(self, model, inputs):
        """
        Accumulate one sample-weighted ZoAR history pair per optimizer window.
        """
        args = self.args

        # What parameters to optimize 
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # store the current loss
        loss = self.zo_forward(model, inputs)

        accum_steps = self._get_current_gradient_accumulation_steps()
        if getattr(self, "zoar_accum_steps", 0) == 0:
            self._reset_zoar_accum_state()
            self.zoar_accum_seed = self._new_zo_seed()

        num_histories = args.num_histories
        if getattr(self, "rl_past", None) is None:
            self.rl_past = []

        batch_size = self._get_zo_batch_size(inputs)
        seed = self.zoar_accum_seed
        reward_pos, reward_neg = self._zo_forward_pair(
            model,
            inputs,
            self._get_zo_minus_inputs(inputs),
            seed,
        )

        weighted_reward_pos = reward_pos.detach().float() * batch_size
        weighted_reward_neg = reward_neg.detach().float() * batch_size
        if self.zoar_accum_reward_pos is None:
            self.zoar_accum_reward_pos = weighted_reward_pos
            self.zoar_accum_reward_neg = weighted_reward_neg
        else:
            self.zoar_accum_reward_pos += weighted_reward_pos
            self.zoar_accum_reward_neg += weighted_reward_neg
        self.zoar_accum_steps += 1
        self.zoar_accum_samples += batch_size

        if self.zoar_accum_steps == accum_steps:
            avg_reward_pos, avg_reward_neg = self._global_zo_loss_average(
                self.zoar_accum_reward_pos,
                self.zoar_accum_reward_neg,
                self.zoar_accum_samples,
            )
            self.rl_past.append([avg_reward_pos, seed])
            self.rl_past.append([avg_reward_neg, seed])

            if len(self.rl_past) > (num_histories + 1) * 2:
                self.rl_past = self.rl_past[-(num_histories + 1) * 2:]

            self.projected_seeds = [p[1] for p in self.rl_past]
            rewards = torch.stack([p[0] for p in self.rl_past]).to(loss.device)
            rewards = rewards - rewards.mean()
            self.projected_grad = rewards / self._get_zo_eps()
            self.projected_grad[1::2].mul_(-1)

        return loss / accum_steps

    def zoar_update(self, model):
        """
        Update the parameters with the estimated gradients using Adam-like optimization.
        """
        args = self.args
        expected_steps = self._get_current_gradient_accumulation_steps()
        if (
            getattr(self, "zoar_accum_seed", None) is None
            or getattr(self, "zoar_accum_steps", 0) != expected_steps
        ):
            raise RuntimeError(
                "ZoAR update requested before the accumulation window was complete."
            )

        if not hasattr(self, 'zo_adam_m'):
            self.zo_adam_m = {}
            self.zo_adam_v = {}
            self.zo_adam_t = 0
            for name, param in self.named_parameters_to_optim:
                self.zo_adam_m[name] = torch.zeros_like(param)
                self.zo_adam_v[name] = torch.zeros_like(param)

        beta1 = args.adam_beta1
        beta2 = args.adam_beta2
        eps = 1e-8
        self.zo_adam_t += 1
        
        # Reset the random seed for sampling zs
        for name, param in self.named_parameters_to_optim:
            # Resample z
            grad = torch.zeros_like(param)
            for seed, pgrad in zip(self.projected_seeds, self.projected_grad):
                torch.manual_seed(seed)
                z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
                # Compute gradient estimate
                grad += pgrad * z
            grad /= len(self.projected_seeds)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                grad += args.weight_decay * param

            self.zo_adam_m[name] = beta1 * self.zo_adam_m[name] + (1 - beta1) * grad
            m_hat = self.zo_adam_m[name]
            self.zo_adam_v[name] = beta2 * self.zo_adam_v[name] + \
                (1 - beta2) * (m_hat * m_hat)
            v_hat = self.zo_adam_v[name]
            if self.zo_adam_t < 30:
                v_hat = torch.ones_like(v_hat, device=v_hat.device)
            param.add_(-self._get_learning_rate() * m_hat / (torch.sqrt(v_hat) + eps))

        self.lr_scheduler.step()
        self._reset_zoar_accum_state()

    @torch.no_grad()
    def do_step(self, model, inputs):
        return self.zoar_step(model, inputs)
    
    @torch.no_grad()
    def do_update(self, model):
        return self.zoar_update(model)

    def _get_zo_eps(self):
        return self.args.zo_eps
