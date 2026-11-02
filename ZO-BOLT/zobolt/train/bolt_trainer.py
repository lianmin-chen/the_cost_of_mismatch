import math
import inspect
import re
import torch
import random
import json
import argparse
import os
import time
import ctypes.util
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, PreTrainedModel
from peft import LoraConfig, PeftModel
from trl import GRPOConfig
from trl.trainer.utils import selective_log_softmax
from typing import List, Callable, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

class BOLTTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        args: GRPOConfig,
        reward_funcs: List[Callable],
        processing_class: AutoTokenizer,
        pregenerated_data_file: str,
        peft_config: Optional[Dict] = None,
        beta: float = 0.1,
        **kwargs,
    ):
        
        self.tokenizer = processing_class
        self.processing_class = processing_class
        self.reward_funcs = reward_funcs
        self.beta = beta
        self.gfol_args = args

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
            model = PeftModel(model, peft_config, autocast_adapter_dtype=False)
            model.enable_input_require_grads()
            model.print_trainable_parameters()

        # Some models support `logits_to_keep`; we detect it once and reuse the result in `compute_loss`.
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
                print(f"Skipping sample due to missing data: {sample}")
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
            prompt_start = total_length - sequence_lengths[i]
            prompt_end = prompt_start + prompt_lengths[i]
            labels[i, prompt_start:prompt_end] = -100
        
        labels[inputs["attention_mask"] == 0] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "weights": weights,
        }

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
    
    # def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
    #             # We add 1 because the last logit is dropped before computing the token loss.
    #             model_kwargs["logits_to_keep"] = logits_to_keep + 1

    #         outputs = model(**model_kwargs)
    #         logits = outputs.logits

    #         # Keep only the completion-side logits/labels; prompt tokens stay masked as -100.
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
    #         per_sample_token_counts = valid_mask.sum(dim=1)

    #         counts = per_sample_token_counts.detach().cpu().tolist()
    #         print(
    #             f"[BOLTTrainer] per-sample token counts: {counts} | "
    #             f"min={min(counts) if counts else 0}, "
    #             f"max={max(counts) if counts else 0}, "
    #             f"mean={(sum(counts) / len(counts)) if counts else 0:.2f}"
    #         )

    #         per_sample_loss_sum = torch.zeros(
    #             chunk_labels.size(0), device=token_nll.device, dtype=token_nll.dtype
    #         )
    #         per_sample_loss_sum.scatter_add_(0, sample_indices, token_nll)

    #         num_valid_tokens = per_sample_token_counts.to(token_nll.dtype).clamp(min=1.0)
    #         avg_loss_per_sample = per_sample_loss_sum / num_valid_tokens
    #         weighted_loss_sum = (avg_loss_per_sample * chunk_weights).sum()
    #         return weighted_loss_sum, outputs

    #     # Keep the full-batch path when outputs are explicitly requested; chunked outputs are harder to merge.
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
