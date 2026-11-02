"""Compute per-example losses for the fixed weighted BOLT objective."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from datasets import Dataset
from tqdm.auto import tqdm


class FixedBoltObjective:
    """Fixed objective: weight times mean response-token NLL per example."""

    def __init__(self, model, tokenizer, dataset: Dataset, config: Mapping[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_prompt_length = int(config["max_prompt_length"])
        self.max_length = self.max_prompt_length + int(config["max_completion_length"])
        self.loss_impl = str(config["loss_impl"])
        self._liger_loss = None

    def collate(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        """Build model inputs with labels masked on prompt and padding tokens."""
        features = [self.dataset[int(index)] for index in indices]
        texts = [
            feature["prompt"] + feature["response"] + self.tokenizer.eos_token
            for feature in features
        ]
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompts = self.tokenizer(
            [feature["prompt"] for feature in features],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        # Match the prompt span retained in the full input, not its nominal budget.
        prompt_lengths = prompts["attention_mask"].sum(dim=1)
        sequence_lengths = inputs["attention_mask"].sum(dim=1)
        total_length = labels.size(1)
        # With left padding, each prompt starts at a row-specific position.
        for row in range(len(indices)):
            prompt_start = total_length - int(sequence_lengths[row])
            prompt_end = prompt_start + int(prompt_lengths[row])
            labels[row, prompt_start:prompt_end] = -100
        labels[inputs["attention_mask"] == 0] = -100
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "weights": torch.tensor(
                [feature["weight"] for feature in features], dtype=torch.float32
            ),
        }

    def per_sample_contributions(
        self, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return one weighted loss per example, not a batch mean."""
        device = next(self.model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        weights = batch["weights"].to(device=device, dtype=torch.float32)

        if self.loss_impl == "original":
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            ).logits
            shifted_logits = logits[:, :-1].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            token_loss = torch.nn.functional.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(shifted_labels)
            mask = shifted_labels.ne(-100)
            sample_loss = (
                (token_loss.float() * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1).float()
            )
            return sample_loss * weights

        from liger_kernel.transformers.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyLoss,
        )

        # Compute Liger loss row by row to retain each example's weight.
        causal_lm = self.model.get_base_model()
        if not hasattr(causal_lm, "model") or not hasattr(causal_lm, "lm_head"):
            raise RuntimeError("Qwen model does not expose model/lm_head")
        backbone, lm_head = causal_lm.model, causal_lm.lm_head
        hidden = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, :-1]
        shifted_labels = labels[:, 1:]
        if self._liger_loss is None:
            self._liger_loss = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100, reduction="mean"
            )
        losses = []
        for row in range(hidden.size(0)):
            valid = shifted_labels[row].ne(-100)
            if bool(valid.any()):
                value = self._liger_loss(
                    lm_head.weight,
                    hidden[row][valid].contiguous(),
                    shifted_labels[row][valid].contiguous(),
                    bias=getattr(lm_head, "bias", None),
                )
            else:
                value = hidden[row].sum() * 0.0
            losses.append(value.float())
        return torch.stack(losses) * weights

    @torch.inference_mode()
    def evaluate_all(self, microbatch_size: int, label: str) -> np.ndarray:
        """Return per-example losses over the dataset in original row order."""
        # Sort by length to reduce padding, then restore the original row order.
        order = sorted(
            range(len(self.dataset)),
            key=lambda index: (
                len(self.dataset[index]["prompt"]) + len(self.dataset[index]["response"]),
                index,
            ),
        )
        contributions = np.empty(len(self.dataset), dtype=np.float32)
        num_batches = math.ceil(len(order) / microbatch_size)
        for start in tqdm(
            range(0, len(order), microbatch_size),
            total=num_batches,
            desc=label,
            unit="microbatch",
            dynamic_ncols=True,
            leave=False,
        ):
            indices = order[start : start + microbatch_size]
            values = (
                self.per_sample_contributions(self.collate(indices))
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            contributions[np.asarray(indices)] = values
        if not np.isfinite(contributions).all():
            raise RuntimeError(f"non-finite objective contributions during {label}")
        return contributions
