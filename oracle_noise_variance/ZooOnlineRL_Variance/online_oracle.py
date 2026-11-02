"""Query the direct on-policy objective using fresh sampled completions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from common import set_all_seeds
from rewards.math_grader import qwen_boxed_accuracy


class OnPolicyOracle:
    """Return negative mean verifier reward on a fixed set of prompts."""

    def __init__(
        self, model, tokenizer, prompt_rows: Sequence[Mapping[str, Any]], config
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = [row["prompt"] for row in prompt_rows]
        self.answers = [row["answer"] for row in prompt_rows]
        self.generation = config["generation"]
        encoded = tokenizer(
            self.prompts,
            padding=True,
            truncation=True,
            max_length=int(config["model"]["max_prompt_length"]),
            return_tensors="pt",
        )
        self.encoded = {key: value.cuda() for key, value in encoded.items()}
        self.prompt_width = int(self.encoded["input_ids"].shape[1])

    @torch.inference_mode()
    def query(self, seed: int) -> float:
        # Different seeds make the two endpoints independent on-policy queries.
        set_all_seeds(seed)
        generated = self.model.generate(
            **self.encoded,
            do_sample=True,
            temperature=float(self.generation["temperature"]),
            top_p=float(self.generation["top_p"]),
            top_k=int(self.generation["top_k"]),
            max_new_tokens=int(self.generation["max_new_tokens"]),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        completions = self.tokenizer.batch_decode(
            generated[:, self.prompt_width :], skip_special_tokens=True
        )
        rewards = qwen_boxed_accuracy(
            prompts=self.prompts,
            completions=completions,
            answer=self.answers,
        )
        if len(rewards) != len(self.prompts):
            raise RuntimeError("verifier returned an unexpected number of rewards")
        return -sum(float(value) for value in rewards) / len(rewards)
