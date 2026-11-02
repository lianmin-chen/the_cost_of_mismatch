"""Load the shared FP32 base model and fresh LoRA adapter at the initial point."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import set_all_seeds


def prepare_model(config: Mapping[str, Any]):
    """Return the model, tokenizer, and trainable LoRA parameters without training."""
    if not torch.cuda.is_available():
        raise RuntimeError("the oracle-noise experiments require a CUDA GPU")
    model_path = Path(config["assets"]["base_model"])
    if not model_path.exists():
        raise FileNotFoundError(f"local base model not found: {model_path}")

    model_config = config["model"]
    seed = int(config["experiment"]["global_seed"])
    set_all_seeds(seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation=model_config["attn_implementation"],
        device_map=None,
    )
    # Loading the base model must not alter the initial LoRA parameters.
    set_all_seeds(seed)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(model_config["lora_r"]),
            lora_alpha=int(model_config["lora_alpha"]),
            lora_dropout=float(model_config["lora_dropout"]),
            target_modules=list(model_config["target_modules"]),
            task_type="CAUSAL_LM",
            init_lora_weights=True,
        ),
    )
    model.cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise RuntimeError("PEFT created no trainable LoRA parameters")
    unexpected = [name for name, _ in named_parameters if "lora_" not in name]
    if unexpected:
        raise RuntimeError(f"non-LoRA parameters are trainable: {unexpected[:5]}")
    b_parameters = [
        parameter for name, parameter in named_parameters if "lora_B" in name
    ]
    if not b_parameters or any(
        torch.count_nonzero(parameter.detach()).item() for parameter in b_parameters
    ):
        raise RuntimeError("fresh LoRA-B parameters must be zero")

    dimension = sum(parameter.numel() for _, parameter in named_parameters)
    print(
        f"[model] {dimension:,} perturbed LoRA parameters; "
        f"{len(b_parameters)} zero-initialized LoRA-B tensors on cuda",
        flush=True,
    )
    return model, tokenizer, named_parameters
