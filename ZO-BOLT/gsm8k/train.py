"""Train the GSM8K first-order BOLT baseline or matched ZO-BOLT."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig

from zobolt.rewards.math_grader import qwen_boxed_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("bolt", "zo-bolt"), required=True)
    parser.add_argument("--data-file", type=Path, required=True, help="Filtered and truncated GSM8K generation JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Checkpoint and log directory")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--no-wandb", action="store_true", help="Disable the preset W&B logging")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    data_file = args.data_file.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    is_main_process = int(os.environ.get("RANK", str(max(args.local_rank, 0)))) == 0
    if is_main_process and output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")

    config_path = Path(__file__).resolve().parent / "configs" / f"{args.method}.json"
    preset = json.loads(config_path.read_text(encoding="utf-8"))
    set_seed(preset["training_args"]["seed"])
    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", "BOLT_EXPS")
    training_args = {
        **preset["training_args"],
        "output_dir": str(output_dir),
        "run_name": output_dir.name,
        "report_to": "none" if args.no_wandb else "wandb",
    }
    if training_args.get("deepspeed"):
        training_args["deepspeed"] = str(ROOT / training_args["deepspeed"])
    if args.method == "bolt":
        from zobolt.train.bolt_trainer import BOLTTrainer as TrainerClass

        ConfigClass = GRPOConfig
    else:
        from zobolt.train.zo_bolt_config import ZOBOLTConfig as ConfigClass
        from zobolt.train.zo_bolt_trainer import ZOBOLTTrainer as TrainerClass

        training_args.update(preset["zo_args"])
    # Pass max_prompt_length through GRPOConfig versions that omit this field.
    prompt_length = None
    if "max_prompt_length" not in inspect.signature(ConfigClass).parameters:
        prompt_length = training_args.pop("max_prompt_length", None)
    config = ConfigClass(**training_args)
    if prompt_length is not None:
        config.max_prompt_length = prompt_length

    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process:
        (output_dir / "run_config.json").write_text(
            json.dumps(
                {"method": args.method, "model": args.model, "data_file": str(data_file), "preset": preset},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    trainer = TrainerClass(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[qwen_boxed_accuracy],
        args=config,
        pregenerated_data_file=str(data_file),
        peft_config=LoraConfig(**preset["peft"]),
    )
    trainer.train()


if __name__ == "__main__":
    main()
