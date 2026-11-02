"""Evaluate GSM8K checkpoints with the existing vLLM boxed-answer evaluator."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def list_checkpoints(root: Path) -> list[Path]:
    """Select numeric Trainer checkpoints in training-step order."""
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.startswith("checkpoint-") and path.name[11:].isdigit()),
        key=lambda path: int(path.name[11:]),
    )


def evaluate_checkpoint(checkpoint: Path, model_name: str, output_dir: Path) -> dict:
    """Merge the LoRA adapter temporarily and run one isolated vLLM process."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    with tempfile.TemporaryDirectory(prefix="merged-model-", dir=output_dir) as temp:
        merged_dir = Path(temp)
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        adapter = PeftModel.from_pretrained(base_model, str(checkpoint))
        merged = adapter.merge_and_unload()
        merged.save_pretrained(merged_dir)
        AutoTokenizer.from_pretrained(model_name).save_pretrained(merged_dir)
        del merged, adapter, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result_dir = output_dir / checkpoint.name
        result_dir.mkdir()
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "zobolt/eval/evaluate_model.py"),
                "--model_name", str(merged_dir),
                "--output_dir", str(result_dir),
            ],
            check=True,
            cwd=result_dir,
        )

    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))["summary"]
    return {
        "checkpoint": checkpoint.name,
        "step": int(checkpoint.name[11:]),
        "score": summary["results"]["gsm8k_eval"],
        "avg_len": summary["avg_lens"].get("gsm8k_eval"),
        "max_len": summary["max_lens"].get("gsm8k_eval"),
        "formatted": summary["formatted"].get("gsm8k_eval"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Training output containing checkpoint-N directories")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for evaluation results")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B", help="Base model used for training")
    parser.add_argument("--checkpoint", default=None, help="Optional single checkpoint directory name")
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve(strict=True)
    if not checkpoint_dir.is_dir():
        parser.error("--checkpoint-dir must be a directory")
    checkpoints = list_checkpoints(checkpoint_dir)
    if args.checkpoint:
        checkpoints = [path for path in checkpoints if path.name == args.checkpoint]
    if not checkpoints:
        parser.error("No matching checkpoints found")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [evaluate_checkpoint(path, args.model, output_dir) for path in checkpoints]
    with (output_dir / "gsm8k_eval.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} checkpoint results to {output_dir / 'gsm8k_eval.csv'}")


if __name__ == "__main__":
    main()
