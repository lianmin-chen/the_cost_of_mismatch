"""Evaluate a merged checkpoint on the fixed GSM8K boxed-answer benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from datasets import load_from_disk

from zobolt.data.gsm8k import build_prompt
from zobolt.rewards.math_grader import boxed_reward_fn


def shutdown_model(model) -> None:
    """Stop the vLLM engine so the evaluation subprocess exits cleanly."""
    engine_core = getattr(getattr(model, "llm_engine", None), "engine_core", None)
    if engine_core is not None:
        engine_core.shutdown()


def evaluate_model(model_name: str, output_dir: Path) -> None:
    """Generate one greedy answer per question and save scores and completions."""
    from vllm import LLM, SamplingParams

    data_root = Path(
        os.environ.get("ZOBOLT_EVAL_DATA_DIR", Path(__file__).resolve().parent / "processed_datasets")
    ).expanduser()
    data_path = data_root / "gsm8k_eval"
    if not data_path.is_dir():
        raise FileNotFoundError(f"GSM8K evaluation dataset not found: {data_path}")
    dataset = load_from_disk(str(data_path))
    prompts = [build_prompt(question) for question in dataset["prompts"][:999999]]
    answers = list(dataset["answers"][:999999])

    sampling = SamplingParams(
        n=1, temperature=0.0, top_p=0.8, max_tokens=1024, seed=int(time.time_ns())
    )
    model = LLM(
        model_name,
        swap_space=32,
        max_model_len=4096,
        tokenizer_mode="auto",
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
    )
    try:
        outputs = model.generate(prompts, sampling)
        scores, lengths, formatted_counts, samples = [], [], [], []
        for index, output in enumerate(outputs):
            gt_repeated = [answers[index]] * sampling.n
            rewards, infos = [], []
            for completion, ground_truth in zip(output.outputs, gt_repeated):
                info, reward = boxed_reward_fn(completion.text, ground_truth, fast=False)
                infos.append(info)
                rewards.append(float(reward))
            scores.append(float(np.mean(rewards)))
            lengths.append([len(completion.token_ids) for completion in output.outputs])
            if infos:
                formatted_counts.append(int(sum(info["formatted"] for info in infos)))
            samples.append(
                {
                    "task_name": "gsm8k_eval",
                    "prompt": output.prompt,
                    "gt": gt_repeated,
                    "model_output": [completion.text for completion in output.outputs],
                    "reward": rewards,
                }
            )

        score = float(np.mean(scores)) if scores else 0.0
        summary = {
            "config": {
                "model_name": model_name,
                "tasks": ["gsm8k_eval"],
                "template": "qwen_boxed_fs",
                "temperature": 0.0,
                "top_p": 0.8,
                "max_tokens": 1024,
                "max_model_len": 4096,
                "tokenizer_mode": "auto",
                "n_samples": 1,
                "max_test": 999999,
            },
            "summary": {
                "results": {"gsm8k_eval": score},
                "avg": score,
                "avg_lens": {"gsm8k_eval": float(np.mean(lengths)) if lengths else 0.0},
                "max_lens": {"gsm8k_eval": int(np.max(lengths)) if lengths else 0},
                "formatted": {
                    "gsm8k_eval": float(np.mean(formatted_counts)) if formatted_counts else 0.0
                },
            },
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=4), encoding="utf-8")
        (output_dir / "samples.json").write_text(json.dumps(samples, indent=4), encoding="utf-8")
        print(f"gsm8k_eval accuracy: {score:.6f}; results: {output_dir}")
    finally:
        shutdown_model(model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True, help="Merged checkpoint model directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Evaluation result directory")
    args = parser.parse_args()
    evaluate_model(args.model_name, args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
