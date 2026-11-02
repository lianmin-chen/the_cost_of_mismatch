"""Generate the fixed GSM8K responses used by both offline BOLT trainers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zobolt.data.gsm8k import build_dataset
from zobolt.rewards.math_grader import qwen_boxed_accuracy


def truncate_after_first_boxed(text: str) -> str:
    """Keep the first complete boxed answer, including nested braces."""
    prefix = r"\boxed{"
    start = text.find(prefix)
    if start < 0:
        return text
    depth = 0
    for index in range(start + len(prefix) - 1, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[: index + 1]
    return text


def filter_all_wrong_samples(rows: list[dict]) -> tuple[list[dict], int]:
    """Keep the original rule: discard a question only when every answer scores zero."""
    kept = []
    for row in rows:
        answers = row["generated_texts"]
        scores = qwen_boxed_accuracy(
            prompts=[row["prompt"]] * len(answers),
            completions=answers,
            answer=[row["ground_truth_answer"]] * len(answers),
        )
        if any(float(score) > 0.5 for score in scores):
            kept.append(row)
    return kept, len(rows) - len(kept)


def save_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B", help="Model used to generate the fixed responses")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for raw, filtered, and truncated JSON files")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on GSM8K training questions")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams

    dataset = build_dataset(limit=args.limit)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
    )
    sampling = SamplingParams(
        n=8, temperature=0.8, top_p=0.95, top_k=-1, min_p=0.0, max_tokens=1024
    )
    outputs = llm.generate(list(dataset["prompt"]), sampling)
    raw = [
        {
            "prompt": output.prompt,
            "generated_texts": [item.text for item in output.outputs],
            "ground_truth_answer": dataset[i]["answer"],
        }
        for i, output in enumerate(outputs)
    ]
    save_json(output_dir / "inference_results.json", raw)
    filtered, removed = filter_all_wrong_samples(raw)
    save_json(output_dir / "inference_results_filtered.json", filtered)
    for row in filtered:
        row["generated_texts"] = [
            truncate_after_first_boxed(text) for text in row["generated_texts"]
        ]
    final_path = output_dir / "inference_results_filtered_truncated.json"
    save_json(final_path, filtered)
    print(f"Filtered: {len(filtered)}/{len(raw)} questions; removed {removed} all-wrong questions")
    print(f"Training data: {final_path}")


if __name__ == "__main__":
    main()
