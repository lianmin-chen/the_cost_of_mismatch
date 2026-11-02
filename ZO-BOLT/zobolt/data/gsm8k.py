"""Build the fixed GSM8K training prompts and reference answers."""

from __future__ import annotations

import re

from datasets import Dataset, load_dataset


INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
EXAMPLE_QUESTION = "What is the largest single-digit prime number?"
EXAMPLE_ANSWER = (
    "9 is divisible by 3 and 8 is divisible by 2, but 7 is prime. "
    "The answer is \\boxed{7}."
)


def extract_answer(text: str) -> str | None:
    """Read the GSM8K final answer after the first ``####`` marker."""
    match = re.search(r"####\s*([^\r\n]+)", text)
    return match.group(1).replace(",", "").strip() if match else None


def build_prompt(question: str) -> str:
    """Render the fixed Qwen boxed few-shot conversation."""
    messages = (
        ("system", "You are a helpful assistant."),
        ("user", f"{EXAMPLE_QUESTION}\n{INSTRUCTION}"),
        ("assistant", EXAMPLE_ANSWER),
        ("user", f"{question}\n{INSTRUCTION}"),
    )
    return "".join(
        f"<|im_start|>{role}\n{content}<|im_end|>\n"
        for role, content in messages
    ) + "<|im_start|>assistant\n"


def build_dataset(limit: int | None = None) -> Dataset:
    """Load GSM8K train, drop missing answers, and return prompt/answer rows."""
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    raw = load_dataset("openai/gsm8k", "main", split="train")
    if limit is not None:
        raw = raw.select(range(min(limit, len(raw))))
    print(f">>> GSM8K raw dataset loaded: split=train, samples={len(raw)}")

    prompts, answers = [], []
    for row in raw:
        answer = extract_answer(row["answer"])
        if answer is not None:
            prompts.append(build_prompt(row["question"]))
            answers.append(answer)
    return Dataset.from_dict({"prompt": prompts, "answer": answers})
