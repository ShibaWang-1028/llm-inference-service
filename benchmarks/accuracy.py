"""Small accuracy check on a GSM8K subset, to prove quantization preserves quality.

Point it at the FP16 endpoint and the AWQ endpoint and compare the scores. We use
greedy decoding and exact-match on the final number.

Example:
    python -m benchmarks.accuracy --config vllm-awq --num-questions 40
"""

from __future__ import annotations

import argparse
import json
import re

import httpx

from benchmarks.common import RESULTS_DIR

SYSTEM = "You are a careful math solver."
INSTRUCTION = (
    "Solve step by step, then put the final numeric answer on the last line as '#### <number>'."
)


def gold_number(answer: str) -> str | None:
    m = re.search(r"####\s*([-\d,.]+)", answer)
    return _normalize(m.group(1)) if m else None


def extract_number(text: str) -> str | None:
    m = re.search(r"####\s*([-\d,.]+)", text)
    if m:
        return _normalize(m.group(1))
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return _normalize(nums[-1]) if nums else None


def _normalize(s: str) -> str:
    s = s.replace(",", "").strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def ask(client: httpx.Client, model: str, question: str, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{question}\n\n{INSTRUCTION}"},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    r = client.post("/v1/chat/completions", json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def load_gsm8k(n: int) -> list[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        out.append((row["question"], row["answer"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="GSM8K accuracy check against an OpenAI endpoint")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--config", required=True, help="label, e.g. vllm-awq")
    ap.add_argument("--num-questions", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    questions = load_gsm8k(args.num_questions)
    correct = 0
    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        for i, (q, gold) in enumerate(questions, 1):
            try:
                answer = ask(client, args.model, q, args.max_tokens)
            except Exception as e:
                print(f"  q{i}: request failed: {e}")
                continue
            pred = extract_number(answer)
            target = gold_number(gold)
            if pred is not None and pred == target:
                correct += 1
            print(f"  q{i}: pred={pred} gold={target} {'OK' if pred == target else 'x'}")

    n = len(questions)
    acc = correct / n if n else 0.0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"config": args.config, "accuracy": round(acc, 4), "correct": correct, "n": n}
    path = RESULTS_DIR / f"accuracy-{args.config}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n{args.config}: GSM8K accuracy {acc:.1%} ({correct}/{n})")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
