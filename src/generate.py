from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASELINE_MODEL = "openai-community/gpt2"
DEFAULT_PROMPTS = [
    "I miss the rain",
    "A sad song about friendship",
    "Dancing under city lights",
    "I found hope in the dark",
    "Goodbye but I still remember",
    "A happy song about starting over",
]


def load_model(model_path: str | Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_text(tokenizer, model, prompt: str, device: torch.device, config: dict) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **encoded,
        max_new_tokens=config.get("max_new_tokens", 120),
        do_sample=True,
        temperature=config.get("temperature", 0.9),
        top_k=config.get("top_k", 0),
        top_p=config.get("top_p", 1.0),
        repetition_penalty=config.get("repetition_penalty", 1.15),
        no_repeat_ngram_size=config.get("no_repeat_ngram_size", 3),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def write_outputs(records: list[dict], output_dir: Path, run_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"generated_samples_{run_id}.json"
    csv_path = output_dir / f"generated_samples_{run_id}.csv"
    md_path = output_dir / f"generated_samples_{run_id}.md"

    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_id", "model_type", "decoding", "prompt", "generated_text"],
        )
        writer.writeheader()
        writer.writerows(records)

    lines = ["# Generated Lyrics Samples", ""]
    for record in records:
        lines.extend(
            [
                f"## {record['model_type']} - {record['decoding']}",
                f"Prompt: `{record['prompt']}`",
                "",
                record["generated_text"],
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {len(records)} samples to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lyrics from baseline and fine-tuned GPT-2.")
    parser.add_argument("--baseline_model", default=BASELINE_MODEL)
    parser.add_argument("--model_dir", type=Path, default=Path("models") / "gpt2-lyrics")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs") / "samples")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--prompts_file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.prompts_file and args.prompts_file.exists():
        prompts = [line.strip() for line in args.prompts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    decoding_configs = [
        (
            "baseline",
            "balanced_sampling",
            args.baseline_model,
            {
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 0.92,
                "max_new_tokens": args.max_new_tokens,
            },
        )
    ]

    if args.model_dir.exists() and (args.model_dir / "config.json").exists():
        decoding_configs.extend(
            [
                (
                    "fine_tuned",
                    "temperature",
                    args.model_dir,
                    {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "max_new_tokens": args.max_new_tokens},
                ),
                (
                    "fine_tuned",
                    "top_k",
                    args.model_dir,
                    {"temperature": 0.8, "top_k": 50, "top_p": 1.0, "max_new_tokens": args.max_new_tokens},
                ),
                (
                    "fine_tuned",
                    "top_p",
                    args.model_dir,
                    {"temperature": 0.85, "top_k": 0, "top_p": 0.9, "max_new_tokens": args.max_new_tokens},
                ),
            ]
        )
    else:
        print(f"Fine-tuned model not found at {args.model_dir}; generating baseline samples only.")

    records: list[dict] = []
    loaded_models: dict[str, tuple] = {}

    for model_type, decoding_name, model_path, config in decoding_configs:
        model_key = str(model_path)
        if model_key not in loaded_models:
            loaded_models[model_key] = load_model(model_path, device)
        tokenizer, model = loaded_models[model_key]

        for prompt in prompts:
            generated = generate_text(tokenizer, model, prompt, device, config)
            records.append(
                {
                    "run_id": run_id,
                    "model_type": model_type,
                    "decoding": decoding_name,
                    "prompt": prompt,
                    "generated_text": generated,
                }
            )

    write_outputs(records, args.output_dir, run_id)


if __name__ == "__main__":
    main()
