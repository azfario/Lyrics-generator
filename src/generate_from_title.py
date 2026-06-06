from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASELINE_MODEL = "openai-community/gpt2"
DEFAULT_FINE_TUNED_MODEL = Path("models") / "gpt2-lyrics"


def load_model(model_path: str | Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def build_prompt(title: str) -> str:
    return f"Song title: {title}\n\n[Verse 1]\n"


@torch.no_grad()
def generate_lyrics(tokenizer, model, prompt: str, device: torch.device, args: argparse.Namespace) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **encoded,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned[:50] or "untitled"


def save_comparison(title: str, prompt: str, baseline_text: str, fine_tuned_text: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"title_comparison_{safe_filename(title)}_{run_id}"

    record = {
        "title": title,
        "prompt": prompt,
        "baseline_gpt2": baseline_text,
        "fine_tuned_gpt2": fine_tuned_text,
    }
    json_path = output_dir / f"{base_name}.json"
    md_path = output_dir / f"{base_name}.md"

    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                f"# Lyrics Generation Comparison: {title}",
                "",
                "## Baseline GPT-2",
                "",
                baseline_text,
                "",
                "## Fine-Tuned GPT-2",
                "",
                fine_tuned_text,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate lyrics from a user-provided song title using baseline and fine-tuned GPT-2."
    )
    parser.add_argument("--title", default=None, help="Song title. If omitted, the script asks for input.")
    parser.add_argument("--baseline_model", default=BASELINE_MODEL)
    parser.add_argument("--fine_tuned_model", type=Path, default=DEFAULT_FINE_TUNED_MODEL)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs") / "samples")
    parser.add_argument("--max_new_tokens", type=int, default=140)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    title = args.title or input("Enter song title: ").strip()
    if not title:
        raise SystemExit("Song title cannot be empty.")

    if not (args.fine_tuned_model / "config.json").exists():
        raise SystemExit(
            f"Fine-tuned model not found at {args.fine_tuned_model}. "
            "Run: conda run -n gpu311 python src/train.py"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = build_prompt(title)
    print(f"Using device: {device}")
    print(f"Prompt:\n{prompt}")

    print("Loading baseline GPT-2...")
    baseline_tokenizer, baseline_model = load_model(args.baseline_model, device)
    baseline_text = generate_lyrics(baseline_tokenizer, baseline_model, prompt, device, args)

    print("Loading fine-tuned GPT-2...")
    fine_tuned_tokenizer, fine_tuned_model = load_model(args.fine_tuned_model, device)
    fine_tuned_text = generate_lyrics(fine_tuned_tokenizer, fine_tuned_model, prompt, device, args)

    saved_path = save_comparison(title, prompt, baseline_text, fine_tuned_text, args.output_dir)

    print("\n" + "=" * 80)
    print("BASELINE GPT-2")
    print("=" * 80)
    print(baseline_text)
    print("\n" + "=" * 80)
    print("FINE-TUNED GPT-2")
    print("=" * 80)
    print(fine_tuned_text)
    print("\nSaved comparison to:")
    print(saved_path)


if __name__ == "__main__":
    main()
