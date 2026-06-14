from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASELINE_MODEL = "openai-community/gpt2"
DEFAULT_FINE_TUNED_MODEL = Path("models") / "gpt2-lyrics"
GENRES = ("Pop", "Hip-Hop/Rap", "R&B", "Rock", "Ballad")
GENRE_ALIASES = {
    "pop": "Pop",
    "hiphoprap": "Hip-Hop/Rap",
    "hiphop": "Hip-Hop/Rap",
    "rap": "Hip-Hop/Rap",
    "rb": "R&B",
    "rnb": "R&B",
    "randb": "R&B",
    "rock": "Rock",
    "ballad": "Ballad",
}
UNIQUE_SECTION_SPECS = (
    ("Verse 1", "verse", 4),
    ("Chorus", "chorus", 4),
    ("Verse 2", "verse", 4),
    ("Bridge", "bridge", 2),
    ("Outro", "outro", 2),
)
UNIQUE_LINE_COUNT = sum(line_count for _, _, line_count in UNIQUE_SECTION_SPECS)
SECTION_LABEL_RE = re.compile(
    r"^(?:"
    r"(?:verse|chorus)(?:\s+[0-9ivx]+)?|"
    r"intro|outro|bridge|hook|refrain|interlude|breakdown|instrumental|"
    r"(?:pre|post)[ -]?chorus|final chorus"
    r")$",
    re.IGNORECASE,
)


@dataclass
class GenerationOutput:
    raw_generation: str
    token_ids: list[int]
    generation_metrics: dict[str, float | int | None] | None = None


def load_model(model_path: str | Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def normalize_genre(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "", value.strip().lower())
    if key not in GENRE_ALIASES:
        choices = ", ".join(GENRES)
        raise ValueError(f"Unsupported genre '{value}'. Choose one of: {choices}.")
    return GENRE_ALIASES[key]


def resolve_title_and_genre(
    title: str | None,
    genre: str | None,
    input_fn: Callable[[str], str] = input,
) -> tuple[str, str]:
    resolved_title = title.strip() if title else input_fn("Enter song title: ").strip()
    if not resolved_title:
        raise ValueError("Song title cannot be empty.")

    genre_prompt = f"Enter genre ({', '.join(GENRES)}): "
    raw_genre = genre.strip() if genre else input_fn(genre_prompt).strip()
    if not raw_genre:
        raise ValueError("Genre cannot be empty.")
    return resolved_title, normalize_genre(raw_genre)


def build_prompt(title: str, genre: str) -> str:
    return f"Genre: {genre}\nSong title: {title}\n\n[Verse 1]\n"


def trim_trailing_special_tokens(generated_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if tokenizer.eos_token_id is not None:
        special_ids.add(tokenizer.eos_token_id)

    end = generated_ids.shape[0]
    while end > 0 and int(generated_ids[end - 1]) in special_ids:
        end -= 1
    return generated_ids[:end]


@torch.no_grad()
def score_continuation(
    model,
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    generated_ids: torch.Tensor,
) -> dict[str, float | int | None]:
    scored_token_count = int(generated_ids.shape[0])
    if scored_token_count == 0:
        return {"loss": None, "perplexity": None, "scored_token_count": 0}

    continuation = generated_ids.unsqueeze(0)
    input_ids = torch.cat([prompt_ids, continuation], dim=1)
    continuation_mask = torch.ones_like(continuation)
    attention_mask = torch.cat([prompt_attention_mask, continuation_mask], dim=1)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = float(outputs.loss.detach().cpu())
    perplexity = math.exp(loss) if math.isfinite(loss) and loss < 709 else None
    return {
        "loss": loss if math.isfinite(loss) else None,
        "perplexity": perplexity,
        "scored_token_count": scored_token_count,
    }


@torch.no_grad()
def generate_lyrics(
    tokenizer,
    model,
    prompt: str,
    device: torch.device,
    args: argparse.Namespace,
    score_generation: bool = False,
) -> GenerationOutput:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = encoded["input_ids"].shape[1]
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
    generated_ids = trim_trailing_special_tokens(output_ids[0, prompt_length:], tokenizer)
    metrics = None
    if score_generation:
        metrics = score_continuation(
            model,
            encoded["input_ids"],
            encoded["attention_mask"],
            generated_ids,
        )
    raw_generation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return GenerationOutput(
        raw_generation=raw_generation,
        token_ids=[int(token_id) for token_id in generated_ids.detach().cpu().tolist()],
        generation_metrics=metrics,
    )


def normalize_generation(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("<|endoftext|>", "").strip()


def split_into_lines(words: list[str], line_count: int) -> list[str]:
    if line_count <= 0:
        return []

    base, remainder = divmod(len(words), line_count)
    lines: list[str] = []
    cursor = 0
    for line_index in range(line_count):
        size = base + (1 if line_index < remainder else 0)
        lines.append(" ".join(words[cursor : cursor + size]))
        cursor += size
    return lines


def _is_section_label(line: str) -> bool:
    candidate = line.strip().rstrip(":").strip()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1].strip()
    return SECTION_LABEL_RE.fullmatch(candidate) is not None


def _source_lines(raw_generation: str) -> list[str]:
    cleaned_lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in raw_generation.splitlines()
        if line.strip() and not _is_section_label(line)
    ]
    if len(cleaned_lines) >= UNIQUE_LINE_COUNT:
        return cleaned_lines[:UNIQUE_LINE_COUNT]

    words = " ".join(cleaned_lines).split()
    return split_into_lines(words, UNIQUE_LINE_COUNT)


def _make_section(
    name: str,
    section_type: str,
    lines: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "type": section_type,
        "lines": list(lines),
    }


def format_lyrics(raw_generation: str) -> dict[str, Any]:
    normalized = normalize_generation(raw_generation)
    source_lines = _source_lines(normalized)
    source_sections: list[dict[str, Any]] = []
    cursor = 0
    for name, section_type, line_count in UNIQUE_SECTION_SPECS:
        lines = source_lines[cursor : cursor + line_count]
        cursor += line_count
        source_sections.append(_make_section(name, section_type, lines))

    verse_1, chorus, verse_2, bridge, outro = source_sections
    repeated_chorus = _make_section("Chorus", "chorus", chorus["lines"])
    sections = [verse_1, chorus, verse_2, repeated_chorus, bridge, outro]

    formatted_lines: list[str] = []
    for section in sections:
        formatted_lines.append(f"[{section['name']}]")
        formatted_lines.extend(section["lines"])
        formatted_lines.append("")

    formatted_lyrics = "\n".join(formatted_lines).rstrip()
    return {
        "raw_generation": normalized,
        "formatted_lyrics": formatted_lyrics,
        "sections": sections,
    }


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned[:50] or "untitled"


def build_model_result(model_name: str, generation: GenerationOutput) -> dict[str, Any]:
    result = format_lyrics(generation.raw_generation)
    return {"model": model_name, **result}


def render_markdown(
    title: str,
    genre: str,
    baseline: dict[str, Any],
    fine_tuned: dict[str, Any],
) -> str:
    return "\n".join(
        [
            f"# Lyrics Generation Comparison: {title}",
            "",
            f"**Genre:** {genre}",
            "",
            "## Baseline GPT-2",
            "",
            baseline["formatted_lyrics"],
            "",
            "## Fine-Tuned GPT-2",
            "",
            fine_tuned["formatted_lyrics"],
            "",
        ]
    )


def _render_section_html(section: dict[str, Any]) -> str:
    lines = "\n".join(
        f'<div class="lyric-line">{"&nbsp;" if not line else html.escape(line)}</div>'
        for line in section["lines"]
    )
    return (
        f'<section class="lyric-section {html.escape(section["type"])}">'
        f'<h3>{html.escape(section["name"])}</h3>'
        f'<div class="section-lines">{lines}</div>'
        "</section>"
    )


def _render_model_card(label: str, result: dict[str, Any], accent_class: str) -> str:
    sections = "".join(_render_section_html(section) for section in result["sections"])
    return (
        f'<article class="model-card {accent_class}">\n'
        '  <header class="card-header">\n'
        f"    <h2>{html.escape(label)}</h2>\n"
        "  </header>\n"
        f'  <div class="lyrics">{sections}</div>\n'
        "</article>"
    )


def render_html(
    title: str,
    genre: str,
    baseline: dict[str, Any],
    fine_tuned: dict[str, Any],
) -> str:
    baseline_card = _render_model_card("Baseline GPT-2", baseline, "baseline")
    fine_tuned_card = _render_model_card("Fine-Tuned GPT-2", fine_tuned, "fine-tuned")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(genre)} Lyrics Comparison - {html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --background: #090b12;
      --panel: #121622;
      --panel-soft: #171c2a;
      --text: #f5f1e8;
      --muted: #a7afc0;
      --line: rgba(255, 255, 255, 0.1);
      --baseline: #58a6ff;
      --fine-tuned: #ff6ba8;
      --chorus: rgba(255, 196, 87, 0.12);
      --chorus-line: #ffc457;
      --bridge: rgba(154, 123, 255, 0.12);
      --bridge-line: #9a7bff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 15% 0%, rgba(88, 166, 255, 0.16), transparent 34rem),
        radial-gradient(circle at 85% 5%, rgba(255, 107, 168, 0.14), transparent 32rem),
        var(--background);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .page {{ width: min(1500px, calc(100% - 32px)); margin: 0 auto; padding: 48px 0 64px; }}
    .hero {{ text-align: center; margin-bottom: 32px; }}
    h1 {{ margin: 0; font-size: clamp(2rem, 5vw, 4.5rem); line-height: 1; }}
    .genre-pill {{
      display: inline-block;
      margin: 16px 0 0;
      padding: 7px 13px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--chorus-line);
      background: var(--chorus);
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }}
    .comparison {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; align-items: start; }}
    .model-card {{
      overflow: hidden;
      border: 1px solid var(--line);
      border-top: 4px solid var(--accent);
      border-radius: 20px;
      background: rgba(18, 22, 34, 0.92);
      box-shadow: 0 24px 70px rgba(0, 0, 0, 0.24);
    }}
    .model-card.baseline {{ --accent: var(--baseline); }}
    .model-card.fine-tuned {{ --accent: var(--fine-tuned); }}
    .card-header {{
      padding: 24px;
      border-bottom: 1px solid var(--line);
    }}
    .card-header h2 {{ margin: 0; font-size: clamp(1.1rem, 2vw, 1.45rem); }}
    .lyrics {{ padding: 24px; }}
    .lyric-section {{
      margin: 0 0 22px;
      padding: 18px;
      border-left: 3px solid var(--accent);
      border-radius: 0 12px 12px 0;
      background: var(--panel-soft);
    }}
    .lyric-section:last-child {{ margin-bottom: 0; }}
    .lyric-section.chorus {{ border-left-color: var(--chorus-line); background: var(--chorus); }}
    .lyric-section.bridge {{ border-left-color: var(--bridge-line); background: var(--bridge); }}
    .lyric-section h3 {{
      margin: 0 0 12px;
      color: var(--accent);
      font-size: 0.78rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }}
    .lyric-section.chorus h3 {{ color: var(--chorus-line); }}
    .lyric-section.bridge h3 {{ color: var(--bridge-line); }}
    .section-lines {{ display: grid; gap: 7px; }}
    .lyric-line {{ font-family: Georgia, "Times New Roman", serif; font-size: 1.02rem; line-height: 1.5; }}
    @media (max-width: 900px) {{
      .page {{ width: min(100% - 20px, 760px); padding-top: 28px; }}
      .comparison {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 520px) {{
      .lyrics, .card-header {{ padding: 18px; }}
      .lyric-section {{ padding: 15px; }}
      .lyric-line {{ font-size: 0.96rem; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <h1>{html.escape(title)}</h1>
      <p class="genre-pill">{html.escape(genre)}</p>
    </header>
    <section class="comparison" aria-label="Model lyrics comparison">
      {baseline_card}
      {fine_tuned_card}
    </section>
  </main>
</body>
</html>
"""


def save_comparison(
    title: str,
    genre: str,
    prompt: str,
    baseline_generation: GenerationOutput,
    fine_tuned_generation: GenerationOutput,
    baseline_model: str,
    fine_tuned_model: str | Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone()
    run_id = generated_at.strftime("%Y%m%d_%H%M%S")
    base_name = f"title_comparison_{safe_filename(genre)}_{safe_filename(title)}_{run_id}"

    baseline = build_model_result(str(baseline_model), baseline_generation)
    fine_tuned = build_model_result(str(fine_tuned_model), fine_tuned_generation)
    record = {
        "title": title,
        "genre": genre,
        "prompt": prompt,
        "baseline_gpt2": baseline,
        "fine_tuned_gpt2": fine_tuned,
    }

    paths = {
        "json": output_dir / f"{base_name}.json",
        "markdown": output_dir / f"{base_name}.md",
        "html": output_dir / f"{base_name}.html",
    }
    paths["json"].write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(title, genre, baseline, fine_tuned), encoding="utf-8")
    paths["html"].write_text(
        render_html(title, genre, baseline, fine_tuned),
        encoding="utf-8",
    )
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate genre-prompted lyrics from a title using baseline and fine-tuned GPT-2."
    )
    parser.add_argument("--title", default=None, help="Song title. If omitted, the script asks for input.")
    parser.add_argument(
        "--genre",
        default=None,
        help=f"Song genre. Choose one of: {', '.join(GENRES)}. If omitted, the script asks for input.",
    )
    parser.add_argument("--baseline_model", default=BASELINE_MODEL)
    parser.add_argument("--fine_tuned_model", type=Path, default=DEFAULT_FINE_TUNED_MODEL)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs") / "samples")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        title, genre = resolve_title_and_genre(args.title, args.genre)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not (args.fine_tuned_model / "config.json").exists():
        raise SystemExit(
            f"Fine-tuned model not found at {args.fine_tuned_model}. "
            "Run: conda run -n gpu311 python src/train.py"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = build_prompt(title, genre)
    print(f"Using device: {device}")
    print(f"Prompt:\n{prompt}")

    print("Loading baseline GPT-2...")
    baseline_tokenizer, baseline_model = load_model(args.baseline_model, device)
    baseline_generation = generate_lyrics(baseline_tokenizer, baseline_model, prompt, device, args)

    print("Loading fine-tuned GPT-2...")
    fine_tuned_tokenizer, fine_tuned_model = load_model(args.fine_tuned_model, device)
    fine_tuned_generation = generate_lyrics(fine_tuned_tokenizer, fine_tuned_model, prompt, device, args)

    saved_paths = save_comparison(
        title=title,
        genre=genre,
        prompt=prompt,
        baseline_generation=baseline_generation,
        fine_tuned_generation=fine_tuned_generation,
        baseline_model=args.baseline_model,
        fine_tuned_model=args.fine_tuned_model,
        output_dir=args.output_dir,
    )

    print(f"\nTitle: {title}")
    print(f"Genre: {genre}")
    print("\nBASELINE GPT-2")
    print(format_lyrics(baseline_generation.raw_generation)["formatted_lyrics"])
    print("\nFINE-TUNED GPT-2")
    print(format_lyrics(fine_tuned_generation.raw_generation)["formatted_lyrics"])
    print("\nSaved comparison files:")
    for output_type, path in saved_paths.items():
        print(f"{output_type.upper()}: {path}")


if __name__ == "__main__":
    main()
