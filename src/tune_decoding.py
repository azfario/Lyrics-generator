from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

try:
    from src.generate_from_title import (
        DEFAULT_FINE_TUNED_MODEL,
        GENRES,
        build_prompt,
        generate_lyrics,
        load_model,
        resolve_title_and_genre,
        safe_filename,
    )
except ModuleNotFoundError:
    from generate_from_title import (
        DEFAULT_FINE_TUNED_MODEL,
        GENRES,
        build_prompt,
        generate_lyrics,
        load_model,
        resolve_title_and_genre,
        safe_filename,
    )


DEFAULT_TEMPERATURES = (0.6, 0.7, 0.8)
DEFAULT_SEEDS = (42, 43, 44)


def text_signals(text: str) -> dict[str, float | int]:
    words = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())
    bigrams = list(zip(words, words[1:]))
    unique_word_ratio = len(set(words)) / len(words) if words else 0.0
    repeated_bigram_rate = 1.0 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    return {
        "word_count": len(words),
        "unique_word_ratio": unique_word_ratio,
        "repeated_bigram_rate": repeated_bigram_rate,
    }


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for temperature in sorted({float(record["temperature"]) for record in records}):
        group = [record for record in records if float(record["temperature"]) == temperature]
        perplexities = [float(record["perplexity"]) for record in group if record["perplexity"] is not None]
        summaries.append(
            {
                "temperature": temperature,
                "runs": len(group),
                "mean_perplexity": statistics.mean(perplexities) if perplexities else None,
                "min_perplexity": min(perplexities) if perplexities else None,
                "max_perplexity": max(perplexities) if perplexities else None,
                "mean_unique_word_ratio": statistics.mean(
                    float(record["unique_word_ratio"]) for record in group
                ),
                "mean_repeated_bigram_rate": statistics.mean(
                    float(record["repeated_bigram_rate"]) for record in group
                ),
                "mean_word_count": statistics.mean(float(record["word_count"]) for record in group),
            }
        )
    return summaries


def recommend_summary(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    available = [summary for summary in summaries if summary["mean_perplexity"] is not None]
    if not available:
        return None
    return min(available, key=lambda summary: float(summary["mean_perplexity"]))


def render_html(
    title: str,
    genre: str,
    settings: dict[str, Any],
    summaries: list[dict[str, Any]],
    recommendation: dict[str, Any] | None,
    records: list[dict[str, Any]],
) -> str:
    rows = []
    for summary in summaries:
        mean_ppl = (
            f"{summary['mean_perplexity']:.2f}"
            if summary["mean_perplexity"] is not None
            else "N/A"
        )
        recommended = (
            recommendation is not None
            and summary["temperature"] == recommendation["temperature"]
        )
        rows.append(
            "<tr class=\"recommended\">" if recommended else "<tr>"
        )
        rows.append(
            f"<td>{summary['temperature']:.2f}</td>"
            f"<td>{summary['runs']}</td>"
            f"<td>{mean_ppl}</td>"
            f"<td>{summary['mean_unique_word_ratio']:.3f}</td>"
            f"<td>{summary['mean_repeated_bigram_rate']:.3f}</td>"
            f"<td>{summary['mean_word_count']:.1f}</td>"
            "</tr>"
        )

    samples = []
    for record in records:
        samples.append(
            "<details>"
            f"<summary>T={record['temperature']:.2f}, seed={record['seed']}, "
            f"perplexity={record['perplexity']:.2f}</summary>"
            f"<pre>{html.escape(record['raw_generation'])}</pre>"
            "</details>"
        )

    recommendation_text = (
        f"Temperature {recommendation['temperature']:.2f} had the lowest mean "
        f"conditional perplexity ({recommendation['mean_perplexity']:.2f})."
        if recommendation
        else "No recommendation was available."
    )
    settings_text = " | ".join(f"{key}: {value}" for key, value in settings.items())
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Decoding Sweep - {html.escape(title)}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#090b12; --panel:#151a28; --text:#f5f1e8; --muted:#a7afc0; --line:#303749; --accent:#ff6ba8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter,system-ui,sans-serif; }}
    main {{ width:min(1100px,calc(100% - 28px)); margin:auto; padding:42px 0 64px; }}
    h1 {{ margin:0 0 8px; font-size:clamp(2rem,5vw,4rem); }}
    .meta,.note {{ color:var(--muted); line-height:1.6; }}
    .recommendation {{ margin:24px 0; padding:18px; border-left:4px solid var(--accent); background:var(--panel); border-radius:0 12px 12px 0; }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:14px; }}
    table {{ width:100%; border-collapse:collapse; min-width:760px; }}
    th,td {{ padding:14px; text-align:right; border-bottom:1px solid var(--line); }}
    th:first-child,td:first-child {{ text-align:left; }}
    th {{ color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; }}
    tr.recommended {{ background:rgba(255,107,168,.11); }}
    h2 {{ margin-top:36px; }}
    details {{ margin:10px 0; border:1px solid var(--line); border-radius:10px; background:var(--panel); }}
    summary {{ cursor:pointer; padding:14px; }}
    pre {{ margin:0; padding:0 14px 16px; white-space:pre-wrap; overflow-wrap:anywhere; color:#d2d8e5; }}
  </style>
</head>
<body>
  <main>
    <p class="meta">{html.escape(genre)} decoding experiment</p>
    <h1>{html.escape(title)}</h1>
    <p class="meta">{html.escape(settings_text)}</p>
    <div class="recommendation"><strong>Recommendation:</strong> {html.escape(recommendation_text)}</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Temperature</th><th>Runs</th><th>Mean perplexity</th><th>Unique-word ratio</th><th>Repeated-bigram rate</th><th>Mean words</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    <p class="note">Lower perplexity means the model was more confident in its sampled continuation. Diversity and repetition indicators are included because the lowest-perplexity output is not automatically the best lyric.</p>
    <h2>Generated samples</h2>
    {''.join(samples)}
  </main>
</body>
</html>
"""


def write_outputs(
    title: str,
    genre: str,
    settings: dict[str, Any],
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    recommendation: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    base_name = f"decoding_sweep_{safe_filename(genre)}_{safe_filename(title)}_{run_id}"
    paths = {
        "json": output_dir / f"{base_name}.json",
        "csv": output_dir / f"{base_name}.csv",
        "html": output_dir / f"{base_name}.html",
    }
    payload = {
        "title": title,
        "genre": genre,
        "settings": settings,
        "summaries": summaries,
        "recommendation": recommendation,
        "runs": records,
    }
    paths["json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    paths["html"].write_text(
        render_html(title, genre, settings, summaries, recommendation, records),
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep fine-tuned GPT-2 decoding temperatures.")
    parser.add_argument("--title", default=None)
    parser.add_argument("--genre", default=None, help=f"One of: {', '.join(GENRES)}")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_FINE_TUNED_MODEL)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs") / "decoding_sweep")
    parser.add_argument("--temperatures", type=float, nargs="+", default=list(DEFAULT_TEMPERATURES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        title, genre = resolve_title_and_genre(args.title, args.genre)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not (args.model_dir / "config.json").exists():
        raise SystemExit(f"Fine-tuned model not found at {args.model_dir}.")
    if any(temperature <= 0 for temperature in args.temperatures):
        raise SystemExit("Temperatures must be greater than zero.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(args.model_dir, device)
    prompt = build_prompt(title, genre)
    records: list[dict[str, Any]] = []

    for temperature in args.temperatures:
        for seed in args.seeds:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            generation_args = SimpleNamespace(
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            generated = generate_lyrics(
                tokenizer,
                model,
                prompt,
                device,
                generation_args,
                score_generation=True,
            )
            signals = text_signals(generated.raw_generation)
            records.append(
                {
                    "temperature": temperature,
                    "seed": seed,
                    "loss": generated.generation_metrics["loss"],
                    "perplexity": generated.generation_metrics["perplexity"],
                    "scored_token_count": generated.generation_metrics["scored_token_count"],
                    **signals,
                    "raw_generation": generated.raw_generation,
                }
            )

    summaries = summarize_records(records)
    recommendation = recommend_summary(summaries)
    settings = {
        "model": str(args.model_dir),
        "temperatures": args.temperatures,
        "seeds": args.seeds,
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }
    paths = write_outputs(
        title,
        genre,
        settings,
        records,
        summaries,
        recommendation,
        args.output_dir,
    )

    print(f"Using device: {device}")
    for summary in summaries:
        print(
            f"T={summary['temperature']:.2f} "
            f"mean perplexity={summary['mean_perplexity']:.2f} "
            f"unique words={summary['mean_unique_word_ratio']:.3f} "
            f"repeated bigrams={summary['mean_repeated_bigram_rate']:.3f}"
        )
    if recommendation:
        print(
            f"Recommended temperature: {recommendation['temperature']:.2f} "
            f"(mean perplexity {recommendation['mean_perplexity']:.2f})"
        )
    for output_type, path in paths.items():
        print(f"{output_type.upper()}: {path}")


if __name__ == "__main__":
    main()
