from __future__ import annotations

import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import torch

from src.generate_from_title import (
    GenerationOutput,
    build_model_result,
    build_prompt,
    format_lyrics,
    generate_lyrics,
    normalize_genre,
    parse_args,
    render_html,
    resolve_title_and_genre,
    save_comparison,
)


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    eos_token_id = 99
    all_special_ids = [99]

    def __init__(self):
        self.prompts: list[str] = []

    def __call__(self, prompt, return_tensors):
        self.prompts.append(prompt)
        self.return_tensors = return_tensors
        return FakeBatch(
            input_ids=torch.tensor([[10, 11, 12]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )

    def decode(self, token_ids, skip_special_tokens):
        self.decoded_ids = token_ids.tolist()
        self.skip_special_tokens = skip_special_tokens
        return "new words only" if self.decoded_ids else ""


class FakeModel:
    def __init__(self, generated_suffix=None, loss=math.log(4)):
        self.generated_suffix = [20, 21, 22, 99] if generated_suffix is None else generated_suffix
        self.loss = loss
        self.score_calls = []

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[10, 11, 12, *self.generated_suffix]])

    def __call__(self, **kwargs):
        self.score_calls.append(kwargs)
        return SimpleNamespace(loss=torch.tensor(self.loss))


def args_namespace() -> Namespace:
    return Namespace(
        max_new_tokens=200,
        temperature=0.85,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
    )


def generation_output(raw_text: str, loss=math.log(4), perplexity=4.0, tokens=3) -> GenerationOutput:
    return GenerationOutput(
        raw_generation=raw_text,
        token_ids=list(range(tokens)),
        generation_metrics={
            "loss": loss,
            "perplexity": perplexity,
            "scored_token_count": tokens,
        },
    )


class GenerateFromTitleTests(unittest.TestCase):
    def test_genre_normalization_and_validation(self):
        cases = {
            "pop": "Pop",
            "HIP HOP": "Hip-Hop/Rap",
            "rap": "Hip-Hop/Rap",
            "r&b": "R&B",
            "RnB": "R&B",
            "rock": "Rock",
            "BALLAD": "Ballad",
        }
        for raw_value, expected in cases.items():
            with self.subTest(raw_value=raw_value):
                self.assertEqual(normalize_genre(raw_value), expected)

        with self.assertRaisesRegex(ValueError, "Unsupported genre"):
            normalize_genre("Jazz")

    def test_prompt_and_interactive_input_resolution(self):
        values = iter(["Midnight Rain", "rnb"])
        title, genre = resolve_title_and_genre(None, None, lambda _prompt: next(values))

        self.assertEqual(title, "Midnight Rain")
        self.assertEqual(genre, "R&B")
        self.assertEqual(
            build_prompt(title, genre),
            "Genre: R&B\nSong title: Midnight Rain\n\n[Verse 1]\n",
        )

        parsed = parse_args(["--title", "Test Song", "--genre", "pop"])
        self.assertEqual(resolve_title_and_genre(parsed.title, parsed.genre), ("Test Song", "Pop"))
        self.assertEqual(parsed.temperature, 0.6)
        self.assertEqual(parsed.top_k, 30)
        self.assertEqual(parsed.top_p, 0.85)
        self.assertEqual(parsed.repetition_penalty, 1.05)

    def test_generation_scores_only_new_tokens_and_trims_eos(self):
        tokenizer = FakeTokenizer()
        model = FakeModel()
        prompt = build_prompt("Test", "Pop")

        result = generate_lyrics(
            tokenizer,
            model,
            prompt,
            torch.device("cpu"),
            args_namespace(),
            score_generation=True,
        )

        self.assertEqual(result.raw_generation, "new words only")
        self.assertEqual(result.token_ids, [20, 21, 22])
        self.assertEqual(tokenizer.decoded_ids, [20, 21, 22])
        self.assertAlmostEqual(result.generation_metrics["loss"], math.log(4), places=6)
        self.assertAlmostEqual(result.generation_metrics["perplexity"], 4.0, places=5)
        self.assertEqual(result.generation_metrics["scored_token_count"], 3)

        score_call = model.score_calls[0]
        self.assertEqual(score_call["input_ids"].tolist(), [[10, 11, 12, 20, 21, 22]])
        self.assertEqual(score_call["attention_mask"].tolist(), [[1, 1, 1, 1, 1, 1]])
        self.assertEqual(score_call["labels"].tolist(), [[-100, -100, -100, 20, 21, 22]])

    def test_both_models_receive_identical_prompt(self):
        prompt = build_prompt("Same Song", "Rock")
        baseline_tokenizer = FakeTokenizer()
        fine_tuned_tokenizer = FakeTokenizer()
        baseline_model = FakeModel()
        fine_tuned_model = FakeModel()

        baseline_result = generate_lyrics(
            baseline_tokenizer,
            baseline_model,
            prompt,
            torch.device("cpu"),
            args_namespace(),
        )
        fine_tuned_result = generate_lyrics(
            fine_tuned_tokenizer,
            fine_tuned_model,
            prompt,
            torch.device("cpu"),
            args_namespace(),
        )

        self.assertEqual(baseline_tokenizer.prompts, [prompt])
        self.assertEqual(fine_tuned_tokenizer.prompts, [prompt])
        self.assertIsNone(baseline_result.generation_metrics)
        self.assertIsNone(fine_tuned_result.generation_metrics)
        self.assertEqual(baseline_model.score_calls, [])
        self.assertEqual(fine_tuned_model.score_calls, [])

    def test_empty_and_eos_only_generations_have_unavailable_metrics(self):
        for suffix in ([], [99], [99, 99]):
            with self.subTest(suffix=suffix):
                tokenizer = FakeTokenizer()
                model = FakeModel(generated_suffix=suffix)
                result = generate_lyrics(
                    tokenizer,
                    model,
                    build_prompt("Quiet", "Ballad"),
                    torch.device("cpu"),
                    args_namespace(),
                    score_generation=True,
                )

                self.assertEqual(result.raw_generation, "")
                self.assertEqual(result.token_ids, [])
                self.assertEqual(
                    result.generation_metrics,
                    {"loss": None, "perplexity": None, "scored_token_count": 0},
                )
                self.assertEqual(model.score_calls, [])

    def test_formatter_uses_exact_structure_and_repeats_chorus(self):
        source_lines = [f"line {index}" for index in range(18)]
        result = format_lyrics("\n".join(source_lines))
        sections = result["sections"]

        self.assertEqual(
            [section["name"] for section in sections],
            ["Verse 1", "Chorus", "Verse 2", "Chorus", "Bridge", "Outro"],
        )
        self.assertEqual(
            [len(section["lines"]) for section in sections],
            [4, 4, 4, 4, 2, 2],
        )
        self.assertEqual(sections[1]["lines"], sections[3]["lines"])

        consumed_lines = []
        for index in (0, 1, 2, 4, 5):
            consumed_lines.extend(sections[index]["lines"])
        self.assertEqual(consumed_lines, source_lines[:16])

    def test_formatter_distributes_words_when_fewer_than_16_lines(self):
        source_words = [f"word{index}" for index in range(35)]
        result = format_lyrics(" ".join(source_words))

        consumed_words = []
        for index in (0, 1, 2, 4, 5):
            for line in result["sections"][index]["lines"]:
                consumed_words.extend(line.split())

        self.assertEqual(consumed_words, source_words)
        self.assertEqual(result["sections"][1]["lines"], result["sections"][3]["lines"])

    def test_formatter_removes_labels_and_handles_short_empty_text(self):
        labelled = format_lyrics(
            "[Verse 1]\nfirst line\n[Chorus]\nsecond line\n"
            "Bridge:\nthird line\n[Outro]\nfourth line"
        )
        empty = format_lyrics("")
        short = format_lyrics("stay with me tonight")
        punctuation = format_lyrics("wait... <b>don't</b> go! & come back?")

        labelled_words = []
        for index in (0, 1, 2, 4, 5):
            for line in labelled["sections"][index]["lines"]:
                labelled_words.extend(line.split())
        self.assertEqual(
            labelled_words,
            ["first", "line", "second", "line", "third", "line", "fourth", "line"],
        )
        self.assertNotIn("[Verse 1]", labelled["formatted_lyrics"].splitlines()[1:])
        self.assertEqual([len(section["lines"]) for section in empty["sections"]], [4, 4, 4, 4, 2, 2])
        self.assertTrue(
            all(not line for section in empty["sections"] for line in section["lines"])
        )
        self.assertNotIn("...", empty["formatted_lyrics"])
        self.assertEqual(short["sections"][0]["lines"], ["stay", "with", "me", "tonight"])
        self.assertIn("<b>don't</b>", punctuation["raw_generation"])

    def test_html_is_responsive_and_escapes_content(self):
        baseline = build_model_result(
            "baseline <model>",
            generation_output("<script>alert(1)</script> & stay"),
        )
        fine_tuned = build_model_result(
            "fine & tuned",
            generation_output("<b>sing</b> together"),
        )
        rendered = render_html(
            "<Unsafe Title>",
            "<Rock>",
            baseline,
            fine_tuned,
        )

        self.assertIn("@media (max-width: 900px)", rendered)
        self.assertIn("&lt;Unsafe Title&gt;", rendered)
        self.assertIn("&lt;Rock&gt;", rendered)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)
        self.assertIn("Baseline GPT-2", rendered)
        self.assertIn("Fine-Tuned GPT-2", rendered)
        self.assertNotIn("Conditional perplexity", rendered)
        self.assertNotIn("Perplexity", rendered)
        self.assertNotIn("Generated:", rendered)
        self.assertNotIn("max_new_tokens", rendered)
        self.assertNotIn("word-counts", rendered)
        self.assertNotIn("<details", rendered)
        self.assertNotIn("View raw model output", rendered)
        self.assertNotIn("method-note", rendered)
        self.assertNotIn("subtitle", rendered)
        self.assertNotIn("GPT-2 Lyrics Experiment", rendered)
        self.assertNotIn("baseline &lt;model&gt;", rendered)
        self.assertNotIn("fine &amp; tuned", rendered)
        self.assertNotIn("<script>alert(1)</script>", rendered)
        self.assertNotIn("<b>sing</b>", rendered)

    def test_save_comparison_omits_metadata_and_perplexity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_comparison(
                title="Test Song",
                genre="Pop",
                prompt=build_prompt("Test Song", "Pop"),
                baseline_generation=generation_output(" ".join(f"base{index}" for index in range(80))),
                fine_tuned_generation=generation_output(" ".join(f"fine{index}" for index in range(80))),
                baseline_model="baseline",
                fine_tuned_model=Path("fine-tuned"),
                output_dir=Path(temp_dir),
            )

            self.assertEqual(set(paths), {"json", "markdown", "html"})
            self.assertTrue(all(path.exists() for path in paths.values()))
            self.assertTrue(all("Pop_Test_Song" in path.name for path in paths.values()))

            record = json.loads(paths["json"].read_text(encoding="utf-8"))
            markdown_text = paths["markdown"].read_text(encoding="utf-8")
            html_text = paths["html"].read_text(encoding="utf-8")
            self.assertEqual(record["genre"], "Pop")
            self.assertEqual(
                set(record),
                {"title", "genre", "prompt", "baseline_gpt2", "fine_tuned_gpt2"},
            )
            self.assertEqual(
                set(record["baseline_gpt2"]),
                {"model", "raw_generation", "formatted_lyrics", "sections"},
            )
            self.assertNotIn("generation_metrics", record["baseline_gpt2"])
            self.assertNotIn("generated_at", record)
            self.assertNotIn("generation_settings", record)
            self.assertNotIn("formatting_note", record)
            self.assertIn("**Genre:** Pop", markdown_text)
            self.assertNotIn("Perplexity", markdown_text)
            self.assertNotIn("post-processing", markdown_text)
            self.assertNotIn("Conditional perplexity", html_text)
            self.assertNotIn("Generated:", html_text)
            self.assertNotIn("max_new_tokens", html_text)
            self.assertNotIn("<details", html_text)
            self.assertNotIn("word-counts", html_text)
            self.assertIn("Pop", html_text)


if __name__ == "__main__":
    unittest.main()
