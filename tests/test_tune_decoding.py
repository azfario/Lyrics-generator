from __future__ import annotations

import unittest

from src.tune_decoding import recommend_summary, render_html, summarize_records, text_signals


class TuneDecodingTests(unittest.TestCase):
    def test_text_signals_measure_diversity_and_repetition(self):
        signals = text_signals("love you love you love tonight")

        self.assertEqual(signals["word_count"], 6)
        self.assertAlmostEqual(signals["unique_word_ratio"], 3 / 6)
        self.assertGreater(signals["repeated_bigram_rate"], 0)

    def test_summary_and_recommendation_choose_lowest_mean_perplexity(self):
        records = [
            {"temperature": 0.6, "perplexity": 10.0, "unique_word_ratio": 0.8, "repeated_bigram_rate": 0.1, "word_count": 100},
            {"temperature": 0.6, "perplexity": 14.0, "unique_word_ratio": 0.9, "repeated_bigram_rate": 0.0, "word_count": 90},
            {"temperature": 0.8, "perplexity": 20.0, "unique_word_ratio": 0.95, "repeated_bigram_rate": 0.0, "word_count": 110},
        ]

        summaries = summarize_records(records)
        recommendation = recommend_summary(summaries)

        self.assertEqual(len(summaries), 2)
        self.assertEqual(recommendation["temperature"], 0.6)
        self.assertEqual(recommendation["mean_perplexity"], 12.0)

    def test_html_escapes_title_genre_and_samples(self):
        summaries = [
            {
                "temperature": 0.6,
                "runs": 1,
                "mean_perplexity": 10.0,
                "min_perplexity": 10.0,
                "max_perplexity": 10.0,
                "mean_unique_word_ratio": 0.9,
                "mean_repeated_bigram_rate": 0.0,
                "mean_word_count": 100,
            }
        ]
        records = [
            {
                "temperature": 0.6,
                "seed": 42,
                "perplexity": 10.0,
                "raw_generation": "<script>alert(1)</script>",
            }
        ]

        rendered = render_html("<Title>", "<Rock>", {}, summaries, summaries[0], records)

        self.assertIn("&lt;Title&gt;", rendered)
        self.assertIn("&lt;Rock&gt;", rendered)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)
        self.assertNotIn("<script>alert(1)</script>", rendered)


if __name__ == "__main__":
    unittest.main()
