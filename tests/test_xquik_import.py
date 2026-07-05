from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from xquik_import import convert_file, normalize_xquik_export


class XquikImportTests(unittest.TestCase):
    def test_normalize_xquik_export_uses_tweet_headers(self) -> None:
        raw_df = pd.DataFrame(
            {
                "Tweet Text": ["Great campaign", "  ", "Poor response"],
                "Username": ["ada", "bea", "cam"],
                "Tweet Created At": ["2026-07-01", "2026-07-02", "2026-07-03"],
            }
        )

        normalized = normalize_xquik_export(raw_df)

        self.assertEqual(normalized["text"].tolist(), ["Great campaign", "Poor response"])
        self.assertEqual(normalized["user"].tolist(), ["ada", "cam"])
        self.assertEqual(normalized["flag"].tolist(), ["NO_QUERY", "NO_QUERY"])

    def test_normalize_xquik_export_requires_text_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "Tweet Text"):
            normalize_xquik_export(pd.DataFrame({"Likes": [1]}))

    def test_convert_file_writes_sentiment140_style_csv_without_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "xquik.csv"
            output_path = Path(directory) / "sentiment140.csv"
            input_path.write_text(
                "Tweet Text,Username\nGreat launch,ada\n",
                encoding="utf-8",
            )

            convert_file(input_path, output_path)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                ",,,NO_QUERY,ada,Great launch\n",
            )


if __name__ == "__main__":
    unittest.main()
