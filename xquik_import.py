from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TEXT_ALIASES = (
    "tweet text",
    "tweet_text",
    "tweettext",
    "text",
    "comment",
    "comments",
    "message",
    "body",
)
DATE_ALIASES = ("tweet created at", "tweet_created_at", "created at", "date")
USER_ALIASES = ("username", "user", "author", "screen name")


def normalize_column_name(column_name: object) -> str:
    return "".join(
        character
        for character in str(column_name).strip().lower()
        if character.isalnum()
    )


def find_column(columns: list[object], aliases: tuple[str, ...]) -> str | None:
    normalized_aliases = {normalize_column_name(alias) for alias in aliases}
    for column in columns:
        if normalize_column_name(column) in normalized_aliases:
            return str(column)
    return None


def normalize_xquik_export(raw_df: pd.DataFrame) -> pd.DataFrame:
    text_column = find_column(list(raw_df.columns), TEXT_ALIASES)
    if text_column is None:
        raise ValueError("CSV must include Tweet Text, text, comment, or message.")

    date_column = find_column(list(raw_df.columns), DATE_ALIASES)
    user_column = find_column(list(raw_df.columns), USER_ALIASES)
    normalized = pd.DataFrame(
        {
            "target": "",
            "id": "",
            "date": (
                raw_df[date_column].fillna("").astype(str).str.strip()
                if date_column
                else ""
            ),
            "flag": "NO_QUERY",
            "user": (
                raw_df[user_column].fillna("").astype(str).str.strip()
                if user_column
                else "xquik_export"
            ),
            "text": raw_df[text_column].fillna("").astype(str).str.strip(),
        }
    )
    normalized = normalized[normalized["text"] != ""].reset_index(drop=True)
    if normalized.empty:
        raise ValueError("CSV must include at least one non-empty text row.")
    return normalized


def convert_file(input_path: Path, output_path: Path) -> None:
    normalized = normalize_xquik_export(pd.read_csv(input_path))
    normalized.to_csv(output_path, index=False, header=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an Xquik export CSV into Sentiment140-style columns."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    convert_file(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
