#!/usr/bin/env python3

"""Calculate total cost of LLM API usage from cache metadata."""

import argparse
import json
from pathlib import Path


def calculate_total_cost(
    cache_dir: Path,
    prompt_token_price_per_1m: float,
    completion_token_price_per_1m: float,
) -> tuple[float, int, int]:
    """Calculate total cost and token usage from cache metadata.

    Args:
        cache_dir: Path to LLM cache directory
        prompt_token_price_per_1m: Price per 1M prompt tokens in dollars
        completion_token_price_per_1m: Price per 1M completion tokens in dollars

    Returns:
        Tuple of (total_cost, total_prompt_tokens, total_completion_tokens)
    """
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Walk through all metadata.json files
    for metadata_file in cache_dir.rglob("metadata.json"):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Extract token counts
            prompt_tokens = metadata.get("prompt_tokens", 0)
            completion_tokens = metadata.get("completion_tokens", 0)

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not process {metadata_file}: {e}")
            continue

    # Calculate total cost (convert from price per 1M tokens to price per token)
    total_cost = (total_prompt_tokens * prompt_token_price_per_1m / 1_000_000) + (
        total_completion_tokens * completion_token_price_per_1m / 1_000_000
    )

    return total_cost, total_prompt_tokens, total_completion_tokens


def _main() -> None:
    parser = argparse.ArgumentParser(description="Calculate LLM API costs from cache")
    parser.add_argument(
        "cache_dir",
        type=Path,
        help="Path to LLM cache directory",
    )
    parser.add_argument(
        "prompt_token_price_per_1m",
        type=float,
        help="Price per 1M prompt tokens in dollars",
    )
    parser.add_argument(
        "completion_token_price_per_1m",
        type=float,
        help="Price per 1M completion tokens in dollars",
    )

    args = parser.parse_args()

    total_cost, total_prompt_tokens, total_completion_tokens = calculate_total_cost(
        args.cache_dir,
        args.prompt_token_price_per_1m,
        args.completion_token_price_per_1m,
    )

    print("\nLLM API Usage Summary:")
    print(f"Total Prompt Tokens: {total_prompt_tokens:,}")
    print(f"Total Completion Tokens: {total_completion_tokens:,}")
    print(f"Total Cost: ${total_cost:.5f}")


if __name__ == "__main__":
    _main()
