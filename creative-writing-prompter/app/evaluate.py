"""Evaluate the built-in n-gram models against a unigram baseline."""

from __future__ import annotations

import argparse
from math import exp, fsum, isfinite, log

from app.core import CORPUS, NGramModel, load_corpus, tokenize

UNKNOWN_TOKEN = "<unk>"


def _perplexity(model: NGramModel, tokens: list[str], add_k: float) -> float:
    mapped_tokens = [
        token if token in model.vocabulary else UNKNOWN_TOKEN for token in tokens
    ]
    negative_log_probabilities = [
        -log(model.probability(mapped_tokens[:index], token, add_k))
        for index, token in enumerate(mapped_tokens)
    ]
    return exp(fsum(negative_log_probabilities) / len(mapped_tokens))


def evaluate_genre(
    genre: str, *, n: int = 3, add_k: float = 1.0
) -> dict[str, int | float]:
    """Train on the first two genre stories and score the final held-out story."""
    if n < 2:
        raise ValueError("n must be at least 2 to compare against a unigram baseline")
    if not isfinite(add_k) or add_k <= 0:
        raise ValueError("add_k must be a finite positive number")

    stories = load_corpus(genre)
    if len(stories) < 2:
        raise ValueError(f"Genre {genre!r} needs at least two stories")

    training_stories, test_story = stories[:-1], stories[-1]
    test_tokens = tokenize(test_story)
    ngram_model = NGramModel(n).train(training_stories)
    unigram_model = NGramModel(1).train(training_stories)
    unknown_count = sum(
        token not in ngram_model.vocabulary for token in test_tokens
    )
    ngram_model.vocabulary.add(UNKNOWN_TOKEN)
    unigram_model.vocabulary.add(UNKNOWN_TOKEN)

    return {
        "genre": genre,
        "training_stories": len(training_stories),
        "test_tokens": len(test_tokens),
        "unknown_tokens": unknown_count,
        "ngram_perplexity": _perplexity(ngram_model, test_tokens, add_k),
        "unigram_perplexity": _perplexity(unigram_model, test_tokens, add_k),
    }


def _format_report(results: list[dict[str, int | float]], n: int) -> str:
    lines = [
        f"Evaluation: {n}-gram vs. unigram baseline (add-k smoothing)",
        "Split: first two built-in stories train; final story held out per genre.",
        "",
        f"{'Genre':<18} {'Train':>5} {'Test tokens':>11} {'Unknown':>8} "
        f"{f'{n}-gram PPL':>12} {'Unigram PPL':>12}",
    ]
    total_tokens = 0
    ngram_log_loss: list[float] = []
    unigram_log_loss: list[float] = []
    for result in results:
        tokens = int(result["test_tokens"])
        total_tokens += tokens
        ngram_log_loss.append(tokens * log(float(result["ngram_perplexity"])))
        unigram_log_loss.append(tokens * log(float(result["unigram_perplexity"])))
        lines.append(
            f"{str(result['genre']):<18} {int(result['training_stories']):>5} "
            f"{tokens:>11} {int(result['unknown_tokens']):>8} "
            f"{float(result['ngram_perplexity']):>12.3f} "
            f"{float(result['unigram_perplexity']):>12.3f}"
        )

    lines.append(
        f"{'Overall':<18} {'':>5} {total_tokens:>11} "
        f"{sum(int(result['unknown_tokens']) for result in results):>8} "
        f"{exp(fsum(ngram_log_loss) / total_tokens):>12.3f} "
        f"{exp(fsum(unigram_log_loss) / total_tokens):>12.3f}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare an n-gram model with a unigram baseline."
    )
    parser.add_argument(
        "--genre",
        choices=["all", *sorted(CORPUS)],
        default="all",
        help="genre to evaluate (default: all)",
    )
    parser.add_argument(
        "--n", type=int, default=3, help="n-gram size, at least 2 (default: 3)"
    )
    parser.add_argument(
        "--add-k",
        type=float,
        default=1.0,
        help="positive add-k smoothing value (default: 1.0)",
    )
    args = parser.parse_args(argv)
    if args.n < 2:
        parser.error("--n must be at least 2")
    if not isfinite(args.add_k) or args.add_k <= 0:
        parser.error("--add-k must be a finite positive number")

    genres = sorted(CORPUS) if args.genre == "all" else [args.genre]
    try:
        results = [evaluate_genre(genre, n=args.n, add_k=args.add_k) for genre in genres]
    except ValueError as error:
        parser.error(str(error))
    print(_format_report(results, args.n))


if __name__ == "__main__":
    main()
