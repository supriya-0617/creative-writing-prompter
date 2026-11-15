"""Corpus loading, text preprocessing, and n-gram next-word prediction."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

CORPUS: dict[str, tuple[str, ...]] = {
    "fantasy": (
        "Mira found a silver key beneath the old oak tree. At midnight the key began to sing, and a door appeared in the hill.",
        "The young dragon guarded a garden of blue roses. Each evening a traveler brought a new story, and the dragon shared its fire.",
        "Beyond the northern mountains, a quiet village waited for spring. A small spell warmed every window before dawn.",
    ),
    "mystery": (
        "Detective Rowan noticed a muddy footprint beside the locked study. The clock had stopped, but a fresh candle still burned.",
        "A letter arrived with no name and a date from tomorrow. Inspector Vale followed its clues to the station at midnight.",
        "The museum was silent after closing. One glass case stood open, and a trail of pale sand crossed the floor.",
    ),
    "science-fiction": (
        "The last signal from Earth reached the ship as the red planet rose. Captain Imani opened the airlock and listened.",
        "A curious robot collected fallen leaves outside the colony. It planted them beside the reactor and waited for green.",
        "When the stars shifted, the navigation screen drew a path home. The crew followed the lights through the quiet dark.",
    ),
}

_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase text and return its words, dropping punctuation and symbols."""
    return _TOKEN_PATTERN.findall(text.lower())


def load_corpus(
    genre: str = "fantasy", corpus_path: str | Path | None = None
) -> list[str]:
    """Load a built-in genre or the contents of a custom UTF-8 text file."""
    if corpus_path is not None:
        path = Path(corpus_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Corpus file does not exist: {path}")
        text = path.read_text(encoding="utf-8")
        if not tokenize(text):
            raise ValueError(f"Corpus file contains no words: {path}")
        return [text]

    normalized_genre = genre.strip().lower()
    if normalized_genre not in CORPUS:
        available = ", ".join(sorted(CORPUS))
        raise ValueError(f"Unknown genre {genre!r}. Choose from: {available}")
    return list(CORPUS[normalized_genre])


class NGramModel:
    """An n-gram language model with MLE and add-k-smoothed probabilities."""

    def __init__(self, n: int = 2) -> None:
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.unigram_counts: Counter[str] = Counter()
        self.vocabulary: set[str] = set()

    def train(self, texts: Iterable[str]) -> NGramModel:
        """Count each observed context and its following word."""
        for text in texts:
            words = tokenize(text)
            self.unigram_counts.update(words)
            self.vocabulary.update(words)
            for index, word in enumerate(words):
                context = tuple(words[max(0, index - self.n + 1) : index])
                self.counts[context][word] += 1
        if not self.vocabulary:
            raise ValueError("Cannot train an n-gram model on an empty corpus")
        return self

    def _context(self, context: str | Iterable[str]) -> tuple[str, ...]:
        words = tokenize(context) if isinstance(context, str) else list(context)
        return tuple(words[-(self.n - 1) :]) if self.n > 1 else ()

    def distribution(
        self, context: str | Iterable[str], add_k: float = 0.0
    ) -> dict[str, float]:
        """Return next-word probabilities over the vocabulary for a context."""
        if add_k < 0:
            raise ValueError("add_k must be non-negative")
        if not self.vocabulary:
            raise ValueError("Train the model before requesting probabilities")

        counts = self.counts.get(self._context(context), Counter())
        total = sum(counts.values())
        vocabulary_size = len(self.vocabulary)
        denominator = total + add_k * vocabulary_size
        if denominator == 0:
            return {word: 0.0 for word in self.vocabulary}
        return {
            word: (counts[word] + add_k) / denominator
            for word in self.vocabulary
        }

    def probability(
        self, context: str | Iterable[str], word: str, add_k: float = 0.0
    ) -> float:
        """Return the MLE (add_k=0) or add-k-smoothed probability."""
        return self.distribution(context, add_k).get(word.lower(), 0.0)

    def suggest(
        self, context: str | Iterable[str], top_k: int = 5, add_k: float = 1.0
    ) -> list[tuple[str, float]]:
        """Return the highest-probability next words in descending order."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        distribution = self.distribution(context, add_k)
        return sorted(
            ((word, probability) for word, probability in distribution.items() if probability),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]

    def sample_next(
        self,
        context: str | Iterable[str],
        *,
        add_k: float = 1.0,
        rng: random.Random | None = None,
    ) -> str:
        """Sample a next word proportionally to its probability."""
        distribution = self.distribution(context, add_k)
        words = list(distribution)
        weights = list(distribution.values())
        if not words or not any(weights):
            raise ValueError("No next-word candidates; use add-k smoothing")
        return (rng or random).choices(words, weights=weights, k=1)[0]


def build_model(
    genre: str = "fantasy",
    n: int = 2,
    corpus_path: str | Path | None = None,
) -> NGramModel:
    """Load a corpus and train an n-gram model for it."""
    return NGramModel(n).train(load_corpus(genre, corpus_path))


def generate_text(
    model: NGramModel,
    prompt: str = "",
    word_count: int = 20,
    *,
    add_k: float = 1.0,
    seed: int | None = None,
) -> str:
    """Continue a prompt by weighted random sampling from the model."""
    if word_count < 0:
        raise ValueError("word_count must be non-negative")
    generated: list[str] = []
    context = tokenize(prompt)
    rng = random.Random(seed)
    for _ in range(word_count):
        word = model.sample_next(context, add_k=add_k, rng=rng)
        generated.append(word)
        context.append(word)
    continuation = " ".join(generated)
    return f"{prompt.rstrip()} {continuation}".strip()
