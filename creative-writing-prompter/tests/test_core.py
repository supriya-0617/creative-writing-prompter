import random

import pytest

from app.core import (
    CORPUS,
    NGramModel,
    build_model,
    generate_text,
    load_corpus,
    tokenize,
)


def test_builtin_corpus_has_three_genres_and_loads_each():
    assert len(CORPUS) >= 3
    for genre in CORPUS:
        assert load_corpus(genre)


def test_custom_corpus_file_loads(tmp_path):
    corpus_file = tmp_path / "stories.txt"
    corpus_file.write_text("The moon glowed softly.", encoding="utf-8")

    assert load_corpus(corpus_path=corpus_file) == ["The moon glowed softly."]
    model = build_model(corpus_path=corpus_file)
    assert model.suggest("the", top_k=1, add_k=0) == [("moon", 1.0)]


def test_tokenize_lowercases_and_removes_punctuation():
    assert tokenize("Hello, WORLD! It's a tale.") == ["hello", "world", "it's", "a", "tale"]


def test_ngram_counts_and_mle_probabilities():
    model = NGramModel(n=2).train(["a b a c"])

    assert model.counts[("a",)] == {"b": 1, "c": 1}
    assert model.probability("a", "b") == pytest.approx(0.5)
    assert model.probability("a", "c") == pytest.approx(0.5)
    assert model.probability("a", "a") == 0.0


def test_add_k_smoothing_supports_unseen_contexts():
    model = NGramModel(n=2).train(["a b a c"])
    smoothed = model.distribution("a", add_k=1)
    unseen = model.distribution("missing", add_k=1)

    assert sum(smoothed.values()) == pytest.approx(1.0)
    assert smoothed["b"] == pytest.approx(0.4)
    assert smoothed["a"] == pytest.approx(0.2)
    assert sum(unseen.values()) == pytest.approx(1.0)
    assert len(set(unseen.values())) == 1


def test_suggestions_rank_contextual_words():
    model = NGramModel(n=2).train(["a b a c"])

    assert model.suggest("a", top_k=2, add_k=0) == [("b", 0.5), ("c", 0.5)]


def test_sampling_and_text_generation_are_repeatable():
    model = build_model("mystery")
    first = model.sample_next("detective", rng=random.Random(12))
    repeated = model.sample_next("detective", rng=random.Random(12))
    generated = generate_text(model, "Detective Rowan", word_count=6, seed=12)

    assert first == repeated
    assert generated.startswith("Detective Rowan ")
    assert len(tokenize(generated)) == 8
    assert generate_text(model, "start", word_count=4, seed=12) == generate_text(
        model, "start", word_count=4, seed=12
    )


def test_invalid_model_inputs_are_reported():
    with pytest.raises(ValueError, match="at least 1"):
        NGramModel(n=0)
    with pytest.raises(ValueError, match="Unknown genre"):
        load_corpus("nonexistent")
    with pytest.raises(ValueError, match="empty corpus"):
        NGramModel().train(["!!!"])
