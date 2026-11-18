import math

import pytest

from app.evaluate import evaluate_genre, main
from app.core import CORPUS


def test_evaluation_compares_models_on_the_same_held_out_tokens():
    for genre, stories in CORPUS.items():
        result = evaluate_genre(genre)

        assert result["training_stories"] == len(stories) - 1
        assert result["test_tokens"] > 0
        assert 0 <= result["unknown_tokens"] <= result["test_tokens"]
        assert math.isfinite(result["ngram_perplexity"])
        assert math.isfinite(result["unigram_perplexity"])
        assert result["ngram_perplexity"] > 0
        assert result["unigram_perplexity"] > 0


def test_evaluation_cli_reports_models_and_overall_metrics(capsys):
    main(["--genre", "fantasy"])

    output = capsys.readouterr().out
    assert "3-gram vs. unigram baseline" in output
    assert "fantasy" in output
    assert "Overall" in output


def test_evaluation_requires_a_higher_order_model():
    with pytest.raises(SystemExit):
        main(["--n", "1"])


def test_evaluation_requires_positive_smoothing():
    with pytest.raises(SystemExit):
        main(["--add-k", "0"])
