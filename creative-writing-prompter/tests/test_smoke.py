def test_the_command_line_runs(capsys):
    from app.cli import main
    main(["hello"])
    assert "is ready" in capsys.readouterr().out


def test_suggest_command_prints_ranked_words(capsys):
    from app.cli import main

    main(["suggest", "the young dragon", "--genre", "fantasy", "--top", "2"])

    output = capsys.readouterr().out
    assert "guarded" in output
    assert len(output.strip().splitlines()) == 2


def test_generate_command_continues_prompt(capsys):
    from app.cli import main

    main(["generate", "Detective Rowan", "--genre", "mystery", "--words", "5", "--seed", "4"])

    output = capsys.readouterr().out.strip()
    assert output.startswith("Detective Rowan ")
    assert len(output.split()) == 7
