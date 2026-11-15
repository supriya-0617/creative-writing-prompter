"""Command-line entry point: ``python -m app.cli --help``."""

import argparse

from app.core import CORPUS, build_model, generate_text, tokenize


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--genre",
        choices=sorted(CORPUS),
        default="fantasy",
        help="built-in corpus genre (default: fantasy)",
    )
    parser.add_argument("--n", type=int, default=2, help="n-gram size (default: 2)")
    parser.add_argument(
        "--corpus",
        help="path to a custom UTF-8 text file (replaces the built-in corpus)",
    )
    parser.add_argument(
        "--add-k",
        type=float,
        default=1.0,
        help="add-k smoothing value (default: 1.0)",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="creative-writing-prompter",
        description="Get genre-aware next-word suggestions or generate a continuation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("hello", help="check that everything is wired up")

    suggest_parser = subparsers.add_parser(
        "suggest", help="show likely next words for a prompt"
    )
    suggest_parser.add_argument("prompt", help="the text to continue")
    suggest_parser.add_argument("--top", type=int, default=5, help="number of suggestions")
    _add_model_options(suggest_parser)

    generate_parser = subparsers.add_parser(
        "generate", help="sample a continuation for a prompt"
    )
    generate_parser.add_argument("prompt", nargs="?", default="", help="starting text")
    generate_parser.add_argument(
        "--words", type=int, default=20, help="number of words to generate"
    )
    generate_parser.add_argument("--seed", type=int, help="seed for repeatable output")
    _add_model_options(generate_parser)

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Creative Writing Prompter is ready. Run with --help to see commands.")
        return

    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.add_k < 0:
        parser.error("--add-k must be non-negative")
    if args.command == "suggest" and args.top < 1:
        parser.error("--top must be at least 1")
    if args.command == "generate" and args.words < 0:
        parser.error("--words must be non-negative")

    try:
        model = build_model(args.genre, args.n, args.corpus)
        if args.command == "suggest":
            context = tokenize(args.prompt)
            for word, probability in model.suggest(
                context, top_k=args.top, add_k=args.add_k
            ):
                print(f"{word}\t{probability:.4f}")
        else:
            print(
                generate_text(
                    model,
                    args.prompt,
                    args.words,
                    add_k=args.add_k,
                    seed=args.seed,
                )
            )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
