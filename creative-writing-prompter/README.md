# Creative Writing Prompter

A Python 3.12 command-line tool that suggests likely next words and generates short continuations from original fantasy, mystery, and science-fiction story samples. It uses a local n-gram model; no AI API or credentials are required.

## Setup

From the repository root, enter the project directory:

```powershell
cd creative-writing-prompter
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, create and activate the environment with:

```bash
cd creative-writing-prompter
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Use

Get ranked next-word suggestions:

```bash
python -m app.cli suggest "the young dragon" --genre fantasy --top 5
```

Generate a continuation (the seed makes sampling repeatable):

```bash
python -m app.cli generate "Detective Rowan" --genre mystery --words 20 --seed 7
```

Available genres are `fantasy`, `mystery`, and `science-fiction`. Use `--n 3` to select a trigram model. A custom UTF-8 text file can replace the built-in genre corpus with `--corpus path/to/stories.txt`. Both commands accept `--add-k` to set the smoothing value. Run `python -m app.cli --help` for all options.

## How it works

The built-in corpus contains short original stories in three genres. Text is lowercased and tokenized into words. The model counts each word after its preceding context of up to `n - 1` words. Maximum Likelihood Estimation (MLE) divides a word's context count by the total count for that context. Add-k smoothing adds `k` to every vocabulary count, allowing unseen words and contexts to receive nonzero probability. Suggestions rank these probabilities; generation samples words proportionally to them.

## Tests

```bash
python -m pytest -q
```

GitHub Actions runs the tests using Python 3.12. The project does not need a `.env` file. `.env` is ignored by Git; do not commit secrets.
