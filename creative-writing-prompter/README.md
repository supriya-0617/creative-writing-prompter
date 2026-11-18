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

## Results and evaluation

The following are real outputs from the built-in corpora using the commands shown. The suggestion probabilities are add-1-smoothed next-word probabilities; the generation run is repeatable with seed `7`. Full terminal transcripts are saved in [`results/cli-examples.txt`](results/cli-examples.txt).

```text
> python -m app.cli suggest "the young dragon" --genre fantasy --top 5
guarded    0.0377
shared     0.0377
a          0.0189
and        0.0189
appeared   0.0189

> python -m app.cli generate "Detective Rowan" --genre mystery --words 20 --seed 7
Detective Rowan followed but of arrived muddy footprint arrived locked after glass arrived at had stopped burned closing of vale museum glass
```

Compare the trigram model with an add-1-smoothed unigram baseline on the same held-out text:

```powershell
python -m app.evaluate
```

For each genre, the first two built-in stories train both models and the final story is the test set. `PPL` is perplexity (lower is better). The evaluator reserves an `<unk>` token for words not in the training vocabulary, so all test tokens are scored; `Unknown` reports how many test tokens use it. Overall perplexity is the token-weighted geometric mean across genres.

| Genre | Training stories | Test tokens | Unknown tokens | Trigram PPL | Unigram PPL |
|---|---:|---:|---:|---:|---:|
| fantasy | 2 | 18 | 15 | 37.108 | 62.314 |
| mystery | 2 | 20 | 16 | 37.147 | 60.519 |
| science-fiction | 2 | 20 | 14 | 34.097 | 43.351 |
| **Overall** | — | **58** | **45** | **36.054** | **54.434** |

On this small split, the trigram model has lower perplexity than the unigram baseline in each genre and overall. That indicates it assigns higher probability to this test text when using its two-word context; it does not mean the generated prose is accurate or that the model generalizes well. Forty-five of 58 held-out tokens are unseen in training, so the results also show how limited the corpus is. The complete evaluation output is in [`results/evaluation.txt`](results/evaluation.txt).

### Limitations and possible improvements

- The built-in training data is only two short stories per genre in this evaluation, so the scores are noisy and genre coverage is narrow.
- The training vocabulary is small. The evaluator maps unseen test words to one reserved `<unk>` token; the generation model itself only generates words present in its training vocabulary.
- Unseen contexts receive add-k probabilities but there is no backoff to shorter contexts, so estimates for rare contexts are weak.
- A trigram uses at most the previous two words and cannot model long-range plot, meaning, or coherence.
- Predictions reflect the wording and genre of the selected training corpus; another genre or corpus may produce different results.
- Useful next steps are a larger and more representative corpus, repeated/stratified train-test splits, smoothing and n-gram tuning, context backoff, and explicit unknown-word handling.

## How it works

The built-in corpus contains short original stories in three genres. Text is lowercased and tokenized into words. The model counts each word after its preceding context of up to `n - 1` words. Maximum Likelihood Estimation (MLE) divides a word's context count by the total count for that context. Add-k smoothing adds `k` to every vocabulary count, allowing unseen words and contexts to receive nonzero probability. Suggestions rank these probabilities; generation samples words proportionally to them.

The evaluation command compares an n-gram model (trigram by default) with a unigram baseline using the same fixed holdout story for each genre. It reports perplexity with add-k smoothing; see the evaluation split and unknown-token handling above when interpreting the scores.

## Tests

```bash
python -m pytest -q
```

GitHub Actions runs the tests using Python 3.12. The project has no runtime dependencies; `pytest` is used for testing. A `.env` file is not required.
