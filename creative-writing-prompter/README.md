# Creative Writing Prompter

This Creative Writing Prompter is a CLI tool designed for aspiring writers. It helps overcome writer's block by offering context-aware next-word suggestions. The core is a next-word prediction engine, built from scratch, that analyzes a corpus of short stories to provide relevant word choices.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that provides context-aware next-word suggestions for aspiring writers, helping them develop stories and overcome writer's block by sampling from a genre corpus, using a next-word prediction engine.

**Done when.** The tool successfully prepares a text corpus, implements n-gram counting with maximum-likelihood probabilities and add-k smoothing, generates text by sampling from the model, and offers a CLI for users to input text and receive suggestions.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a text corpus of short stories.**  
      Hint: Use text processing techniques to clean and tokenize the corpus.
- [ ] **2. Implement n-gram counting and maximum-likelihood probabilities.**  
      Hint: Implement n-gram counting and maximum-likelihood estimation for probabilities.
- [ ] **3. Add add-k smoothing to handle unseen n-grams.**  
      Hint: Apply add-k smoothing to handle unseen n-grams.
- [ ] **4. Develop a function to generate text by sampling from the model.**  
      Hint: Develop a sampling function to pick the next word based on probabilities.
- [ ] **5. Create a CLI interface for users to input text and get suggestions.**  
      Hint: Create a command-line interface to interact with the prediction engine.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify the n-gram size for the prediction model.
- Enable users to load their own text corpus for custom genre suggestions.
- Add a feature to generate a short sequence of words or a full sentence based on a starting prompt.
