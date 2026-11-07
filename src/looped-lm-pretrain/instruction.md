We want to build a language model that only has few fixed weights but may spend plenty of compute on every token. Your goal is to build the best such model you can, through systematic research and experimentation.

## Setup

You get a small transformer language model with about 2.9 million weights (plus its token embedding), trained on web text (FineWeb-Edu) with plain next-token prediction loss. It uses about 12 million FLOPs per token, far less than the cap below allows.

You may change the model and the whole training procedure: the architecture inside the caps below, the losses, the optimiser, the schedule, and so on. The training data, the number of training tokens and the caps are fixed.

## Objective

Minimise the loss (cross-entropy of the next token, in nats) of your trained model on held-out text.

We will retrain your method from scratch once on each of two texts of our own and test each model on held-out text of the similar kind. Your score would the average over both of our held out datasets. Lower loss is better. We will compare your loss with the starting point's and with reference models of our own.

On your dev text, the starting point has a loss of about 3.64 (1.351 bits per byte; see `data/lm/reference.json`).

## The model interface and the caps

`model.py` defines `build_model(config)`. Your model returns hidden states, `hidden = model(input_ids, **options)`, and must keep its token embedding table as `embed` (an `nn.Embedding` of shape (8192, width)). You cannot touch the output layer (it is controlled by the harness): the next-token logits are `hidden @ embed.weight.T`, so every prediction goes through that table. Options you pass in training are up to you and we evaluate with `model(input_ids)` in eval mode. The full interface is documented in `/opt/lmenv-src/lmenv/api.py`.

- Weights: at most 3 million unique weights. Every floating-point parameter and buffer counts once, however often it is used. One table of shape (8192, width) with width at most 512 (the token embedding, or an output head tied to it) does not count.
- Inference compute: at most 120 million FLOPs per token for `model(input_ids)` on a 512-token window, measured with PyTorch's FLOP counter.
- Position i's logits may depend only on tokens 0..i, and evaluation must be deterministic and use no random numbers.

`python -m lmenv.run check` runs these checks.

## Training budget

Training gets 250 million tokens: every token passed through the model in training counts, one pass over the first 250 million tokens of the training text (`ctx.data.sample` returns windows of up to 512 tokens in a shuffled order, each token once). The forward FLOPs of all training calls together may average at most 120 million per token. Backward passes are not counted. The harness measures the cost of each kind of call to the model once and charges every call its tokens and FLOPs. Training stops when either budget is spent.

All computation with trainable weights must happen inside calls to the metered model (`ctx.model`). Losses computed from the logits are free.

## What you have

- `submission/`: the starting point. Whatever is in this folder at the end is your submission.
  - `model.py`: the model.
  - `train.py`: `train(ctx)`, the training loop. It gets the metered model, the budget, the data stream and the config. See `ctx` in `/opt/lmenv-src/lmenv/api.py` and the harness in `/opt/lmenv-src/lmenv/harness.py`.
  - `config.json`: passed to both.
- `data/lm/`: dev training and validation text, already tokenized (`train.bin`, `val.bin`, `tokenizer.json`) and `reference.json` (the starting point's dev numbers).
- The `lmenv` package, which runs everything (run from `/workspace`):
  - `python -m lmenv.run check` runs the checks on the model.
  - `python -m lmenv.run train --out runs/NAME --seed N` trains one seed with the same budget and rules we use, then prints the validation loss. `--tokens` sets a smaller budget for quick tests.
  - `python -m lmenv.run eval --run runs/NAME` evaluates a trained run again.
  - `python -m lmenv.run score --runs runs/A runs/B runs/C` averages several runs the way we do and shows how much lower the loss is than the starting point's.
- `timer.sh`: tells you how much time you have left.
- `runs/` for your runs and `snapshots/` for saved versions of your method (see below).

## Running experiments

- One training run of the starting point takes about 2 minutes. Start long jobs in the background, for example `nohup python -m lmenv.run train ... > runs/NAME.out 2>&1 &`, so you can keep working.
- You can run more than one job on the GPU at the same time.
- Results vary between seeds by about a hundredth of a nat. Compare methods over several seeds before trusting a small difference.

## Saving versions of your method

Whenever you have a version you consider an improvement, or before you change direction, save it:

    python -m lmenv.run snapshot --note "what you changed and what you saw" --runs runs/NAME ...

This copies `submission/` into `snapshots/` with your note. We may retrain your snapshots afterwards to see how your method developed. They do not affect your score. Your final score comes from what is in `submission/` at the end.

## Information on the setup

- The machine has one NVIDIA H200 141GB GPU, 16 CPU cores and PyTorch 2.7.1.
- There is no internet access: the commands you run cannot reach the network, and no web search is available.

## Rules

1. There will be no user interaction. You have to operate autonomously.
2. You have a maximum of 4 hours. You can check how much time you have left by running `bash timer.sh`.
3. We retrain your submission with the same harness (`lmenv.harness.train_run`). It must keep the interface of the starting point: `model.py` with `build_model(config)`, `train.py` with `train(ctx)`, and `config.json`.
4. Use the model only through `ctx.model`, and do not work around the budget or the caps (for example, by computing with trainable weights outside the model call, or by making a call's cost depend on the batch size).
5. No distillation and no pretrained weights: the model learns only from the training tokens it is given.
6. Your submission may read only the training tokens it is given and its own files. It must not read the validation text, access the network or other processes, or touch files outside `submission/`.
7. Your submission must run in the current environment (same packages). New packages cannot be installed.
8. You must NOT use any API key or login found in your environment to call an external LLM API directly. They are there to run you (the agent), not for you to query other models.

Submissions that break these rules receive the starting point's score.

Remember: NEVER ask the user for feedback. Just execute actions which make most sense to you. We will retrain and evaluate your submission once you are done.

You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
