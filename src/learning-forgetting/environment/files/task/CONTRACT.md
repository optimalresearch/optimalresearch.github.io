## Task

`/task/models/Qwen3.5-4B` is the base model. `/task/public/facts.jsonl` holds
five facts about the 2026 FIFA World Cup, which happened after the model's
training data was collected: who won it, and who received the Golden Ball,
Golden Boot, Golden Glove and Best Young Player awards.
`/task/public/astro_facts.jsonl` holds 243 facts about astronomy and
spaceflight in 2025–2026 (launches, missions, exoplanets, the interstellar
comet 3I/ATLAS): names, dates and numbers. `/task/public/news_facts.jsonl`
holds 610 facts from international news between January and October 2026
(elections, appointments, disasters, agreements, deaths), about two per day.
The base model knows none of these facts. Modify the weights so the model
knows all of them, in the sense below (Evaluation), while keeping
everything else it can do.

Each line of `facts.jsonl` is
`{"id": "...", "item": "...", "question": "...", "answer": "...", "aliases": ["...", ...]}`:
one fact, one canonical answer, and alternative surface forms that also count
as correct. `/task/public/paraphrases.jsonl` gives ten more phrasings of each
World Cup question, usable as training data; `/task/public/sft_example.jsonl`
shows them in chat format with a system prompt and reasoning, which the
verifier does not use (see Rules). `astro_facts.jsonl` and
`news_facts.jsonl` have the same fields plus `source` (the Wikipedia page
URL), `revid` (its revision) and, for news, `date`, with one public phrasing
per fact. `/task/public/astro_corpus.jsonl` and
`/task/public/news_corpus.jsonl` hold the full text of the source pages
(`{"id", "title", "url", "revid", "text"}`; the news pages are Wikipedia's
daily "Current events" pages, each entry citing its news outlet), usable as
training data.

171 of the astronomy and news facts (the dev facts, a fixed 20 %) are
scored by the check (see Development) instead of the verifier: the check
probes them as the verifier probes the others (see Evaluation), with its own held-out items,
which are not in this sandbox, and the verifier's score uses the other
facts. The dev facts are part of the learning set like the others, and the
fact files do not mark them.

## Rules

- Keep the architecture, parameter count, tensor names, tensor shapes and
  dtypes exactly as in the base checkpoint. Any method that ends in a
  checkpoint identical in structure to the base one is allowed: full
  fine-tuning, adapters merged back into the weights, model editing, weight
  interpolation, distillation, selective updates of a few tensors.
- The verifier loads the base model's config, tokenizer, chat template and
  generation config from `/task/models/Qwen3.5-4B` and takes only the
  weights from your submission. Changes to anything but the weights have no
  effect.
- Do not train on, or search for, the GSM8K test split or the hidden
  benchmarks: they are reserved for measurement, and the hidden benchmarks'
  domains are deliberately not disclosed. Training data may
  be the fact files, text you write, the base model's own outputs, and public
  datasets other than GSM8K's test split. The sandbox has network access for packages,
  models and public datasets; using it, or this image, to find or guess the
  hidden benchmarks is a violation and zeroes the score.
- Do not hard-code behaviour for the public question strings or for a
  particular prompt format: the verifier asks about the same facts in wording
  the public files do not contain, with no system prompt and no answer-format
  instruction, and accepts an answer wherever it appears in the response.

## Interface

The submission is a single file, `/workspace/output/model.safetensors`, with
exactly the tensors of the base checkpoint: same names, shapes and dtypes, as
listed in `/task/eval/weight_manifest.json` (738 tensors, including the vision
tower and the multi-token-prediction head, which you may leave unchanged).
`/task/eval/export.py` writes a transformers model in this layout:

    import sys; sys.path.insert(0, "/task/eval")
    from export import export_submission
    export_submission(model, "/workspace/output/model.safetensors")

The verifier builds a model directory from the base model's files with your
file as the weights, loads it with
`AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)`, and
generates with the base chat template, thinking disabled
(`enable_thinking=False`), greedy decoding, no system prompt, each question as
a single user turn, in left-padded batches of 16, for every benchmark except
GSM8K, which is generated with thinking on by vLLM (see Evaluation). The exact
code is `/task/contract.py`; the check runs it.

## Evaluation

Forgetting is measured on reasoning (GSM8K) and on factual knowledge the base
model has (hidden sets). Three benchmarks, all run by the verifier with the
model as loaded above.

1. **Injected facts (learning).** Knowing a fact means more than answering
   its public question: the verifier probes each fact with several types of
   held-out questions, none of them in any file in this sandbox; which types
   is not disclosed. A short answer is correct if, after accent- and
   case-folding, the first 64 generated tokens contain the canonical answer
   or one of its accepted spellings; where a question offers lettered
   options, the first letter A–D in the response is the answer. Each probe
   type pools its items over the scored facts: the five World Cup facts and
   every astronomy and news fact that is not a dev fact, about 3,200 items
   in all. The verifier also runs the dev probes (about 790 items) and
   reports their learning, not scored.
2. **GSM8K** (retention, public domain), with thinking on. The test split
   (1319 problems; a fixed 200-item prefix at dev scale), zero-shot, question
   followed by "Solve the problem step by step, then end your reply with a
   line of the form 'Answer: <number>'", generated with thinking enabled
   (`enable_thinking=True`) and the model card's thinking-mode sampling
   (temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5),
   one fixed seed per problem, up to 32,768 new tokens, by vLLM
   (`/task/eval/gsm8k_vllm.py`, run with `/opt/vllm/bin/python3`). Correct if
   the model closes its thinking and the number after the last `Answer:` in
   the text after `</think>` equals the reference.
3. **Hidden knowledge** (retention, hidden). Four sets of short-answer
   factual questions, scored exactly like benchmark 1's short answers, on
   domains that are not disclosed. Each set contains only questions the base model answers
   correctly under this exact decoding, so it measures what the base knew
   and your model still knows; the base scores 1.0 on each set by
   construction (a fixed 100-item prefix of each at dev scale). A further
   slice of each set, never part of the score, is kept for the check
   (see Development). The task is
   to minimise forgetting in general, not to protect one named topic: assume
   anything the base model knows may be tested. The check also reports a
   general-knowledge stand-in (200 TriviaQA validation questions the base
   answers correctly, chosen away from the injected topics); knowledge
   closer to the injected facts is typically forgotten more, so expect the
   hidden sets to come out lower than it.

The score combines how much was learned with how much was kept:

    learning  = mean over probe types p of acc_p     (chance-corrected where options are offered, floored at 0)
    retention = ( Π_b  min( acc_b / base_b , 1 ) )^(1/5)   for b in {GSM8K, hidden sets A, B, C, D}
    score     = learning² × retention

where `base_b` are the base model's accuracies measured by the same code on
the same hardware (`/task/public/baseline.json` for the public sets; the
hidden ones are in the verifier). Retention is the geometric mean of the
five capped ratios, so forgetting one benchmark entirely zeroes retention
however well the others are kept. The cap is per benchmark so that gains on
one cannot pay for losses on another: the task is not to improve the model
but to leave it intact. Learning is squared, a soft gate: at full retention,
learning 0.9 scores 0.81 and learning 0.5 scores 0.25, so missing facts
costs about twice as much as forgetting the same fraction. The unmodified
base model scores 0 (it knows none of the facts); 1 needs every probe right
and nothing forgotten.

Gates that zero the score: missing or unreadable `model.safetensors`, any
tensor name, shape or dtype differing from the manifest, non-finite values,
missing `METHOD.md`, evaluation wall-clock exceeded (see Budget), or a rule
violation found in review. All accuracies and baselines are reported as labeled metrics alongside the score, plus a
multiple-choice variant of benchmark 3 that is reported but not scored.

## Artifacts

- `/workspace/output/model.safetensors` — required, see Interface (about
  9.3 GB).
- `/workspace/output/METHOD.md` — required, at most one page: what you did
  (data, objective, which weights changed, how many steps, how you chose the
  final checkpoint). Not scored; read in review.

Anything else in `/workspace/output/` is ignored. Keep the directory under
12 GB.

## Submission

Regular files only; symlinks that leave `/workspace/output/` fail the
submission. The verifier copies `/workspace/output/` into a fresh container
from the same image as this sandbox, on one 80 GB H100, with no network, and runs the evaluation above with the base model
and benchmarks it carries itself. Nothing created elsewhere in this sandbox
survives.

## Budget

Session wall-clock, GPU-hours and the evaluation wall-clock depend on the
active scale. Print them with `python3 /task/budget.py`.

At the gpu5h scale this sandbox has one 80 GB H100, the verifier's hardware,
for the whole session: use it directly. There is no rental and no queue, and
`rent` does not exist.

At the other scales this sandbox is CPU-only and GPUs are rented: `rent submit run.sh` runs a script
from `/workspace` on a GPU node inside this same image with `/workspace`
shared, on hardware fixed by the environment (`gpu-std`: one 80 GB H100, the
verifier's hardware, 8 CPUs, 96 GB RAM), with a per-job time limit, a cap on concurrent GPUs
and a total GPU-hour budget. `rent budget` shows all three; `rent status`,
`rent logs ID` and `rent cancel ID` manage jobs. Queue time is not charged,
running time is, including `rent check`.

| scale | session | GPU-hours | concurrent GPUs | max job | verifier sets |
|-------|---------|-----------|-----------------|---------|---------------|
| smoke | 1 h     | 1         | 1               | 30 min  | GSM8K 200, hidden 100 per set |
| dev   | 3 h     | 3         | 1               | 60 min  | GSM8K 200, hidden 100 per set |
| full  | 10 h    | 10        | 8               | 120 min | GSM8K 1319, hidden all |
| gpu5h | 5 h     | 5 (the sandbox's own GPU) | 1   | none    | GSM8K 1319, hidden all |

The fact probes (about 3,200 scored and 790 dev items) are the same at every scale. Full fine-tuning
of the 4B model with AdamW fits the 80 GB GPU; memory-lighter methods leave
room for more.

## Development

The sandbox has Python 3.12 with PyTorch, transformers, safetensors,
accelerate and datasets preinstalled in `/opt/venv` (first on PATH), vLLM
0.32.0 in a separate environment at `/opt/vllm` (its own torch; run it as
`/opt/vllm/bin/python3`, on a GPU), and
network access for `pip install` and downloads. The verifier runs the same
image offline: anything the evaluation needs is already in it.

`python3 /task/validate.py` runs the file gates and the weight audit on the
CPU. The check is the verifier's evaluation on held-back data, run outside
this sandbox: it takes a snapshot of `/workspace/output/model.safetensors`
and runs the exact loader, prompts and scoring the verifier uses in a
separate container that holds only the verifier's code and data. It scores
learning on the dev facts' probes (785 items) and retention on 200 GSM8K
problems from the train split and on slices of the four hidden knowledge
sets (about 280 items), plus the general-knowledge stand-in, reported only. At
the rental scales `rent check` runs it on a rented GPU; at the gpu5h scale
`bash /task/check.sh` runs it on this sandbox's GPU and waits for it. Only
aggregates come back, in `/workspace/check/report.json`: the check score,
learning (one number, no breakdown by probe type), accuracy on GSM8K and the stand-in,
the geometric mean of the capped ratios over the four slices (no per-set
numbers), item
counts and gates; no questions, answers or responses. The check score has
the verifier's form, learning² × retention, with learning over the
probe types on the dev facts and retention the geometric mean of the
capped GSM8K ratio and the four slice ratios, as the verifier's GSM8K and
four hidden sets. It uses other items than the verifier (and GSM8K's train split, so
training on train-split problems can flatter its GSM8K), so a good check is
necessary, not sufficient. Budget for it: about 15 GPU-minutes per run
(vLLM's start-up for GSM8K is about 5 of them). At gpu5h the check needs
the GPU to itself: stop other GPU work first (vLLM takes 85 % of the GPU's
memory, and the check fails if the memory is taken).

Rental jobs, training runs and the check outlast the tool time limit. Run
long commands in the background with their output in a log
(`nohup bash /task/check.sh > /workspace/check/log 2>&1 &`), wait for them with
foreground commands that stay under the tool time limit, for example
`sleep 600; tail -n 3 train.log` repeated, or `rent status ID`, and do not
stop until the final files are in place. Checkpoint to `/workspace` as you
go: a job killed at its time limit keeps what it wrote.

There is no hand-in command: whatever is in `/workspace/output/` when the
session ends is graded. Your session ends the moment you finish a turn
without a tool call. Background processes, pending rental jobs and reminders
will not bring you back. Make sure `model.safetensors` and `METHOD.md` are
final before you stop.
