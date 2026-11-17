# Figures and tables

## Figures

- `figures/passk.png`, `.svg` (on the page): <strong>Additional Opus attempts narrow, but do not close, the score gap to Sol and Astra.</strong> Pass@k averages the highest held-out score across every size-k subset of the four Opus runs.
- `figures/tradeoff.png`, `.svg` (on the page): <strong>Residual hacking can outweigh stronger task performance.</strong> Opus’s strongest submission earns more true reward than Sol or Astra, but loses that advantage to hack penalties.
- `figures/transfer.png`, `.svg` (on the page): <strong>The ranking changes on unseen environments.</strong> The grader uses each final submission to train fresh models in both sets of environments.
- `figures/ideas_landscape.png`, `.svg` (on the page): <strong>The strongest runs achieve low hack rates with different training methods.</strong>
- `figures/ideas_timeline.png`, `.svg` (on the page): <strong>Sol and Astra explore a wider range of training methods than the Opus runs.</strong>
- `figures/dev_curves.png`, `.svg`: Each run's dev score at each full launcher run, in the agent's own envs; ◆ the submission's held-out score.
- `figures/dev_curves_picked.png`, `.svg`: As dev_curves, for each agent's best@N, worst@N and dev-selected@N runs: ◆ best, ▼ worst, ■ dev-selected.
- `figures/tradeoff_clean.png`, `.svg`: Clean reward (true reward, 0 on hacking completions) against hack rate: each run's dev checks in order, ◆ its held-out result.
- `figures/submissions.png`, `.svg`: Held-out score of every iteration each run submitted, when submitted: ◆ graded at the end of the run, ○ earlier ones (regraded); scores below −1 drawn at −1.
- `figures/idea_map.png`, `.svg`: Idea map: every iteration each run tried, by method family (subagents judged the families from each iteration's code and notes; neighbors are the most similar); ◆ final submissions with their run number.
- `figures/idea_timeline.png`, `.svg`: Method family of each iteration against hours into the run, one panel per agent and one line per run; ◆ final submissions.
- `figures/idea_space.png`, `.svg`: Idea space: every iteration of both agents in one 2-D layout (near = similar method family and shared building blocks), a blob per family (faded: the agent never tried it), a line from each run's first iteration (○) to its final submission (●).

## Tables

Hack rates are fractions (0.01 = 1%).

- `tables/results.md`, `.csv`: Table 1: Opus best@4 and worst@4, Sol, Astra, and GRAFT (held-out means).
- `tables/picked.md`, `.csv`: Each agent's best@N, worst@N and dev-selected@N runs, then the baselines.
- `tables/runs.md`, `.csv`: Every run, best held-out score first, with its tags; then the baselines.
- `tables/held_out.md`, `.csv`: Per held-out env, each agent's mean over its runs, and the baselines.
- `tables/dev.md`, `.csv`: Per dev env (the agent's own), each agent's mean over its runs, and the baselines.
