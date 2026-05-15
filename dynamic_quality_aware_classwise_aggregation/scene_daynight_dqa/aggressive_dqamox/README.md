# Aggressive DQA-MoX Experiments

This directory isolates the next DQA-MoX attempts from the older exploratory
notebooks and outputs.

- `notebooks/`: user-facing notebooks for aggressive DQA-MoX loops.
- `scripts/`: controllers that launch full experiments.
- `output/`: generated checkpoints, logs, pseudo labels, and evaluation files.
- `reports/`: compact CSV/Markdown summaries.
- `logs/`: reserved for outer execution logs.

The current loop is `24_aggressive_until_target`.  It starts from the same
full-from-warmup DQA-MoX philosophy, but deliberately uses client-dominant
settings instead of defensive repair-preserving settings.
