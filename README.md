# Track 1 — Learned Page Replacement

A small, reproducible CSE-307 Operating Systems experiment comparing **FIFO**, **LRU**, **Optimal (Belady)**, and a transparent **learned eviction policy**.

## What the project studies

The same page-reference trace is used for every policy. Its first half has strong locality; halfway through, the workload changes to a wider random pattern. The experiment reports page faults and hit ratios before and after the shift.

The learned policy is deliberately lightweight: it trains a two-class nearest-centroid classifier on the first phase. Training examples describe resident-page candidates using normalized recency, access frequency, and age. The label is whether Belady's oracle would evict that candidate. During evaluation, the policy uses only information available up to the current reference.

## Requirements

- Ubuntu or another Unix-like system
- Python 3.10 or newer
- No third-party Python packages

## Run

```bash
cd track1_page_replacement
python3 src/run_experiment.py
```

The command creates or refreshes:

- `results/access_trace.csv` — generated trace and phase labels
- `results/policy_metrics.csv` — faults and hit ratios by policy and phase
- `results/summary.json` — compact overall summary
- `results/hit_ratio.svg` — dependency-free chart

To inspect the CSV results:

```bash
column -s, -t < results/policy_metrics.csv
```

## Project structure

```text
src/run_experiment.py   implementation and experiment driver
results/                generated raw results and figure
report/term_paper.md    short report draft to edit after rerunning
```

## Important interpretation note

Optimal uses future references by definition, so it is a benchmark rather than an implementable online policy. The learned policy does not use future references during evaluation; future references are used only to create training labels from the training phase.

The exact numbers are reproducible with seed `307`, but students should understand the code, rerun the experiment, and write their own interpretation before submission.

## Assistance disclosure

This repository was assembled with implementation assistance from an AI coding assistant. The experiment design, generated results, interpretation, and final submission should be reviewed and edited by the student, who remains responsible for understanding and explaining every part of the work.
