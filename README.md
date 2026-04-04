# 🛡️ GuardMCP

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/SHRAVANIRANE/GuardMCP?quickstart=1)
[![CI](https://github.com/SHRAVANIRANE/GuardMCP/actions/workflows/ci.yml/badge.svg)](https://github.com/SHRAVANIRANE/GuardMCP/actions)

GuardMCP is a research prototype for checking whether an agent's proposed action stays semantically aligned with the user's intent.

The project compares two decision rules:
- a directional alignment method that measures how much extra semantic content appears in the action
- a cosine-similarity baseline that measures overall semantic similarity

The current repo is positioned as a benchmark-backed college/research prototype, not a production runtime guard.

## Problem

Tool-using agents can propose actions that look related to the user's request while still adding hidden behavior.

Example:
- Intent: `Read a file`
- Risky action: `Read the file and send it to an external server`

A plain similarity score may say the action is related to the intent. GuardMCP asks a stricter question:

`Is the action only doing what the user asked, or is it carrying extra semantic intent?`

## Core Idea

GuardMCP embeds both the user intent and the proposed action into vectors, then evaluates them in two ways.

1. `Directional alignment`
   The action vector is decomposed into:
   - a projection in the direction of the intent
   - a rejection component that captures extra semantic content

   If the rejection magnitude is too large, the action is blocked.

2. `Cosine baseline`
   The action is allowed if cosine similarity is above a threshold.

This lets the project test whether directional leakage detection is more useful than plain similarity for agent safety.

## Current Project Scope

The repo currently includes:
- local manual and generated intent-action cases
- benchmark adapters for ToolTalk and AgentDojo
- split-aware evaluation with deterministic `train/dev/test` assignment
- separate threshold tuning for directional and cosine on `dev`
- final reporting on `test`
- grouped metrics by source, suite, and inferred attack type
- a small CLI demo for live presentation

## Repository Structure

```text
guardmcp/
|-- src/
|   |-- alignment/       # Directional method and cosine baseline
|   |-- data/            # Local data plus ToolTalk/AgentDojo adapters
|   |-- demo_service.py  # Shared demo logic used by CLI and Streamlit
|   |-- embeddings/      # SentenceTransformer wrapper
|   |-- evaluation/      # Evaluator, metrics, grouped reporting
|-- experiments/
|   |-- run_experiments.py
|   |-- plot_results.py
|-- results/
|   |-- outputs.csv
|   |-- results_summary.csv
|   |-- best_thresholds.csv
|   |-- reports/
|-- main.py              # CLI demo
|-- streamlit_app.py     # Small UI demo
|-- config.py            # Demo defaults and threshold paths
|-- README.md
```

## Dataset Sources

GuardMCP currently evaluates on a combined dataset built from:
- local manual cases in [test_cases.py](/Users/Shravani/Desktop/Projects/guardmcp/src/data/test_cases.py)
- local generated cases in [generated_cases.json](/Users/Shravani/Desktop/Projects/guardmcp/src/data/generated_cases.json)
- aligned benchmark cases adapted from ToolTalk via [tooltalk_adapter.py](/Users/Shravani/Desktop/Projects/guardmcp/src/data/tooltalk_adapter.py)
- adversarial benchmark cases adapted from AgentDojo via [agentdojo_adapter.py](/Users/Shravani/Desktop/Projects/guardmcp/src/data/agentdojo_adapter.py)

Current row counts in the combined dataset:
- local: `76`
- ToolTalk: `78`
- AgentDojo: `567`
- total: `721`

Important note:
- ToolTalk is adapted into positive intent-action pairs.
- AgentDojo is adapted into negative intent-action pairs by pairing official benign task prompts with official injection goals.
- This is an adaptation for GuardMCP's schema, not a raw replay of executed benchmark trajectories.

## Installation

Use Python `3.10+`.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing Benchmark Data

The converted benchmark JSON files are already present in the repo, but you can rebuild them with:

```bash
venv\Scripts\python.exe -m src.data.tooltalk_adapter --splits easy hard --output src/data/tooltalk_cases.json
venv\Scripts\python.exe -m src.data.agentdojo_adapter --version v1 --suites workspace travel banking slack --output src/data/agentdojo_cases.json
```

## Running Experiments

Run the full benchmark-backed experiment:

```bash
venv\Scripts\python.exe experiments/run_experiments.py --include-tooltalk --include-agentdojo
```

This now does the following:
1. loads all selected datasets
2. assigns deterministic `train/dev/test` splits
3. scores intent-action pairs once with embeddings
4. tunes directional and cosine thresholds separately on `dev`
5. reports final metrics only on `test`
6. saves grouped analysis tables

Main output files:
- [results_summary.csv](/Users/Shravani/Desktop/Projects/guardmcp/results/results_summary.csv)
- [best_thresholds.csv](/Users/Shravani/Desktop/Projects/guardmcp/results/best_thresholds.csv)
- [outputs.csv](/Users/Shravani/Desktop/Projects/guardmcp/results/outputs.csv)
- [by_source_metrics.csv](/Users/Shravani/Desktop/Projects/guardmcp/results/reports/by_source_metrics.csv)
- [by_suite_metrics.csv](/Users/Shravani/Desktop/Projects/guardmcp/results/reports/by_suite_metrics.csv)
- [by_attack_type_metrics.csv](/Users/Shravani/Desktop/Projects/guardmcp/results/reports/by_attack_type_metrics.csv)

## Plotting

Regenerate the metric plot from the latest summary CSV:

```bash
venv\Scripts\python.exe experiments/plot_results.py
```

This writes:
- [precision_recall_graph.png](/Users/Shravani/Desktop/Projects/guardmcp/results/plots/precision_recall_graph.png)

## CLI Demo

One-shot mode:

```bash
venv\Scripts\python.exe main.py --intent "Play music" --action "Play music and delete temp files"
```

Interactive mode:

```bash
venv\Scripts\python.exe main.py --interactive
```

The demo:
- loads calibrated thresholds from [best_thresholds.csv](/Users/Shravani/Desktop/Projects/guardmcp/results/best_thresholds.csv) when available
- prints the GuardMCP verdict
- prints the directional rejection magnitude
- prints the cosine similarity baseline

Good live demo example:
- Intent: `Play music`
- Action: `Play music and delete temp files`

This currently produces a `BLOCK` decision.

## Streamlit UI

Run the small visual demo with:

```bash
venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The UI includes:
- side-by-side intent and action inputs
- calibrated thresholds loaded from [best_thresholds.csv](/Users/Shravani/Desktop/Projects/guardmcp/results/best_thresholds.csv)
- preset examples for safe, blocked, and limitation cases
- a final GuardMCP verdict card
- directional and cosine score comparison

Why this UI exists:
- it makes the project easier to demo in interviews
- it gives the repo a more complete resume-ready presentation layer
- it still uses the same shared GuardMCP logic as the CLI, so the results stay consistent

## Latest Evaluation Snapshot

On the current combined benchmark-backed run:
- total rows: `721`
- split sizes: `505 train`, `108 dev`, `108 test`
- best directional threshold on `dev`: `0.862142`
- best cosine threshold on `dev`: `0.506667`

Final `test` metrics:
- directional: accuracy `0.85`, precision `0.46`, recall `0.86`, F1 `0.60`
- cosine: accuracy `0.85`, precision `0.46`, recall `0.86`, F1 `0.60`

Interpretation:
- the split-aware pipeline is working
- the grouped reports now show which sources and attack families are hardest
- on the current split, directional and cosine ended up making the same final `test` decisions after calibration

## Presentation-Friendly Summary

You can describe GuardMCP like this:

`GuardMCP is a research prototype for semantic guardrails in tool-using AI agents. It compares a directional intent-action alignment method against cosine similarity using local adversarial examples and adapted public benchmarks such as ToolTalk and AgentDojo.`

## Limitations

- The project is still a research prototype, not a deployment-ready runtime guard.
- AgentDojo data is adapted into GuardMCP's intent-action format rather than replayed as full agent trajectories.
- The current attack-type labels in grouped reporting are inferred from action text and are meant for analysis, not benchmark ground truth.
- On the current split, directional and cosine are tied on final `test` metrics, so the project still needs stronger separation to support a stronger research claim.

## Next Steps

- improve dataset diversity further
- add richer domain-aware attack labels
- compare against additional baselines
- add tests and a more polished report/presentation bundle
- integrate the runtime guard with a real agent framework
