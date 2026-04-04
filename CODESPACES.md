# Running GuardMCP in Codespaces

Dependencies are already installed. Choose what to run:

## Streamlit demo (recommended for first look)

```bash
python -m streamlit run streamlit_app.py
```

The browser tab opens automatically. Use the preset examples in the sidebar.

## CLI — one-shot mode

```bash
python main.py --intent "Play music" --action "Play music and delete temp files"
```

## CLI — interactive mode

```bash
python main.py --interactive
```

## Full benchmark experiment

```bash
python experiments/run_experiments.py --include-tooltalk --include-agentdojo
python experiments/plot_results.py
```

Results write to `results/`. The plot saves as `precision_recall_graph.png`.
