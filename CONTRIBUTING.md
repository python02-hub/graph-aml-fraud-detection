# Contributing

Thanks for considering a contribution — this is a small project, so the process is kept
simple.

## Setup

```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Before opening a PR

Run the smoke test locally — it's the same check CI runs on every push, so if this passes,
your PR almost certainly will too:

```bash
python aml_gnn.py --demo --epochs 10 --out runs/ci --no-baseline
python visualize_graph.py --run runs/ci
```

Confirm it finishes without errors and that `runs/ci/graph.html` opens in a browser.

## Ideas for contributions

- Additional graph features (e.g. temporal motifs, community detection scores)
- Support for other public AML/fraud datasets (adjust `COLMAP` in `aml_gnn.py`)
- Model calibration (e.g. Platt scaling / isotonic regression on validation scores)
- A proper train/serve split so the model can score a stream of new transactions
  incrementally instead of batch-only
- Tests beyond the current end-to-end smoke test

## Reporting issues

Please include: the command you ran, the full error/traceback, your OS and Python version,
and (if relevant) a few rows of anonymized sample data that reproduce the issue.
