# BP Algorithm Experiments

Tools and experiments for belief propagation and tensor-network workflows. Core scripts live in this repo, with notebooks demonstrating experiments and diagnostics.

## Layout
- `algo_bp.py`, `algo_cooling.py`: main belief-propagation and cooling routines.
- `gen_loop.py`, `gen_loop_tn.py`: loop generation helpers for tensor networks.
- `quf.py`: utilities and experiment helpers (partitioning, contractions, plotting).
- `register_.py`: custom JAX/PyTorch registrations and linear algebra helpers.
- `test_*.ipynb`, `energy_var*.ipynb`: example notebooks and exploratory runs.
- `cash/`: local cache/artifacts (ignored).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# For GPU JAX/PyTorch, install the appropriate wheels from the vendor instructions.
```

## Notes
- `.gitattributes` marks notebooks as binary to avoid noisy diffs. Use comments/screenshots or tools like `nbdiff` for review.
- `.gitignore` excludes caches, checkpoints, and `cash/` artifacts; keep transient data there or elsewhere.
- If you add new notebook outputs or large data, prefer storing them outside the repo or use Git LFS if needed.
