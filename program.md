# autoresearch-pinn

This is an experiment to have the LLM autonomously optimize a Physics-Informed Neural Network (PINN) for solving the Lugiato-Lefever Equation (LLE) strictly on a Kaggle T4 GPU.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `pinn-mar5`). 
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the files**:
   - `launch.py` — Wraps `train.py` and submits it to Kaggle via API. Do not modify.
   - `prepare.py` — Isolated ground truth execution logic for Kaggle. Do not modify.
   - `train.py` — The file you modify. Contains the MLP, physics equations, sampling, and optimizer.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.

## Experimentation

Each experiment is sent to Kaggle using `uv run launch.py`. The Kaggle kernel has an internal time limit of ~13 minutes inside `train.py`, so total turnaround time (including Kaggle queue) is ~15-20 minutes.

**What you CAN do (in `train.py`):**
- Modify neural network architecture (depth, width, activations like Sine, Swish).
- Modify collocation point sampling strategies (e.g., adaptive, residual-based sampling).
- Modify the optimizer logic, hyperparameters, or learning rate schedulers.
- Adjust loss weighting (`w_pde`, `w_ic`, `w_bc`) or try dynamic weightings (ReLoBRaLo, NTK, etc.).
- Enhance model logic (e.g. Fourier feature embeddings).

**What you CANNOT do:**
- Modify `prepare.py` or `launch.py`.
- Exceed the `TIME_BUDGET` variable in `train.py`.
- Look at the interior ground truth data during training.

## Output format

When `launch.py` finishes, it downloads the Kaggle stdout log which contains:
```
---
val_mse:          1.234567e-03
training_seconds: 795.2
peak_vram_mb:     1250.5
num_steps:        14500
num_params:       33540
```
You can extract it from the local log exactly like: `grep "^val_mse:" run.log`

## Logging results

Log to `results.tsv` (tab-separated):
```
commit	val_mse	memory_gb	status	description
```
*Note: memory_gb = peak_vram_mb / 1024.*

## The experiment loop

LOOP FOREVER:
1. Tune `train.py` with a new PINN optimization idea.
2. `git commit`
3. Run the experiment on Kaggle: `uv run launch.py > run.log 2>&1`
4. Read results: `grep "^val_mse:\|^peak_vram_mb:" run.log`
5. If empty, check `tail -n 50 run.log` for Kaggle crashes/stack traces. 
6. Record to `results.tsv`.
7. If `val_mse` improved (LOWER is better), keep it. If worse, `git reset --hard HEAD~1`.

**Do not stop.** Keep iterating and thinking of new mathematical/architectural concepts for PINNs until interrupted by the human.