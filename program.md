# autoresearch-pinn

This is an experiment to have the AI autonomously optimize a Physics-Informed Neural Network (PINN) for solving the Lugiato-Lefever Equation (LLE) strictly on a Kaggle T4 GPU.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `pinn-mar5`). 
2. **Create the branch**: `git checkout -b autoresearch-kaggle/<tag>` from current master.
3. **Read the files**:
   - `launch.py` — Wraps `train.py` and submits it to Kaggle via API. Do not modify.
   - `prepare.py` — Isolated ground truth execution logic for Kaggle. Do not modify.
   - `train.py` — The file you modify. Contains the MLP, physics equations, sampling, and optimizer.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.

## Experimentation

Each experiment is sent to Kaggle using `uv run launch.py`. The Kaggle kernel has an internal time budget of ~13 minutes (T4 GPU). Total turnaround time per run (including queue) is ~15-20 minutes.

**What you CAN do (in `train.py`):**
- Modify neural network architecture (depth, width, activations like Sine, Swish, Fourier Features).
- Modify collocation point sampling strategies (e.g., adaptive, residual-based sampling).
- Modify the optimizer logic, hyperparameters, or learning rate schedulers.
- Adjust loss weighting (`w_pde`, `w_ic`, `w_bc`) or try dynamic weightings (ReLoBRaLo, NTK, etc.).

**What you CANNOT do:**
- Modify `prepare.py` or `launch.py`.
- Exceed or remove the `TIME_BUDGET` variable usage in `train.py`.
- Look at or use the interior ground truth data during training.

## Logging results

Log to `results.tsv` (tab-separated):
```tsv
commit	val_mse	memory_gb	status	description
```
*Note: memory_gb = peak_vram_mb / 1024.*

## The experiment loop

LOOP FOREVER:
1. Brainstorm and implement a new PINN mathematical/architectural improvement in `train.py`.
2. `git commit -am "experiment description"`
3. Run the experiment on Kaggle: `uv run launch.py > run.log 2>&1`
4. Read results: `grep "^val_mse:\|^peak_vram_mb:" run.log`
5. If the log is empty or lacks `val_mse`, check the tail of `run.log` for Python crash traces. Record as CRASH.
6. Record the experiment outcome to `results.tsv`.
7. **Decision:** 
   - If `val_mse` improved (LOWER is better), KEEP it. 
   - If worse or crashed, you MUST revert: `git reset --hard HEAD~1`.

**Do not stop.** Keep iterating autonomously until interrupted by the human.