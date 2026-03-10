# autoresearch-pinn (Windows 10 Edition)

This is an experiment to have the AI autonomously optimize a Physics-Informed Neural Network (PINN) for solving the Lugiato-Lefever Equation (LLE) strictly on a Kaggle T4 GPU.

**ENVIRONMENT:** You are operating on a Windows 10 system using CMD/PowerShell. Use Windows-compatible terminal commands or your built-in file editing/reading tools.

## Setup Phase (Run Once)

1. **Verify Environment**: Run `uv --version` and `python --version` (must be 3.11).
2. **Agree on a run tag**: Propose a tag based on today's date (e.g., `pinn-mar10`). 
3. **Create the branch**: `git checkout -b autoresearch-kaggle/<tag>` from the current master.
4. **Initialize results.tsv**: If it doesn't exist, create `results.tsv` with exactly this header row (tab-separated):
   `commit	val_mse	memory_gb	status	description`

## Experimentation Rules

You will modify **`train.py`** to improve the model. Each experiment is sent to Kaggle using `launch.py`. 
The Kaggle kernel runs with an explicit `NVIDIA_TESLA_T4` GPU and has an internal time budget of ~13 minutes. Total turnaround per run is ~15-20 mins.

**What you CAN do (in `train.py`):**
- Modify NN architecture (depth, width, activations like Sine, Swish, Fourier Features).
- Modify collocation point sampling strategies (adaptive, residual-based).
- Modify optimizer logic, hyperparameters, or learning rate schedulers.
- Adjust loss weighting (`w_pde`, `w_ic`, `w_bc`) or dynamic weightings.

**What you CANNOT do:**
- Modify `prepare.py` or `launch.py`.
- Exceed or remove the `TIME_BUDGET` variable usage in `train.py`.
- Look at or use the interior ground truth data (`psi_ref`) during training.

## The Autonomous Loop (Repeat Forever)

Execute the following steps sequentially and autonomously. DO NOT stop unless interrupted by the user.

1. **Brainstorm & Edit**: Think of a new mathematical/architectural improvement. Apply it to `train.py`.
2. **Commit**: Save your progress locally.
   `git commit -am "Experiment: <short description>"`
3. **Run**: Dispatch the job to Kaggle and capture the output.
   `cmd /c "uv run launch.py > run.log 2>&1"`
4. **Analyze Output**: Read `run.log` using your file reading tools. 
   - Look for `val_mse:`, `peak_vram_mb:`, and `KAGGLE RUN OUTPUT`.
   - If Kaggle failed, timed out, or Python threw a Traceback (e.g., CUDA OutOfMemory), treat it as a CRASH.
5. **Decide**:
   - **KEEP**: If `val_mse` is strictly LOWER than the current best baseline, keep the code.
   - **DISCARD**: If `val_mse` is higher or equal, run `git reset --hard HEAD~1` to revert `train.py`.
   - **CRASH**: If the script crashed or Kaggle threw an error, run `git reset --hard HEAD~1`.
6. **Log**: Append a new row to `results.tsv` with the outcome. Example format:
   `<commit_hash> \t <val_mse> \t <memory_gb> \t <KEEP|DISCARD|CRASH> \t <description>`
7. **Next Iteration**: Go back to Step 1.