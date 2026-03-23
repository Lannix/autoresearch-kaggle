# autoresearch-pinn (Windows 10 Edition)

This is an experiment to have the AI autonomously optimize a Physics-Informed Neural Network (PINN) for solving the Lugiato-Lefever Equation (LLE) strictly on a Kaggle T4 GPU utilizing the DeepXDE framework.

**ENVIRONMENT:** You are operating on a Windows 10 system using CMD/PowerShell. Use Windows-compatible terminal commands or your built-in file editing/reading tools.

Locally:
- uv version is 0.8.3
- python in uv is 3.11.6 (destination in relative path `.venv\Scripts\Activate.ps1`)
Kaggle:
- python version is 3.12.12
- torch version is 2.9.0+cu126

Use DeepXDE (stable) in Kaggle.

## Setup Phase (Run Once)

1. **Verify Environment**: Run `uv --version` and `python --version` (must be 3.11).
2. **Agree on a run tag**: Propose a tag based on today's date (e.g., `pinn-mar10`). 
3. **Create the branch**: `git checkout -b autoresearch-kaggle/<tag>` from the current master.
4. **Initialize results.tsv**: Keep `results.tsv` as the historical 30-minute ledger.
5. **Initialize results_1hr.tsv**: If it doesn't exist, create `results_1hr.tsv` with exactly this header row (tab-separated):
   `commit	val_mse	memory_gb	status	description`
6. **Locate agent_hypotheses_1hr.md**: Ensure this file exists in the root directory. This is the active 1-hour inspiration board and tracking sheet.
7. **Locate agent_hypotheses_30m.md**: Ensure this file exists in the root directory. This is the historical 30-minute archive and should be used for lessons and retest ideas.
8. **Locate crash_logs.md**: Ensure this file exists in the root directory. This is the crash triage ledger for failed experiments and must be updated whenever an experiment crashes or a past crash is revisited.

## Experimentation Rules

You will modify **`train.py`** to improve the PINN. Each experiment is sent to Kaggle using `launch.py`. 
The Kaggle kernel runs with an explicit `NVIDIA_TESLA_T4` GPU and now uses an internal training budget of ~60 minutes. Kaggle timeout is intentionally set higher than the training budget so evaluation, notebook export, and log download can finish safely. Total turnaround per run is roughly ~65-75 mins.

**What you CAN do (in `train.py`):**
- Modify NN architecture (e.g., activations, skip connections, Fourier features) strictly leveraging DeepXDE structure.
- Modify collocation point sampling strategies via DeepXDE dataset objects.
- Modify optimizer logic, learning rate schedulers, or the balance between Adam and L-BFGS utilizing `dde.callbacks`.
- Adjust loss weighting (static, dynamic, or soft attention mechanisms).
- **Be Creative:** You are an autonomous AI researcher. You can pick an idea from `agent_hypotheses_1hr.md`, combine multiple ideas, or invent completely new mathematical and architectural approaches that are not on the list.

**What you CANNOT do during ordinary loop iterations:**
- Modify `prepare.py` or `launch.py` unless the user explicitly asks for an infrastructure change.
- Exceed or remove the `TIME_BUDGET` variable usage in `train.py`.
- Look at or use the interior ground truth data (`psi_ref`) during training. The network must learn the physics, not overfit the validation set.

## The Autonomous Loop (Repeat Forever)

Execute the following steps sequentially and autonomously. DO NOT stop unless interrupted by the user.

1. **Brainstorm & Plan**: 
   - Read `agent_hypotheses_1hr.md` for the active 1-hour roadmap.
   - Read `agent_hypotheses_30m.md` for historical lessons and retest candidates.
   - Read `crash_logs.md` before retrying any previously crashed experiment. Do not rerun a crash blindly.
   - Decide whether to test an unchecked hypothesis from the list, modify an existing one, or implement a brand new idea of your own design.
   - If you invent a new idea, append it to `agent_hypotheses_1hr.md` as a new entry before testing it.
2. **Edit**: Apply your chosen modifications to `train.py`.
3. **Commit Code**: Save your progress locally.
   `git commit -am "Experiment: <Short description of your idea>"`
4. **Run**: Dispatch the job to Kaggle and capture the output.
   `cmd /c "uv run launch.py > run.log 2>&1"`
5. **Analyze Output**: Read `run.log` using your file reading tools. 
   - Look for `val_mse:`, `peak_vram_mb:`, and `KAGGLE RUN OUTPUT`.
   - Determine the status: KEEP (val_mse improved), DISCARD (val_mse worsened/stagnated), or CRASH (errors out/timeout).
6. **Update Roadmap (`agent_hypotheses_1hr.md`)**: 
   - If you tested an item from the list, check the box `[x]`, and fill in Outcome, Delta, and Notes. 
   - If you tried something new that you appended in Step 1, update its status. 
   - Save the file.
7. **Update Crash Ledger (`crash_logs.md`)**:
   - If the run **crashed**, record the commit hash, a short bug summary, the most likely root cause, and the concrete action being taken next.
   - If you retried a previous crash, update that entry with the fix attempt and whether the bug is now **FIXED**, **STILL OPEN**, or **BLOCKED**.
   - Do not leave a crash undocumented.
8. **Commit or Revert**:
   - **KEEP**: Run `git commit -am "Update roadmap: KEEP <Experiment description>"`.
   - **DISCARD/CRASH**: 
     1. Commit the roadmap update: `git commit -am "Update roadmap: DISCARD/CRASH <Experiment description>"`. 
     2. Revert `train.py` to the last good state: `git checkout HEAD~1 train.py`.
     3. Commit the reversion: `git commit -am "Revert train.py after failed experiment"`.
9. **Log**: Append a new row to `results_1hr.tsv` with the outcome. Keep `results.tsv` untouched as the historical 30-minute archive. Example format:
   `<commit_hash> \t <val_mse> \t <memory_gb> \t <KEEP|DISCARD|CRASH> \t <description>`
10. **Reflect & Iterate**: Think about *why* the experiment succeeded or failed. Use this reasoning to formulate your next move. Go back to Step 1.
