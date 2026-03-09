# autoresearch-pinn

![teaser](progress.png)

*One day, frontier physics and PDE research used to be done by meat computers manually tuning learning rates, sampling strategies, and loss weights in between eating and sleeping. That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents pushing compute jobs to cloud clusters. The agents claim that we are now in the 10,205th generation of the PINN architecture, in any case no one could tell if that's right or wrong as the loss landscapes of these Physics-Informed Neural Networks have grown beyond human comprehension. This repo is the story of how it all began. - Adapted from @karpathy, autoresearch*

The idea: give an AI agent a baseline Physics-Informed Neural Network (PINN) designed to solve the **Lugiato-Lefever Equation (LLE)**. Give it access to a Free Kaggle T4 GPU via an API wrapper, and let it experiment autonomously overnight. 

The agent modifies the code, pushes it to Kaggle, trains for about 13 minutes using **only physics constraints** (initial conditions + boundary conditions + PDE residual), checks if the True Mean Squared Error (`val_mse`) against an isolated ground-truth dataset improved, keeps or discards the changes, and repeats. You wake up in the morning to a log of experiments and (hopefully) a PINN that perfectly models complex non-linear optical dynamics.

The core idea is that you're not touching any of the Python files yourself. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents.

## How it works

Unlike standard ML tasks, PINNs learn the *physics* of the problem, not just data mapping. To strictly enforce this, and to utilize free cloud GPUs, the repo is structured around these core files:

- **`launch.py`** — The Kaggle API wrapper. It bundles the training code, submits it as an isolated script to Kaggle, waits for the GPU to finish, and fetches the logs. Run by the agent locally.
- **`prepare.py`** — Executes *inside* Kaggle. Handles loading the LLE dataset, provides initial/boundary conditions to the model, and performs the final isolated ground-truth `evaluate_mse` check. **Not modified.**
- **`train.py`** — The single file the agent edits. Contains the MLP architecture, the LLE physical residuals, data collocation sampling, and the optimizer (Adam + L-BFGS). **This file is edited and iterated on by the agent**.
- **`program.md`** — Baseline instructions for the agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed time budget of ~13 minutes** on a Kaggle T4 GPU (total turnaround time ~15-20 mins). The metric is **val_mse** — lower is better. 

## The Physics Problem

The model attempts to solve the 1D Lugiato-Lefever Equation (LLE), describing microresonator frequency combs:
$$ \frac{\partial \psi}{\partial T} = - (1 + i \zeta_0) \psi + \frac{i}{2} \frac{\partial^2 \psi}{\partial \Theta^2} + i |\psi|^2 \psi + f $$

The PINN only knows the state at $t=0$ and the system parameters. It must deduce the complex spatiotemporal evolution strictly by minimizing the equation's residual over a sampled grid.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), and a Kaggle account with API credentials.

1. **Set up Kaggle API credentials:**
Create a `.env` file in the root directory:
```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

2. **Install dependencies:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

3. **Manually run a single training experiment (~15-20 min total):**
```bash
uv run launch.py
```

If the Kaggle submission succeeds and logs are successfully downloaded showing a `val_mse`, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your favorite AI coding assistant (Claude 3.5 Sonnet / GPT-4o / Cursor / Aider) in this repo, ensure it doesn't have permissions to go rogue outside the folder, and prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file acts as a system prompt and lightweight "skill" for the agent.

## Project structure

```
.env            — Kaggle credentials (you create this)
launch.py       — Submits the job to Kaggle
prepare.py      — Kaggle-side isolated evaluation (do not modify)
train.py        — PINN model, physics logic, optimization loop (agent modifies this)
program.md      — Agent instructions
pyproject.toml  — Dependencies
```

## Design choices

- **Remote Cloud Execution.** Doing AI research normally requires expensive local GPUs. By writing a wrapper (`launch.py`), the autonomous swarm can operate from a cheap laptop while dispatching heavy PDE calculations to free Kaggle T4 GPUs.
- **Strict Data Isolation.** The dataset is only used at $t=0$ for initial conditions. The agent cannot "cheat" by training on the full grid because `prepare.py` hides the ground truth and only uses it to calculate the final `val_mse` score. 
- **Single file to modify.** The agent only touches `train.py`. This keeps the context window small and diffs easily reviewable.
- **Fixed time budget.** The model must learn as much physics as possible within ~13 minutes. This forces the agent to discover efficient sampling methods (like residual-based adaptive refinement), better optimizers (L-BFGS vs Adam transitions), and optimal architectural tweaks rather than just "training longer."

## Credits & License

This project is an adaptation of [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch), tailored for scientific machine learning (SciML), Physics-Informed Neural Networks, and Cloud execution limits. 

MIT License.