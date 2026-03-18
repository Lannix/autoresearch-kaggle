# Autoresearch PINN (DeepXDE) - Hypothesis Checklist

**AGENT INSTRUCTIONS:**
This is your primary research roadmap for adapting the Lugiato-Lefever Equation (LLE) solver using the DeepXDE library. Keep in mind the strict **30-minute time budget on a T4 GPU**. When starting a new iteration, pick ONE untested hypothesis from this list. 
After the Kaggle run finishes, update this file:
1. Change `[ ]` to `[x]`.
2. Fill in the `Outcome:` with either **KEEP**, **DISCARD**, or **CRASH**.
3. Fill in the `Delta:` with the improvement or regression in `val_mse`.
4. Add a brief `Notes:` explaining what happened.
5. Commit this file along with `train.py` if keeping the changes, or commit it separately if reverting `train.py`.

*DeepXDE Notes:* 
- Custom PyTorch architectures must inherit from `dde.nn.pytorch.nn.NN` or `torch.nn.Module`.
- Use `dde.grad.jacobian` and `dde.grad.hessian` for all PDE derivatives unless testing spectral derivatives.
- Extra physics constraints can be added by returning multiple items in the `pde` function or via `dde.icbc.OperatorBC`.

---

## Category 1: Network Architectures (Basics, ResNet, MsFFN)
Standard MLPs with Tanh suffer from spectral bias and gradient vanishing.

- [x] **HYP-1.1: SIREN (Sine Activations)**
  - *Idea:* Change the `activation` string in `dde.nn.FNN` to `"sin"` and set `initializer="Glorot uniform"`. Sine activations are proven to better capture high-frequency components of breathers.
  - *Outcome:* [DISCARD] | *Delta:* [+4.874e-01 val_mse regression]
  - *Notes:* Replaced the kept baseline's `tanh` hidden activations with `sin` while leaving the hard-periodic input transform, causal PDE weighting, and tuned Adam plus L-BFGS schedule unchanged. The Kaggle T4 run stayed numerically stable, but validation collapsed from `6.828914e-01` to `1.170289e+00`, peak VRAM increased to `1790.4 MB`, and the optimizer never recovered from the noisier oscillatory representation, so this simple sine-activation swap is a poor fit for the current DeepXDE setup.

- [ ] **HYP-1.2: Multi-scale Fourier Feature Network (MsFFN)**
  - *Idea:* Replace `dde.nn.FNN` with `dde.nn.MsFFN(layer_sizes, activation, initializer, sigmas=[1, 10])`. This maps inputs to a Fourier space natively in DeepXDE, heavily reducing spectral bias.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

- [ ] **HYP-1.3: Deep Residual Network (ResNet)**
  - *Idea:* Use `dde.nn.ResNet` to safely increase network depth. Breather dynamics might require a deeper manifold than a standard MLP can provide within 30 mins.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

- [ ] **HYP-1.4: Modified MLP (Wang et al. 2021)**
  - *Idea:* Build a custom PyTorch module with temporal/spatial gating mechanisms and pass it to DeepXDE. Greatly improves gradient flow.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

- [ ] **HYP-1.5: Polar Coordinate Representation / Phase-Amplitude**
  - *Idea:* Let the network output Amplitude $A$ and Phase $\phi$ instead of $u$ and $v$. Inside the `pde(x, y)` function, reconstruct $u = A \cos(\phi)$ and $v = A \sin(\phi)$ *before* computing derivatives.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

- [x] **HYP-1.6: Smooth Non-Saturating Activations (SiLU / GELU)**
  - *Idea:* Replace the `tanh` hidden activations in the kept normalized hard-IC model with a smoother non-saturating activation such as `silu` or `gelu` to reduce vanishing gradients while keeping second derivatives stable.
  - *Outcome:* [DISCARD] | *Delta:* [+3.414e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-4.6 normalized chain-rule baseline, swapped the core `dde.nn.FNN` activation from `tanh` to `silu` while leaving the exact Fourier hard-IC ansatz, normalized-coordinate PDE residual, and optimizer schedule unchanged. The Kaggle T4 run stayed numerically stable and Adam descended quickly, but final `val_mse` regressed from `5.809172e-02` to `6.150582e-02`, peak VRAM rose sharply to `3330.2 MB`, and the L-BFGS phase effectively stalled after a few extra iterations, so this simple activation swap did not improve the current architecture.

- [x] **HYP-1.7: GELU Activation on the Normalized Hard-IC Model**
  - *Idea:* Replace the `tanh` hidden activations in the kept normalized hard-IC model with `gelu` to keep smooth second derivatives while testing a softer nonlinearity than `silu`.
  - *Outcome:* [DISCARD] | *Delta:* [+1.582e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-4.6 normalized chain-rule baseline, swapped the core `dde.nn.FNN` activation from `tanh` to `gelu` while leaving the exact Fourier hard-IC ansatz, normalized-coordinate PDE residual, and optimizer schedule unchanged. Kaggle T4 stayed stable and let L-BFGS run much longer than the SiLU variant, but final `val_mse` still regressed from `5.809172e-02` to `5.967371e-02`, peak VRAM increased to `2494.0 MB`, and the smoother activation still failed to beat the original `tanh` baseline, so `gelu` is better than `silu` here but not a new best.

## Category 2: Advanced SciML Architectures (PIKAN, Separable, Complex)
Cutting-edge models from 2024-2026 to drastically reduce parameter count and increase speed.

- [ ] **HYP-2.1: Physics-Informed Kolmogorov-Arnold Networks (PIKAN)**
  - *Idea:* Implement a custom PyTorch KAN (learnable B-splines/wavelets on edges) and pass to DeepXDE. PIKANs achieve standard MLP accuracy with ~30x fewer parameters, making them incredibly fast and budget-friendly.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

- [ ] **HYP-2.2: Separable PINNs (SPINN)**
  - *Idea:* Implement a custom PyTorch network that processes $t$ and $\theta$ through independent sub-networks and combines them via tensor products. Can yield 10x-100x speedups in evaluation.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

- [ ] **HYP-2.3: Complex-Valued PINN (CVPINN)**
  - *Idea:* Implement a custom PyTorch network using `torch.complex64` weights. Map $(t, \theta)$ to complex domain natively, bypassing the $u, v$ split entirely.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

## Category 3: Spectral Methods & Fourier Domain
Leveraging the periodic, frequency-rich nature of the Lugiato-Lefever Equation.

- [ ] **HYP-3.1: Spectral-Informed PINN (FFT for Spatial Derivatives)**
  - *Idea:* Instead of `dde.grad.hessian`, use `torch.fft.fft` and `torch.fft.ifft` inside the `pde(x, y)` function to compute $\partial^2 \psi / \partial \theta^2$. *Requires a strictly uniform grid for collocation points.*
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

- [ ] **HYP-3.2: Neural Spectral Methods (Fully Spectral Domain)**
  - *Idea:* Move the network output out of physical space. Parametrize $\psi$ as a Fourier series and train the network to predict time-dependent Fourier coefficients $c_k(t)$.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

## Category 4: Hard Constraints & Boundary Enforcements
Do not waste network capacity learning what is already known mathematically.

- [x] **HYP-4.1: Hard Periodic Boundary Conditions (Fourier Features)**
  - *Idea:* Remove `dde.icbc.PeriodicBC`. Instead, force exact spatial periodicity by applying `net.apply_feature_transform(transform_fn)`, where `transform_fn` maps $(t, \theta) \to (t, \cos(\theta), \sin(\theta), \dots)$.
  - *Outcome:* [KEEP] | *Delta:* [-2.602e-01 val_mse improvement]
  - *Notes:* Replaced the normalization-only transform with `(t_scaled, cos(theta), sin(theta))`, removed periodic BC losses, and trained with only PDE plus IC constraints. On Kaggle T4 this cut `val_mse` from `9.460400e-01` to `6.858262e-01`, reduced peak VRAM from `3733.9 MB` to `1470.5 MB`, and converged faster while preserving the time-budgeted Adam plus L-BFGS flow.

- [x] **HYP-4.2: Hard Initial Conditions (Time-based Ansatz)**
  - *Idea:* Remove `dde.icbc.IC`. Force the output to exactly match the IC at $t=0$ using `net.apply_output_transform(transform_fn)`. E.g., `Output = IC_interp(theta) + (1 - exp(-c*t)) * NN_out`.
  - *Outcome:* [DISCARD] | *Delta:* [+2.432e-01 val_mse regression]
  - *Notes:* Added a periodic linear interpolation of the known initial slice and enforced `psi(theta, t0)` exactly via `apply_output_transform`, removing the IC point-set losses. The run stayed stable and used slightly less VRAM, but PDE training plateaued near a total loss of `2.09` and `val_mse` worsened from `6.858262e-01` to `9.290239e-01`, suggesting the hard ansatz made the solution manifold too restrictive or introduced poor derivative behavior.

- [x] **HYP-4.3: Multi-Harmonic Feature Search for Time and Theta**
  - *Idea:* Generalize the kept hard-periodic feature transform into a Fourier bank with configurable harmonic counts for both normalized time and angle, then search for the best `(t_harmonics, theta_harmonics)` pair under the same DeepXDE training budget.
  - *Outcome:* [DISCARD] | *Delta:* [`(0, 1)` remains best among tested settings]
  - *Notes:* Sweep results: `(1, 2)` regressed to `6.940603e-01`, `(0, 2)` regressed to `6.908429e-01`, `(1, 1)` regressed to `6.940841e-01`, and the user-requested large bank `(25, 5)` regressed to `6.951220e-01`. All tested expansions were worse than the kept `(0, 1)` baseline at `6.858262e-01`; wider harmonic banks also increased parameter count without improving validation, so the current best harmonic setting remains one theta harmonic and no time harmonics.

- [x] **HYP-4.4: Exact Fourier Hard Initial-Condition Ansatz**
  - *Idea:* Replace the point-set IC losses with a hard output transform that reconstructs the exact initial slice from its Fourier series and gates the network correction with `1 - exp(-5 * (t - t0))`.
  - *Outcome:* [KEEP] | *Delta:* [-6.071e-01 val_mse improvement]
  - *Notes:* Starting from the kept HYP-7.2 baseline, removed the soft IC losses entirely and applied a hard output transform that reconstructs both real and imaginary initial-condition fields from exact `rfft` coefficients on the original theta grid, then adds the network residual through a temporal exponential gate. On Kaggle T4 this dramatically improved `val_mse` from `6.828914e-01` to `7.575254e-02`, with peak VRAM rising modestly to `1999.1 MB` and the time-bounded L-BFGS phase using nearly the full training budget to refine the PDE-only loss.

- [x] **HYP-4.5: Revisit Large Harmonic Bank on the Exact Hard-IC Baseline**
  - *Idea:* Re-test the previously discarded `(time=25, theta=5)` Fourier feature bank, but now on top of the kept exact-Fourier hard initial-condition ansatz instead of the older soft-IC baseline.
  - *Outcome:* [DISCARD] | *Delta:* [+8.051e-03 val_mse regression]
  - *Notes:* Kept the exact Fourier hard-IC output transform from HYP-4.4 and swapped the input feature transform to the earlier large harmonic bank with `25` time harmonics and `5` theta harmonics. This was far better than the old soft-IC `(25, 5)` result, but it still regressed from `7.575254e-02` to `8.380339e-02`, increased peak VRAM to `2108.8 MB`, and raised the parameter count to `74242`, so the simpler hard-IC feature set remains best.

- [x] **HYP-4.6: Domain Normalization with Exact Chain-Rule Scaling**
  - *Idea:* Feed the network only normalized physical coordinates `theta_n, t_n in [-1, 1]`, then compute PDE derivatives in normalized space and rescale them back to the physical LLE using the exact chain rule.
  - *Outcome:* [KEEP] | *Delta:* [-1.766e-02 val_mse improvement]
  - *Notes:* Starting from the kept HYP-4.4 exact-Fourier hard-IC baseline, replaced the hard-periodic feature map with a custom network that internally normalizes both physical coordinates to `[-1, 1]`, keeps the exact Fourier hard initial-condition ansatz, and exposes the normalized inputs so the PDE residual can differentiate with respect to normalized coordinates before applying the exact `dt_n / dt` and `dtheta_n / dtheta` scaling factors. On Kaggle T4 this improved `val_mse` from `7.575254e-02` to `5.809172e-02`, slightly reduced peak VRAM to `1981.3 MB`, and lowered the parameter count to `66690`, so the cleaner normalized-coordinate representation generalized better than the earlier `(t_scaled, cos(theta), sin(theta))` feature transform on the hard-IC model.

- [x] **HYP-4.7: Explicit Periodic Boundary Losses on the Normalized Hard-IC Model**
  - *Idea:* Add strict periodic boundary losses at paired `theta_min` and `theta_max` points for random times, penalizing both value mismatches `(u, v)` and first-derivative mismatches `(u_theta, v_theta)`.
  - *Outcome:* [DISCARD] | *Delta:* [+1.311e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-4.6 baseline, added `6000` paired left/right boundary times and two explicit BC losses: one for `(u, v)` continuity and one for `(u_theta, v_theta)` continuity, with the derivative BC computed in normalized coordinates and rescaled by the exact `dtheta_n / dtheta` chain-rule factor to match the prior PyTorch formulation. The BC losses themselves converged to nearly zero, but Kaggle T4 still regressed from `5.809172e-02` to `5.940263e-02`, peak VRAM jumped sharply to `4333.1 MB`, and the heavier boundary-gradient bookkeeping cut the total optimizer progress down to `5064` steps, so the current hard-IC normalized model already appears periodic enough without the extra explicit PBC terms.

## Category 5: Physics Priors & Augmented Losses (The Breather Physics)
Guide the network using known asymptotic behaviors of LLE.

- [ ] **HYP-5.1: Temporal Periodicity Penalty (Breather Cycle Loss)**
  - *Idea:* Breathrs eventually oscillate with a stable period. Add a custom `OperatorBC` or append a term to the `pde` return list penalizing differences at late times: $\| \psi(t_{max}, \theta) - \psi(t_{max} - \Delta t, \theta) \|^2$.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

- [ ] **HYP-5.2: Background (CW) Matching Penalty**
  - *Idea:* Most of the spatial domain rests at a Continuous Wave (CW) background. Return an extra loss in `pde`: `(y - psi_cw) * torch.exp(-abs(dy_dtheta))` to force flat regions to match theoretical background quickly.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

- [ ] **HYP-5.3: Global Energy (Intracavity Power) Stabilization**
  - *Idea:* Penalize the time-derivative of the integrated power $\int |\psi(t, \theta)|^2 d\theta$ at late times to force the system to settle into a stable attractor.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

## Category 6: Collocation Sampling & Adaptive Refinement
Uniform sampling is inefficient because breathers occupy a tiny fraction of the $(t, \theta)$ domain.

- [x] **HYP-6.1: Automated Collocation Resampling**
  - *Idea:* Use `dde.callbacks.PDEPointResampler(period=1000)` to routinely redraw the collocation points randomly.
  - *Outcome:* [DISCARD] | *Delta:* [+5.988e-03 val_mse regression]
  - *Notes:* Switched `TimePDE` sampling to `train_distribution="pseudo"` and resampled PDE points every 500 Adam steps. The Kaggle T4 run finished cleanly, but `val_mse` worsened from `9.460400e-01` to `9.520277e-01`, with a larger train/test gap during L-BFGS, so the extra collocation churn did not improve generalization here.

- [x] **HYP-6.2: Residual-Based Adaptive Refinement (RAR)**
  - *Idea:* Periodically evaluate the PDE residual on a dense candidate grid, pick the points with the highest error, and manually add them using `data.add_anchors(X_new)`.
  - *Outcome:* [DISCARD] | *Delta:* [+2.783e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-7.2 baseline, evaluated the causal PDE residual on a `96 x 96` candidate grid after Adam, added the top `1536` high-residual points as anchors, then ran the usual short-burst L-BFGS phase on the enlarged PDE set. The run stayed stable and peak VRAM was unchanged (`1471.0 MB`), but L-BFGS spent much longer adapting to the anchor-augmented loss and final `val_mse` worsened from `6.828914e-01` to `6.856742e-01`, so this one-shot RAR pass over-focused the sampled residual hot spots without improving global generalization.

- [x] **HYP-6.3: R3 Sampling (Retain-Resample-Release)**
  - *Idea:* Write a custom callback that adds high-error points (like RAR) but also *removes* training points where the residual is near zero, avoiding propagation failure and saving compute time.
  - *Outcome:* [DISCARD] | *Delta:* [+1.201e-02 val_mse regression]
  - *Notes:* Implemented an R3-style callback during Adam that evaluated the causal PDE residual on the current `30000` PDE points every `1000` steps, retained only points with residual above the current mean, and refilled the rest with fresh interior samples so the collocation budget stayed fixed. The retain rate stabilized around `19%` to `31%`, but Kaggle T4 regressed from `6.828914e-01` to `6.949009e-01` and peak VRAM rose to `2590.8 MB`, so aggressively releasing low-residual points disrupted the stronger baseline distribution more than it helped.

- [x] **HYP-6.5: Residual-Based Adaptive Distribution (RAD)**
  - *Idea:* Periodically replace the PDE collocation set with a new set sampled from a residual-weighted PDF `p(x) ∝ ε(x)^k / E[ε(x)^k] + c`, using the paper's default `k = 1`, `c = 1`.
  - *Outcome:* [DISCARD] | *Delta:* [+3.584e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-7.2 baseline, added a RAD callback that every `2000` Adam steps scored `60000` fresh candidate points by the causal PDE residual, built the paper-style residual PDF with `k = 1`, `c = 1`, and replaced the `30000` PDE collocation points with a weighted sample from that pool. The sampled sets consistently had higher average residual than the candidate pools, so the biasing worked mechanically, but Kaggle T4 still regressed from `6.828914e-01` to `6.864758e-01` and peak VRAM jumped to `3813.2 MB`, so this residual-weighted redistribution was not worth the extra second-derivative overhead in the current setup.

- [x] **HYP-6.6: RAR-D (Residual-Based Adaptive Refinement with Distribution Sampling)**
  - *Idea:* Periodically score a fresh candidate pool with the PDE residual, sample a small batch of new anchors from the paper's residual-weighted PDF, and append them to the existing collocation set rather than replacing points outright.
  - *Outcome:* [DISCARD] | *Delta:* [+1.125e-02 val_mse regression]
  - *Notes:* Starting from the kept HYP-7.2 baseline, added a callback that every `2000` Adam steps scored `20000` fresh candidate points, sampled `512` new anchors from a RAD-style PDF with `k = 1`, `c = 1`, and appended them to the PDE set until `2048` extra anchors were added. The weighted sampling strongly favored higher-residual candidates, but the enlarged PDE set still regressed from `6.828914e-01` to `6.941374e-01`, raised peak VRAM to `2196.2 MB`, and slowed the L-BFGS phase, so this incremental RAR-D variant did not improve generalization on the current budget.

- [x] **HYP-6.7: Hybrid RAR with Static Base Pool and Moving Adaptive Pool**
  - *Idea:* Keep a fixed structural collocation pool for global coverage, maintain a second adaptive pool for residual hot spots, and periodically refresh only part of that adaptive pool with the worst points from a fresh candidate set.
  - *Outcome:* [DISCARD] | *Delta:* [+1.053e-02 val_mse regression]
  - *Notes:* Starting from the kept HYP-4.4 exact-Fourier hard-IC baseline, replaced the single `30000`-point PDE sample with `20000` static Hammersley base points plus `10000` pseudo-random adaptive points, then during Adam refreshed `1000` adaptive points every `2500` steps by retaining the current worst seekers and injecting the top residual outliers from `20000` new candidates. The hybrid logic worked mechanically and completed two refreshes before the Adam time cap, but it regressed from `7.575254e-02` to `8.628124e-02`, held peak VRAM essentially flat at `1999.1 MB`, and left L-BFGS spending the rest of the budget recovering from a worse collocation distribution, so the stronger hard-IC baseline still prefers the simpler fixed PDE set.

- [x] **HYP-6.4: Quasi-Random Sequences (Sobol/Halton)**
  - *Idea:* Change `train_distribution` in `dde.data.TimePDE` to `"Sobol"` or `"Halton"` to reduce sampling "holes" in the 2D domain.
  - *Outcome:* [DISCARD] | *Delta:* [+2.465e-03 val_mse regression]
  - *Notes:* Tested `train_distribution="Sobol"` on top of the kept causal-weighted HYP-7.4 model. The run completed normally but `val_mse` worsened from `6.840374e-01` to `6.865027e-01`, and Kaggle emitted Sobol balance warnings because the point counts were not powers of two, so this low-discrepancy sampling change did not help the current setup.

## Category 7: Optimization & Gradient Balancing
Fixing gradient pathologies between PDE, IC, and BC losses.

- [x] **HYP-7.1: NysNewtonCG (NNCG) Optimizer**
  - *Idea:* Replace L-BFGS with DeepXDE's advanced PyTorch optimizer: `dde.optimizers.set_NNCG_options()`, then `model.compile("NNCG")`. Highly effective for "stiff" gradients.
  - *Outcome:* [DISCARD] | *Delta:* [+3.994e-04 val_mse regression]
  - *Notes:* Kept the hard-periodic HYP-4.1 model and added an NNCG phase after L-BFGS only when more than 120 seconds remained. Kaggle T4 executed the extra phase, but the best checkpoint stayed at the pre-NNCG step, `val_mse` slipped from `6.858262e-01` to `6.862256e-01`, peak VRAM jumped from `1470.5 MB` to `4955.3 MB`, and NNCG emitted PCG non-convergence warnings, so the refinement was not worth the cost.

- [x] **HYP-7.2: Adam + L-BFGS Ratio Tuning**
  - *Idea:* Adjust the `iterations` limit for Adam and `maxiter` for L-BFGS/NNCG (e.g., 20k Adam + 10k second-order).
  - *Outcome:* [KEEP] | *Delta:* [-5.042e-04 val_mse improvement]
  - *Notes:* Starting from the kept HYP-7.7 baseline, reduced Adam to 60% of the training budget and added an explicit helper that updates DeepXDE's cached PyTorch `iter_per_step` alongside the total L-BFGS budget. Kaggle T4 reached the best Adam checkpoint around step `11000`, then improved further during short L-BFGS bursts to `val_mse = 6.828914e-01`, beating `6.833956e-01` while keeping peak VRAM flat at `1470.6 MB` and shortening total training time to `1140.4s`.

- [ ] **HYP-7.3: Self-Adaptive PINN (Learnable Loss Weights)**
  - *Idea:* Make the loss weights trainable. Create `w_pde = dde.Variable(1.0)` and pass it to `model.compile(external_trainable_variables=[w_pde])`. Multiply the residual by this weight in the `pde` function to setup a min-max adversarial training dynamic.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

- [x] **HYP-7.4: Causal Training (Exponential Time Weights)**
  - *Idea:* In the `pde` function, multiply the returned residual by $e^{-\epsilon t}$. This enforces physical causality by forcing the network to solve early time steps first.
  - *Outcome:* [KEEP] | *Delta:* [-1.789e-03 val_mse improvement]
  - *Notes:* Weighted both PDE residual components by `exp(-2 * t_norm)` while keeping the hard-periodic feature transform and IC losses unchanged. On Kaggle T4 this improved `val_mse` from `6.858262e-01` to `6.840374e-01`, kept peak VRAM essentially flat (`1470.7 MB`), and slightly shortened total training time, so emphasizing early-time consistency appears to help this LLE trajectory.

- [ ] **HYP-7.5: PDE Residual Splitting**
  - *Idea:* In `pde(x,y)`, return the linear (dispersion) and non-linear (Kerr) terms as multiple separate outputs in a list: `return [res_linear, res_nonlinear]`. Weight them independently in `model.compile(loss_weights=[...])`.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

- [ ] **HYP-7.6: Gradient-Enhanced Loss (gPINN)**
  - *Idea:* Compute the spatial derivative of the residual using `dde.grad.jacobian(res, x, j=1)` and return it as an extra element in the PDE return list to guide the optimizer in sharp gradient regions.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

- [x] **HYP-7.7: Callback-Safe L-BFGS and Lean IC Bookkeeping**
  - *Idea:* Limit PyTorch L-BFGS internal `maxiter` so DeepXDE callbacks can enforce time limits, run many outer L-BFGS iterations explicitly, and remove redundant randomly generated initial points so only the `PointSetBC` IC data is used.
  - *Outcome:* [KEEP] | *Delta:* [-6.418e-04 val_mse improvement]
  - *Notes:* Replaced the fragile IC coordinate assembly with `np.column_stack`, removed redundant `num_initial` sampling, detached evaluation inputs when gradients are not needed, and switched the optimizer setup to a bounded L-BFGS configuration after Adam. On Kaggle T4 this improved `val_mse` from `6.840374e-01` to `6.833956e-01`, kept peak VRAM essentially flat at `1469.9 MB`, and reduced total training time to `1338.3s`.

- [x] **HYP-7.8: NAdam Warmup with Gradient Clipping**
  - *Idea:* Replace the Adam warmup with `NAdam`, use an exponential decay schedule, and clip gradients during the first-order stage before handing off to the existing time-bounded L-BFGS phase.
  - *Outcome:* [DISCARD] | *Delta:* [+1.411e-02 val_mse regression]
  - *Notes:* Starting from the kept HYP-4.4 exact-Fourier hard-IC baseline, swapped the Adam phase for a custom clipped `NAdam` optimizer with `lr=2e-3`, `gamma=0.99954`, and `max_grad_norm=1.0`, while leaving the hard-IC ansatz and L-BFGS schedule unchanged. Kaggle T4 trained stably, but the optimizer plateaued at a much worse PDE-only loss surface and final `val_mse` regressed from `7.575254e-02` to `8.986620e-02`, so the benchmark-style first-order schedule did not transfer cleanly to the DeepXDE setup.

## Category 8: Compute Precision (The 30-Min T4 Limit)

- [x] **HYP-8.1: Mixed Precision Training (FP16/FP32)**
  - *Idea:* Call `dde.config.set_default_float("mixed")`. This utilizes T4 Tensor Cores, saving ~50% VRAM and potentially doubling step throughput, allowing a wider network or more points.
  - *Outcome:* [DISCARD] | *Delta:* [+1.274e-03 val_mse regression]
  - *Notes:* Enabled DeepXDE mixed precision and set PyTorch's default device to CUDA so the framework's internal autocast path actually targeted the T4. This sharply reduced peak VRAM from `1470.6 MB` to `908.0 MB` and let Adam reach about `25000` steps within the same wall-clock budget, but the final `val_mse` worsened from `6.828914e-01` to `6.841658e-01` and L-BFGS only made a tiny follow-up improvement, so the faster mixed-precision trajectory generalized worse than the float32 baseline.

- [ ] **HYP-8.2: Float64 Precision for Hessians**
  - *Idea:* Call `dde.config.set_default_float("float64")`. Stiff PDEs suffer from FP32 rounding errors in `dde.grad.hessian`, which halts L-BFGS early. FP64 slows down iterations but can drastically improve mathematical precision and final val_mse.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...
