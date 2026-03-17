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

- [ ] **HYP-1.1: SIREN (Sine Activations)**
  - *Idea:* Change the `activation` string in `dde.nn.FNN` to `"sin"` and set `initializer="Glorot uniform"`. Sine activations are proven to better capture high-frequency components of breathers.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

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

- [ ] **HYP-6.2: Residual-Based Adaptive Refinement (RAR)**
  - *Idea:* Periodically evaluate the PDE residual on a dense candidate grid, pick the points with the highest error, and manually add them using `data.add_anchors(X_new)`.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

- [ ] **HYP-6.3: R3 Sampling (Retain-Resample-Release)**
  - *Idea:* Write a custom callback that adds high-error points (like RAR) but also *removes* training points where the residual is near zero, avoiding propagation failure and saving compute time.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

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

- [ ] **HYP-7.2: Adam + L-BFGS Ratio Tuning**
  - *Idea:* Adjust the `iterations` limit for Adam and `maxiter` for L-BFGS/NNCG (e.g., 20k Adam + 10k second-order).
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

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

## Category 8: Compute Precision (The 30-Min T4 Limit)

- [ ] **HYP-8.1: Mixed Precision Training (FP16/FP32)**
  - *Idea:* Call `dde.config.set_default_float("mixed")`. This utilizes T4 Tensor Cores, saving ~50% VRAM and potentially doubling step throughput, allowing a wider network or more points.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

- [ ] **HYP-8.2: Float64 Precision for Hessians**
  - *Idea:* Call `dde.config.set_default_float("float64")`. Stiff PDEs suffer from FP32 rounding errors in `dde.grad.hessian`, which halts L-BFGS early. FP64 slows down iterations but can drastically improve mathematical precision and final val_mse.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...
