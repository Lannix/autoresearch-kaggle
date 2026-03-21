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

- [x] **HYP-1.2: Multi-scale Fourier Feature Network (MsFFN)**
  - *Idea:* Replace `dde.nn.FNN` with `dde.nn.MsFFN(layer_sizes, activation, initializer, sigmas=[1, 10])`. This maps inputs to a Fourier space natively in DeepXDE, heavily reducing spectral bias.
  - *Outcome:* [KEEP] | *Delta:* [-1.197e-02 val_mse improvement]
  - *Notes:* Starting from the kept HYP-5.3 global-power baseline, replaced the plain normalized-input `tanh` core with a custom MsFFN-style front end because this DeepXDE install does not expose `dde.nn.MsFFN`: two fixed Gaussian Fourier projection banks with scales `sigma in {1.0, 10.0}`, `16` frequencies per scale, raw normalized coordinates concatenated with `sin` and `cos` features, and then the same five-layer `tanh` MLP on top. Kaggle T4 stayed stable, peak VRAM rose modestly to `2104.2 MB`, the parameter count increased to `74882`, and both Adam and L-BFGS converged much more cleanly, improving `val_mse` from `5.661969e-02` to `4.465095e-02`, so this multi-scale Fourier encoding is the new best-performing architecture in the current DeepXDE pipeline.

- [x] **HYP-1.3: Deep Residual Network (ResNet)**
  - *Idea:* Use `dde.nn.ResNet` to safely increase network depth. Breather dynamics might require a deeper manifold than a standard MLP can provide within 30 mins.
  - *Outcome:* [DISCARD] | *Delta:* [+7.910e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-1.2 MsFFN-style baseline, replaced the plain five-layer `tanh` head with a deeper custom residual MLP because this DeepXDE install does not expose `dde.nn.ResNet`: the same multi-scale Fourier encoder feeds a `128`-wide input layer, then `5` residual blocks with two linear layers each and tanh activations, followed by the output projection, while leaving the hard-IC ansatz, Gaussian-biased collocation, and global-power prior unchanged. Kaggle T4 stayed numerically stable, but the residual stack nearly doubled the parameter count to `173954`, raised peak VRAM sharply to `3683.2 MB`, cut total progress to `5955` steps, and regressed `val_mse` from `4.465095e-02` to `5.256128e-02`, so the heavier residual core was too expensive for the current 30-minute budget.

- [x] **HYP-1.4: Modified MLP (Wang et al. 2021)**
  - *Idea:* Build a custom PyTorch module with temporal/spatial gating mechanisms and pass it to DeepXDE. Greatly improves gradient flow.
  - *Outcome:* [DISCARD] | *Delta:* [+2.208e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.8 Gaussian-biased static-collocation baseline, replaced the inner `dde.nn.FNN` with a Wang-style modified MLP that learns two global projections `U` and `V` from normalized inputs and blends them through tanh gate layers of the form `H = (1 - Z) * U + Z * V`. The Kaggle T4 run stayed stable and L-BFGS still refined the model, but final `val_mse` regressed from `5.800351e-02` to `6.021131e-02`, peak VRAM rose to `3098.0 MB`, and the total parameter count increased to `67458`, so this gating architecture did not improve the current hard-IC static-sampling setup within the 30-minute budget.

- [x] **HYP-1.5: Polar Coordinate Representation / Phase-Amplitude**
  - *Idea:* Let the network output Amplitude $A$ and Phase $\phi$ instead of $u$ and $v$. Inside the `pde(x, y)` function, reconstruct $u = A \cos(\phi)$ and $v = A \sin(\phi)$ *before* computing derivatives.
  - *Outcome:* [DISCARD] | *Delta:* [+1.178e-01 val_mse regression]
  - *Notes:* Starting from the kept HYP-4.6 normalized chain-rule baseline, reinterpreted the core network's two outputs as a polar residual `(A, phi)`, converted that residual to Cartesian form via `(A cos(phi), A sin(phi))`, and only then applied the exact Fourier hard-IC output gate. The Kaggle T4 run stayed numerically stable, used essentially the same VRAM (`1986.3 MB`), and drove the PDE loss down smoothly through Adam and L-BFGS, but final `val_mse` still regressed badly from `5.809172e-02` to `1.759055e-01`, suggesting that the residual polar parameterization made the PDE-only objective easier to satisfy locally while hurting field-level generalization on the isolated validation grid.

- [x] **HYP-1.6: Smooth Non-Saturating Activations (SiLU / GELU)**
  - *Idea:* Replace the `tanh` hidden activations in the kept normalized hard-IC model with a smoother non-saturating activation such as `silu` or `gelu` to reduce vanishing gradients while keeping second derivatives stable.
  - *Outcome:* [DISCARD] | *Delta:* [+3.414e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-4.6 normalized chain-rule baseline, swapped the core `dde.nn.FNN` activation from `tanh` to `silu` while leaving the exact Fourier hard-IC ansatz, normalized-coordinate PDE residual, and optimizer schedule unchanged. The Kaggle T4 run stayed numerically stable and Adam descended quickly, but final `val_mse` regressed from `5.809172e-02` to `6.150582e-02`, peak VRAM rose sharply to `3330.2 MB`, and the L-BFGS phase effectively stalled after a few extra iterations, so this simple activation swap did not improve the current architecture.

- [x] **HYP-1.7: GELU Activation on the Normalized Hard-IC Model**
  - *Idea:* Replace the `tanh` hidden activations in the kept normalized hard-IC model with `gelu` to keep smooth second derivatives while testing a softer nonlinearity than `silu`.
  - *Outcome:* [DISCARD] | *Delta:* [+1.582e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-4.6 normalized chain-rule baseline, swapped the core `dde.nn.FNN` activation from `tanh` to `gelu` while leaving the exact Fourier hard-IC ansatz, normalized-coordinate PDE residual, and optimizer schedule unchanged. Kaggle T4 stayed stable and let L-BFGS run much longer than the SiLU variant, but final `val_mse` still regressed from `5.809172e-02` to `5.967371e-02`, peak VRAM increased to `2494.0 MB`, and the smoother activation still failed to beat the original `tanh` baseline, so `gelu` is better than `silu` here but not a new best.

- [x] **HYP-1.8: Random Fourier Features Before the Normalized MLP**
  - *Idea:* Keep the normalized hard-IC architecture, but replace the raw 2D normalized input to the `tanh` MLP with a fixed non-trainable random Fourier feature encoding `[\cos(Bx), \sin(Bx)]` so the network can represent higher-frequency structure without relying on the first layer to synthesize sinusoidal bases from scratch.
  - *Outcome:* [DISCARD] | *Delta:* [+8.022e-04 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, inserted a fixed random Fourier encoding with `32` frequencies and Gaussian scale `sigma = 4.0` ahead of the existing five-layer `tanh` core, expanding the first learned layer from `2` inputs to `64` encoded features while leaving the exact Fourier hard-IC ansatz, normalized chain-rule PDE residual, and optimizer schedule unchanged. Kaggle T4 trained stably and L-BFGS still refined the model down to `10373` total steps, but final `val_mse` regressed from `5.666258e-02` to `5.746475e-02`, peak VRAM rose to `2093.5 MB`, and parameter count increased to `74626`, so this fixed RFF front-end added capacity without beating the simpler normalized-coordinate baseline.

- [x] **HYP-1.9: Wider, Shallower Normalized Hard-IC MLP**
  - *Idea:* Keep the normalized hard-IC architecture and beta-biased static collocation, but trade depth for width by replacing the five hidden `128`-unit `tanh` layers with a shallower wider stack such as `[256] * 3` to improve GPU throughput and second-order refinement under the same wall-clock budget.
  - *Outcome:* [DISCARD] | *Delta:* [+2.625e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, replaced the core `tanh` network `[128] * 5` with a shallower wider `[256, 256, 256]` stack while leaving the exact Fourier hard-IC ansatz, normalized chain-rule PDE residual, static collocation anchors, and optimizer schedule unchanged. Kaggle T4 did use more of the available VRAM (`2319.1 MB`) and doubled the parameter count to `132866`, but it did not improve throughput: Adam only reached `5000` steps before the time cap instead of `6000`, total progress fell to `9346` steps, and final `val_mse` regressed from `5.666258e-02` to `5.928781e-02`, so this width-for-depth trade was slower and less accurate on the current budget.

- [x] **HYP-1.10: Three-Scale MsFFN Feature Bank Search**
  - *Idea:* Since HYP-1.2 was the strongest improvement so far, keep the same hard-IC MsFFN pipeline but retune the Fourier bank itself: use three smoother scales instead of two coarse scales so the network gets denser low/mid/high frequency coverage without materially increasing the parameter count.
  - *Outcome:* [DISCARD] | *Delta:* [+4.240e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-1.2 MsFFN-style baseline, replaced the original two-scale Fourier bank `sigmas=(1.0, 10.0), features_per_scale=16` with a three-scale bank `sigmas=(0.5, 2.0, 8.0), features_per_scale=12` so the encoded feature budget stayed nearly flat while covering low, medium, and high frequencies more evenly. Kaggle T4 stayed numerically stable and parameter count only rose slightly to `75906`, but the new bank regressed `val_mse` from `4.465095e-02` to `4.889051e-02`, increased peak VRAM slightly to `2122.7 MB`, and showed noisier Adam behavior around steps `2000` and `5000`, so the broader three-scale encoding was less effective than the original stronger high-frequency two-scale MsFFN.

## Category 2: Advanced SciML Architectures (PIKAN, Separable, Complex)
Cutting-edge models from 2024-2026 to drastically reduce parameter count and increase speed.

- [x] **HYP-2.1: Physics-Informed Kolmogorov-Arnold Networks (PIKAN)**
  - *Idea:* Implement a custom PyTorch KAN (learnable B-splines/wavelets on edges) and pass to DeepXDE. PIKANs achieve standard MLP accuracy with ~30x fewer parameters, making them incredibly fast and budget-friendly.
  - *Outcome:* [DISCARD] | *Delta:* [+3.187e-02 val_mse regression]
  - *Notes:* Starting from the kept HYP-1.2 MsFFN-style baseline, replaced the dense `tanh` head with a custom spline PIKAN-style core: the same fixed multi-scale Fourier encoder feeds two hidden KAN layers of width `48`, where each edge uses a cubic B-spline basis over `8` centers plus a residual linear path, followed by the usual hard-IC ansatz, Gaussian-biased collocation, and global-power prior. Kaggle T4 stayed numerically stable, but the spline basis bookkeeping was far more expensive than expected: peak VRAM jumped to `9710.3 MB`, total progress fell to just `2215` steps, the run overran the intended time reserve to `1952.9s`, and final `val_mse` regressed from `4.465095e-02` to `7.651660e-02`, so this direct spline PIKAN implementation is not competitive with the simpler MsFFN head under the current training budget.

- [x] **HYP-2.2: Separable PINNs (SPINN)**
  - *Idea:* Implement a custom PyTorch network that processes $t$ and $\theta$ through independent sub-networks and combines them via tensor products. Can yield 10x-100x speedups in evaluation.
  - *Outcome:* [DISCARD] | *Delta:* [+4.004e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, replaced the dense 2D `tanh` core inside the normalized hard-IC wrapper with a separable architecture that sends normalized `theta` and `t` through independent three-layer `tanh` branches, reshapes both outputs into two `64`-rank bases, and combines them by per-output tensor products. Kaggle T4 stayed numerically stable, but the inductive bias was not a good fit here: Adam briefly reached a decent solution around step `3000` and then destabilized sharply by step `5000`, peak VRAM rose to `2477.0 MB`, parameter count increased to `99584`, total progress dropped to `9217` steps, and final `val_mse` regressed from `5.666258e-02` to `6.066724e-02`, so this simple separable core underperformed the original dense MLP on the current budget.

- [x] **HYP-2.3: Complex-Valued PINN (CVPINN)**
  - *Idea:* Implement a custom PyTorch network using `torch.complex64` weights. Map $(t, \theta)$ to complex domain natively, bypassing the $u, v$ split entirely.
  - *Outcome:* [DISCARD] | *Delta:* [+2.242e-02 val_mse regression]
  - *Notes:* Starting from the kept HYP-1.2 MsFFN-style baseline, replaced the real-valued pointwise head with a genuinely complex-valued core: fixed multi-scale Fourier features on normalized `(theta, t)` are cast into the complex domain, passed through `4` hidden complex linear layers with `96` channels and complex `tanh`, and the network outputs a single complex residual field that is added to the exact complex initial condition through the same hard temporal gate. The PDE was also rewritten in native complex form before splitting the residual back into real and imaginary channels for DeepXDE. Kaggle T4 stayed numerically stable and the model was parameter-efficient at only `34465` complex weights, but peak VRAM still rose to `2388.1 MB`, total progress dropped to `7631` steps, and final `val_mse` regressed from `4.465095e-02` to `6.707246e-02`, so the native complex parameterization did not beat the simpler real-valued MsFFN head under the current time budget.

## Category 3: Spectral Methods & Fourier Domain
Leveraging the periodic, frequency-rich nature of the Lugiato-Lefever Equation.

- [x] **HYP-3.1: Spectral-Informed PINN (FFT for Spatial Derivatives)**
  - *Idea:* Instead of `dde.grad.hessian`, use `torch.fft.fft` and `torch.fft.ifft` inside the `pde(x, y)` function to compute $\partial^2 \psi / \partial \theta^2$. *Requires a strictly uniform grid for collocation points.*
  - *Outcome:* [DISCARD] | *Delta:* [+2.834e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-5.3 global-power baseline, replaced the Gaussian-biased theta anchors with a time-biased tensor-product grid of `117` sampled times by `256` uniformly spaced theta points and computed the spatial second derivatives spectrally with `torch.fft.fft` and `torch.fft.ifft` instead of autograd Hessians. Kaggle T4 became much more throughput-efficient, cutting peak VRAM from `1981.6 MB` to `925.4 MB` and reaching `21049` total steps in `1480.0s`, but final `val_mse` still regressed from `5.661969e-02` to `5.945323e-02`, so the uniform spectral grid likely gave up too much peak-focused collocation quality despite the cheaper periodic derivative operator.

- [x] **HYP-3.2: Neural Spectral Methods (Fully Spectral Domain)**
  - *Idea:* Move the network output out of physical space. Parametrize $\psi$ as a Fourier series and train the network to predict time-dependent Fourier coefficients $c_k(t)$.
  - *Outcome:* [DISCARD] | *Delta:* [+1.390e+00 val_mse regression]
  - *Notes:* Starting from the kept HYP-1.2 MsFFN-style baseline, replaced the pointwise field head with a fully spectral parameterization that takes only normalized time as input, predicts residual Fourier coefficients for both real and imaginary parts across all `257` retained modes, reconstructs the field analytically in theta, and uses analytic Fourier `d^2 / dtheta^2` inside the PDE while keeping the hard-IC gate, Gaussian-biased collocation, and global-power prior. Kaggle T4 stayed numerically stable, used only `1590.7 MB` peak VRAM, and reached `15306` total steps, but the objective plateaued at very large PDE losses and final `val_mse` collapsed from `4.465095e-02` to `1.434264e+00`, so the unrestricted full-mode coefficient head was far too hard to optimize within the current time budget.

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

- [x] **HYP-4.8: Periodic Boundary Losses Only During L-BFGS**
  - *Idea:* Reuse the best explicit periodic BC loss from HYP-4.7, but enable it only during the second-order L-BFGS phase so Adam keeps the cheap baseline objective while refinement enforces boundary consistency.
  - *Outcome:* [DISCARD] | *Delta:* [+3.494e-04 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, kept Adam on the original two-channel PDE objective, then turned on the full HYP-4.7 periodic value and derivative losses only for the L-BFGS phase using `6000` fixed left/right boundary pairs and weights `[2.0, 2.0]` for the periodic channels. Kaggle T4 stayed stable and the periodic losses were driven extremely close to zero during L-BFGS (roughly `7e-06` to `2e-05` by the end), but final `val_mse` still regressed from `5.666258e-02` to `5.701195e-02`, peak VRAM rose to `2156.8 MB`, and total progress fell to `10509` steps, so restricting the periodic constraints to the second-order phase reduced the original overhead but still did not beat the simpler baseline.

## Category 5: Physics Priors & Augmented Losses (The Breather Physics)
Guide the network using known asymptotic behaviors of LLE.

- [x] **HYP-5.1: Temporal Periodicity Penalty (Breather Cycle Loss)**
  - *Idea:* Breathrs eventually oscillate with a stable period. Add a custom `OperatorBC` or append a term to the `pde` return list penalizing differences at late times: $\| \psi(t_{max}, \theta) - \psi(t_{max} - \Delta t, \theta) \|^2$.
  - *Outcome:* [DISCARD] | *Delta:* [+1.927e-05 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, added a cheap late-time pair-consistency prior by evaluating the model on `2048` evenly spaced theta points at `t_max` and `t_max - 0.9990` and returning the per-field differences as two extra loss channels with weights `[0.5, 0.5]`. Kaggle T4 stayed stable, peak VRAM remained effectively flat at `1981.6 MB`, and the added late-time channels were driven almost to zero throughout both Adam and L-BFGS, but final `val_mse` still regressed by a hair from `5.666258e-02` to `5.668185e-02`, so the temporal cycle prior is promisingly close yet still not an improvement over the simpler baseline.

- [x] **HYP-5.2: Background (CW) Matching Penalty**
  - *Idea:* Most of the spatial domain rests at a Continuous Wave (CW) background. Return an extra loss in `pde`: `(y - psi_cw) * torch.exp(-abs(dy_dtheta))` to force flat regions to match theoretical background quickly.
  - *Outcome:* [DISCARD] | *Delta:* [+1.314e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, solved the cubic CW steady-state equation from the known LLE parameters, selected the positive real root whose complex field matched the initial-condition edge mean, and added two extra loss channels `exp(-|psi_theta|) * (u - u_cw)` and `exp(-|psi_theta|) * (v - v_cw)` with mild weights `[0.5, 0.5]`. Kaggle T4 stayed stable and the chosen low-intensity CW target matched the initial edges exactly (`u=0.1635, v=-0.6600, |psi|^2=0.4624`), but the added background channels plateaued around `9.17e-03` and `1.78e-02`, peak VRAM stayed essentially flat at `1981.5 MB`, and final `val_mse` regressed from `5.666258e-02` to `5.797618e-02`, so this explicit CW-matching prior over-regularized the background without improving the overall solution.

- [x] **HYP-5.3: Global Energy (Intracavity Power) Stabilization**
  - *Idea:* Penalize the time-derivative of the integrated power $\int |\psi(t, \theta)|^2 d\theta$ at late times to force the system to settle into a stable attractor.
  - *Outcome:* [KEEP] | *Delta:* [-4.289e-05 val_mse improvement]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, added a cheap global-power stabilization channel that evaluates the model on a fixed late-time grid of `512` evenly spaced theta points across `6` times spanning the last `20%` of the trajectory, computes the integrated intracavity power by uniform quadrature, and penalizes finite-difference `dP/dt` with late-time weights and a mild loss weight of `0.5`. Kaggle T4 stayed stable, peak VRAM remained effectively flat at `1981.6 MB`, the extra power-loss channel was driven from `2.88e-03` at initialization down to roughly `1e-07`-`1e-06` during training, and final `val_mse` improved slightly from `5.666258e-02` to `5.661969e-02`, so this low-cost late-time global-energy prior is the new best-performing objective.

- [x] **HYP-5.4: Asymptotic Breather Stabilization Loss**
  - *Idea:* Add a late-time physics prior that penalizes the temporal derivative of the local intensity `|psi|^2` so the solution is nudged toward a stable long-time breather attractor after the early transient has been learned.
  - *Outcome:* [DISCARD] | *Delta:* [+5.583e-04 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, added a third PDE loss channel that computes `d|psi|^2 / dt` from the normalized-input graph, masks it by `time_frac^4` so the penalty is concentrated near the end of the trajectory, and trains with loss weights `[3.0, 3.0, 1.0]`. Kaggle T4 stayed stable and drove the new stabilization term to tiny values, but the extra regularizer raised peak VRAM to `2328.5 MB`, reduced total progress to `9375` steps, and final `val_mse` regressed from `5.666258e-02` to `5.722090e-02`, so this pointwise steady-state prior was not worth the extra training cost on the current budget.

- [x] **HYP-5.5: Spatial Parity Prior Around the Soliton Peak**
  - *Idea:* Add a symmetry prior that penalizes differences between `psi(theta_peak + dtheta, t)` and `psi(theta_peak - dtheta, t)` so the network does not waste capacity on asymmetric spatial noise.
  - *Outcome:* [DISCARD] | *Delta:* [+3.429e-04 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, exposed the soliton-center `theta_peak`, mirrored each collocation point around that peak inside `pde(x, y)`, and added two cheap parity channels `0.5 * (u - u_mirror)` and `0.5 * (v - v_mirror)` with loss weights `[3.0, 3.0, 1.0, 1.0]`. Kaggle T4 stayed stable and drove the symmetry losses down to nearly zero, but the extra mirror forward pass raised peak VRAM to `2162.8 MB`, reduced total progress to `10054` steps, and final `val_mse` regressed from `5.666258e-02` to `5.700551e-02`, so the model already captures the dominant spatial parity well enough without paying for this explicit prior.

- [x] **HYP-5.6: CW Edge-Derivative Damping**
  - *Idea:* Add a cheap background prior that penalizes first spatial derivatives more strongly near the edges of the normalized domain, where the solution should resemble a flat continuous-wave background.
  - *Outcome:* [DISCARD] | *Delta:* [+5.245e-04 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, restored the global `theta_peak` variable used by the Gaussian sampler and added two "zero-cost" edge penalties using the already-computed normalized first derivatives: `edge_mask * du/dtheta_norm` and `edge_mask * dv/dtheta_norm`, with `edge_mask = theta_norm^2` and loss weights `[3.0, 3.0, 0.5, 0.5]`. Kaggle T4 stayed stable and kept peak VRAM essentially flat at `1981.7 MB`, but the edge penalties plateaued around `1.4e-03` and `3.7e-03`, final `val_mse` regressed from `5.666258e-02` to `5.718708e-02`, and the extra background shaping did not improve the current hard-IC beta-sampled model.

- [x] **HYP-5.7: Late-Time Cycle Consistency on the MsFFN Baseline**
  - *Idea:* Revisit the near-miss late-time periodicity prior from HYP-5.1, but now on top of the kept MsFFN plus global-power baseline. Penalize differences between `psi(t_max, theta)` and `psi(t_max - dt, theta)` on a fixed theta grid so the stronger spectral representation gets a light hint toward the final breather cycle.
  - *Outcome:* [DISCARD] | *Delta:* [+3.584e-04 val_mse regression]
  - *Notes:* Starting from the kept HYP-1.2 MsFFN-style baseline, added two late-time cycle-consistency channels on a fixed grid of `2048` theta points comparing `psi(t_max, theta)` against `psi(t_max - 0.9990, theta)` with weights `[0.5, 0.5]`, while preserving the existing global-power stabilization channel and all other training settings. Kaggle T4 stayed stable, peak VRAM remained effectively flat at `2104.3 MB`, and the added cycle losses were driven from about `1e-1` down to `1e-6`, but final `val_mse` still regressed slightly from `4.465095e-02` to `4.500937e-02`, so this late-time periodic hint remains promisingly close yet still does not beat the simpler MsFFN baseline.

- [x] **HYP-5.8: Global Power Curvature Penalty (Soft Attractor)**
  - *Idea:* Replace the current first-difference late-time global-power prior with a softer attractor penalty on the second time derivative of the integrated intracavity power, so the model is nudged toward a stable late-time breather without directly forcing zero slope too early.
  - *Outcome:* [DISCARD] | *Delta:* [+3.424e-04 val_mse regression]
  - *Notes:* Starting from the kept HYP-1.2 MsFFN-style baseline, replaced the existing late-time first-difference global-power prior with a softer second-difference curvature penalty on the integrated intracavity power, keeping the same `512 x 6` late-time evaluation grid but reducing the stabilization weight from `0.5` to `0.1`. Kaggle T4 stayed stable, peak VRAM remained effectively flat at `2104.2 MB`, and both Adam and L-BFGS converged cleanly, but final `val_mse` still regressed slightly from `4.465095e-02` to `4.499337e-02`, so the original first-derivative global-power stabilization remains the better attractor prior in this setup.

- [x] **HYP-5.9: Hard CW-Background Subtraction**
  - *Idea:* Treat the dominant continuous-wave background as known structure and let the network spend its capacity only on the breather residual by transitioning from the exact initial condition toward `psi_cw + residual`, with `psi_cw = [sqrt(f), 0]`.
  - *Outcome:* [DISCARD] | *Delta:* [+2.165e+00 val_mse regression]
  - *Notes:* Starting from the kept HYP-1.2 MsFFN-style baseline, modified the hard-IC ansatz so the model transitions from the exact initial condition toward `psi_cw + residual` with `psi_cw = [sqrt(f), 0]`, while leaving the Gaussian-beta collocation sampler, causal PDE weighting, and global-power stabilization prior unchanged. After fixing an unrelated Kaggle dataset-mount mismatch in `launch.py` and rerunning, the training stayed numerically stable but generalized very poorly: peak VRAM rose to `3194.1 MB`, total progress fell to `7415` steps, and final `val_mse` collapsed from `4.465095e-02` to `2.210094e+00`, so forcing this simple CW background as a hard prior is badly mismatched to the current breather dynamics.

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

- [x] **HYP-6.8: Gaussian-Biased Static Spatial Sampling**
  - *Idea:* Replace DeepXDE's uniform interior sampling with a fixed custom collocation set that biases `theta` samples toward the soliton region using a Gaussian centered on the initial-condition peak, while keeping time samples uniform.
  - *Outcome:* [KEEP] | *Delta:* [-8.821e-05 val_mse improvement]
  - *Notes:* Starting from the kept HYP-4.6 normalized chain-rule baseline, disabled `num_domain` sampling and supplied `30000` static anchors directly: `24000` Gaussian-biased `theta` samples plus `6000` uniform-background `theta` samples, all paired with shuffled uniform time samples. On Kaggle T4 the initial-condition peak landed at `theta = 0`, the static sampler used `sigma = 1.8813`, peak VRAM stayed flat at `1981.3 MB`, and L-BFGS refined cleanly out to `11228` total steps; the final `val_mse` improved slightly from `5.809172e-02` to `5.800351e-02`, so this cheap static bias outperformed the earlier dynamic adaptive samplers without adding runtime overhead.

- [x] **HYP-6.9: Beta-Biased Time Sampling on Static Collocation**
  - *Idea:* Keep the Gaussian-biased static `theta` anchors from HYP-6.8, but replace uniform time coverage with a fixed beta-distributed sampler that concentrates more collocation points near the early transient phase.
  - *Outcome:* [KEEP] | *Delta:* [-1.341e-03 val_mse improvement]
  - *Notes:* Starting from the kept HYP-6.8 baseline, kept the `24000` Gaussian + `6000` uniform `theta` anchor mix centered on the initial-condition peak and changed only the time coordinates: instead of shuffled uniform `linspace` values, sampled `t` from `Beta(1.0, 3.0)` mapped onto `[t_min, t_max]` so the static anchor set emphasized the chaotic startup regime near `t = 0`. On Kaggle T4 this preserved the same `1981.3 MB` peak VRAM and nearly the same wall-clock budget, but L-BFGS refined to a stronger final solution and improved `val_mse` from `5.800351e-02` to `5.666258e-02`, making this a cheap win over both uniform-time static sampling and the earlier dynamic adaptive methods.

- [x] **HYP-6.10: Mixed Beta and Uniform Time Sampling**
  - *Idea:* Keep the spatial sampler from HYP-6.9, but mix mostly early-time beta-distributed `t` samples with a smaller uniform-time slice so the collocation set still covers the late-time breather regime explicitly.
  - *Outcome:* [DISCARD] | *Delta:* [+5.275e-04 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 baseline, preserved the `24000` Gaussian + `6000` uniform `theta` anchor mix and the same `Beta(1.0, 3.0)` early-time bias, but reserved `20%` of the time coordinates for direct uniform sampling so the collocation set covered the late-time breather regime more explicitly. Kaggle T4 stayed stable, kept peak VRAM flat at `1981.3 MB`, and L-BFGS still refined to `10955` total steps, but final `val_mse` regressed from `5.666258e-02` to `5.719010e-02`; Adam also peaked earlier around step `4000`, so the extra uniform tail coverage slightly diluted the stronger transient-focused sampler.

- [x] **HYP-6.11: Stronger Beta Time Bias on Static Collocation**
  - *Idea:* Keep the Gaussian-biased static `theta` anchors from HYP-6.9, but increase the early-time emphasis further by changing the fixed time sampler from `Beta(1.0, 3.0)` to `Beta(1.0, 4.0)`.
  - *Outcome:* [DISCARD] | *Delta:* [+4.553e-04 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 baseline, kept the `24000` Gaussian + `6000` uniform `theta` anchor mix centered on the initial-condition peak and changed only the fixed time sampler from `Beta(1.0, 3.0)` to the more transient-heavy `Beta(1.0, 4.0)`. Kaggle T4 stayed stable, preserved the same `1981.3 MB` peak VRAM, and refined cleanly to `10957` total steps, but Adam peaked earlier around step `4000` and final `val_mse` regressed from `5.666258e-02` to `5.711783e-02`, so the stronger early-time concentration over-focused the startup transient and under-served the later trajectory compared with the kept `Beta(1.0, 3.0)` sampler.

- [x] **HYP-6.12: Narrower Gaussian Spatial Bias on Static Collocation**
  - *Idea:* Keep the winning `Beta(1.0, 3.0)` time sampler from HYP-6.9 and the `80/20` Gaussian plus uniform `theta` mix, but reduce the Gaussian width so more anchors concentrate near the soliton core around `theta_peak`.
  - *Outcome:* [DISCARD] | *Delta:* [+1.107e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 baseline, preserved the `24000` Gaussian + `6000` uniform `theta` anchor mix and the winning `Beta(1.0, 3.0)` time bias, but narrowed the spatial Gaussian from `0.15 * (theta_max - theta_min)` to `0.10 * (theta_max - theta_min)` so more static anchors landed near the soliton core. Kaggle T4 stayed stable and peak VRAM remained flat at `1981.3 MB`, but Adam peaked later and less effectively, total progress fell from `11228` to `10775` steps, and final `val_mse` regressed from `5.666258e-02` to `5.776918e-02`, so the narrower spatial sampler over-concentrated on the center and degraded global coverage.

- [x] **HYP-6.13: Ten-Batch RAR Pool with 20% Static Anchors**
  - *Idea:* Expand the effective collocation set to `10x` the baseline by maintaining a `300000`-point master pool split into `10` batches of `30000`, keep `20%` of the master pool fixed, and update the dynamic `80%` batch-by-batch toward the highest residual locations with a batched RAR callback.
  - *Outcome:* [DISCARD] | *Delta:* [+2.338e-04 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, replaced the single fixed `30000`-point anchor set with a `300000`-point master pool split into `10` rotating batches of `30000`, with `60000` static anchors (`20%`) never moving and `240000` dynamic anchors updated every `600` Adam steps by replacing each active `24000`-point dynamic batch with the top-residual points from its union with `24000` fresh candidates. Kaggle T4 stayed stable, peak VRAM stayed flat at `1981.3 MB`, and the batched RAR updates consistently increased the selected residual mass versus both the current batch and the candidate batch, but final `val_mse` still regressed slightly from `5.666258e-02` to `5.689638e-02` with `10954` total steps, so the rotating 10x pool was mechanically successful without quite beating the simpler static baseline.

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

- [x] **HYP-7.3: Self-Adaptive PINN (Learnable Loss Weights)**
  - *Idea:* Make the loss weights trainable. Create `w_pde = dde.Variable(1.0)` and pass it to `model.compile(external_trainable_variables=[w_pde])`. Multiply the residual by this weight in the `pde` function to setup a min-max adversarial training dynamic.
  - *Outcome:* [DISCARD] | *Delta:* [+6.718e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, replaced the fixed PDE weighting with two trainable PyTorch logits registered as `external_trainable_variables`, then converted them through a softmax constrained to a fixed total weight of `6.0` and scaled the residual channels by the square roots of those adaptive weights so the effective MSE coefficients could move without trivially collapsing to zero. Kaggle T4 trained stably and reached very small PDE losses with essentially unchanged peak VRAM (`1981.8 MB`), but validation regressed from `5.666258e-02` to `6.338103e-02` because the optimizer drove the learned balance to a degenerate `u=6.0000, v=0.0000` split, over-focusing on one residual channel instead of improving field accuracy.

- [x] **HYP-7.4: Causal Training (Exponential Time Weights)**
  - *Idea:* In the `pde` function, multiply the returned residual by $e^{-\epsilon t}$. This enforces physical causality by forcing the network to solve early time steps first.
  - *Outcome:* [KEEP] | *Delta:* [-1.789e-03 val_mse improvement]
  - *Notes:* Weighted both PDE residual components by `exp(-2 * t_norm)` while keeping the hard-periodic feature transform and IC losses unchanged. On Kaggle T4 this improved `val_mse` from `6.858262e-01` to `6.840374e-01`, kept peak VRAM essentially flat (`1470.7 MB`), and slightly shortened total training time, so emphasizing early-time consistency appears to help this LLE trajectory.

- [x] **HYP-7.5: PDE Residual Splitting**
  - *Idea:* In `pde(x, y)`, split the LLE residual into separate linear and Kerr channels for each field component so DeepXDE can weight those pieces independently during Adam and L-BFGS.
  - *Outcome:* [DISCARD] | *Delta:* [+6.048e-01 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, replaced the two original PDE residual channels with four separate outputs: linear and Kerr pieces for each of the `u` and `v` equations, weighted uniformly as `[1.5, 1.5, 1.5, 1.5]` to preserve the previous total PDE weight of `6.0`. Kaggle T4 stayed numerically stable and peak VRAM remained flat at `1981.3 MB`, but Adam plateaued at a much worse split-objective loss, L-BFGS could only reduce the four channels to a combined loss of about `1.36e-01`, and final `val_mse` collapsed from `5.666258e-02` to `6.614328e-01`, confirming that forcing the linear and nonlinear pieces toward zero separately changes the physics too aggressively for this DeepXDE setup.

- [x] **HYP-7.6: Gradient-Enhanced Loss (gPINN)**
  - *Idea:* Compute the spatial derivative of the residual using `dde.grad.jacobian(res, x, j=1)` and return it as an extra element in the PDE return list to guide the optimizer in sharp gradient regions.
  - *Outcome:* [DISCARD] | *Delta:* [+2.140e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-4.6 normalized chain-rule baseline, added Sobolev-style gradient supervision by differentiating both PDE residual channels with respect to `theta` inside `pde(x, y)`, returning `[causal_weight * res_u, causal_weight * res_v, 0.1 * dres_u/dtheta, 0.1 * dres_v/dtheta]` with loss weights `[3.0, 3.0, 0.5, 0.5]`. Kaggle T4 stayed stable and drove the extra residual-gradient losses down smoothly, but the heavier higher-order autograd cut total progress to `3864` steps, peak VRAM ballooned to `5617.5 MB`, and final `val_mse` regressed from `5.809172e-02` to `6.023192e-02`, so this gPINN regularization was too expensive for the current 30-minute T4 budget.

- [x] **HYP-7.7: Callback-Safe L-BFGS and Lean IC Bookkeeping**
  - *Idea:* Limit PyTorch L-BFGS internal `maxiter` so DeepXDE callbacks can enforce time limits, run many outer L-BFGS iterations explicitly, and remove redundant randomly generated initial points so only the `PointSetBC` IC data is used.
  - *Outcome:* [KEEP] | *Delta:* [-6.418e-04 val_mse improvement]
  - *Notes:* Replaced the fragile IC coordinate assembly with `np.column_stack`, removed redundant `num_initial` sampling, detached evaluation inputs when gradients are not needed, and switched the optimizer setup to a bounded L-BFGS configuration after Adam. On Kaggle T4 this improved `val_mse` from `6.840374e-01` to `6.833956e-01`, kept peak VRAM essentially flat at `1469.9 MB`, and reduced total training time to `1338.3s`.

- [x] **HYP-7.8: NAdam Warmup with Gradient Clipping**
  - *Idea:* Replace the Adam warmup with `NAdam`, use an exponential decay schedule, and clip gradients during the first-order stage before handing off to the existing time-bounded L-BFGS phase.
  - *Outcome:* [DISCARD] | *Delta:* [+1.411e-02 val_mse regression]
  - *Notes:* Starting from the kept HYP-4.4 exact-Fourier hard-IC baseline, swapped the Adam phase for a custom clipped `NAdam` optimizer with `lr=2e-3`, `gamma=0.99954`, and `max_grad_norm=1.0`, while leaving the hard-IC ansatz and L-BFGS schedule unchanged. Kaggle T4 trained stably, but the optimizer plateaued at a much worse PDE-only loss surface and final `val_mse` regressed from `7.575254e-02` to `8.986620e-02`, so the benchmark-style first-order schedule did not transfer cleanly to the DeepXDE setup.

- [x] **HYP-7.9: Adam Learning-Rate Scheduler Before L-BFGS**
  - *Idea:* Add an Adam learning-rate scheduler so the first-order phase ends at a much lower step size before handing off to L-BFGS, ideally giving the second-order phase a cleaner local Hessian.
  - *Outcome:* [DISCARD] | *Delta:* [+2.794e-04 val_mse regression]
  - *Notes:* Starting from the kept HYP-4.6 normalized chain-rule baseline, kept Adam as the first-stage optimizer but replaced the static `1e-3` rate with a clamped cosine-style LambdaLR schedule that decayed from `1e-3` to `1e-4` by step `4000` and to `1e-5` by step `6000`, then held there until the time callback stopped Adam. Kaggle T4 stayed stable, pushed Adam all the way to `6000` steps, and handed L-BFGS a smoother loss surface that refined cleanly to `10999` total steps while keeping peak VRAM flat at `1981.3 MB`, but final `val_mse` still regressed slightly from `5.809172e-02` to `5.837114e-02`, so the scheduler improved optimization behavior without beating the current best validation result.

- [x] **HYP-7.10: Phase-Wise Causal Weight Annealing**
  - *Idea:* Keep the stronger early-time causal weighting during Adam, but relax it during L-BFGS so the second-order phase can spend more capacity refining the full trajectory after the transient has been learned.
  - *Outcome:* [DISCARD] | *Delta:* [+5.983e-04 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, left the Adam phase at the existing `exp(-2.0 * t_frac)` causal weighting but reduced the L-BFGS phase to `exp(-1.0 * t_frac)` so the second-order refinement stage would be less biased toward the transient. Kaggle T4 remained stable, kept peak VRAM flat at `1981.3 MB`, and reached `10865` total steps, but final `val_mse` regressed from `5.666258e-02` to `5.726087e-02`, so the stronger causal emphasis still appears beneficial even after the sampler already biases collocation points toward early times.

- [x] **HYP-7.11: Intensity-Scaled Residuals**
  - *Idea:* Keep the current hard-IC and static-collocation baseline, but scale the PDE residual channels by `1 / (1 + |psi|^2)` using detached intensity so the optimizer does not over-focus on the breather peak at the expense of the background.
  - *Outcome:* [DISCARD] | *Delta:* [+4.569e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, multiplied both PDE residual channels by a detached intensity factor `1 / (1 + |psi|^2)` on top of the existing causal weighting so high-intensity breather points contributed less to the loss. Kaggle T4 stayed stable and peak VRAM remained nearly flat at `1981.8 MB`, but the scaled objective became too easy: Adam drove the weighted loss down to `7.96e-04`, L-BFGS stopped after only `7484` total steps, and final `val_mse` regressed from `5.666258e-02` to `6.123170e-02`, so flattening the landscape this way under-trained the true physics instead of improving background fidelity.

- [x] **HYP-7.12: Mean-Absolute-Error PDE Loss**
  - *Idea:* Keep the current hard-IC and static-collocation baseline, but replace the default MSE objective with MAE so the large peak residuals do not dominate Adam and L-BFGS updates as strongly.
  - *Outcome:* [DISCARD] | *Delta:* [+5.248e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, changed both DeepXDE compile calls from the default MSE loss to `mean absolute error` while leaving the hard-IC ansatz, PDE residual, static collocation anchors, and optimizer schedule unchanged. Kaggle T4 stayed fully stable, kept peak VRAM flat at `1981.3 MB`, and even reached a healthy `11269` total steps, but the loss plateaued at much larger absolute residual values and final `val_mse` regressed from `5.666258e-02` to `6.191092e-02`, so the nonsquared objective was more robust numerically without improving the actual learned solution.

- [x] **HYP-7.13: Stronger Physical-Time Causal Weighting**
  - *Idea:* Keep the current MsFFN baseline and global-power prior, but replace the normalized-time causal factor with a stronger physical-time decay `exp(-3 * (t - t_min))` so the optimizer focuses more aggressively on the breather formation stage before refining the oscillatory tail.
  - *Outcome:* [DISCARD] | *Delta:* [+5.146e-02 val_mse regression]
  - *Notes:* Starting from the kept HYP-1.2 MsFFN-style baseline, replaced the current normalized-time causal factor `exp(-2 * t_frac)` with a much stronger physical-time decay `exp(-3 * (t - t_min))` while leaving the exact hard-IC ansatz, Gaussian-beta collocation sampler, and global-power stabilization prior unchanged. Kaggle T4 stayed numerically stable, peak VRAM remained flat at `2104.2 MB`, and the weighted PDE losses became extremely small, but validation collapsed from `4.465095e-02` to `9.610675e-02`, showing that the stronger physical-time decay over-focused the early transient and effectively under-trained the later breather dynamics.

## Category 8: Compute Precision (The 30-Min T4 Limit)

- [x] **HYP-8.1: Mixed Precision Training (FP16/FP32)**
  - *Idea:* Call `dde.config.set_default_float("mixed")`. This utilizes T4 Tensor Cores, saving ~50% VRAM and potentially doubling step throughput, allowing a wider network or more points.
  - *Outcome:* [DISCARD] | *Delta:* [+1.274e-03 val_mse regression]
  - *Notes:* Enabled DeepXDE mixed precision and set PyTorch's default device to CUDA so the framework's internal autocast path actually targeted the T4. This sharply reduced peak VRAM from `1470.6 MB` to `908.0 MB` and let Adam reach about `25000` steps within the same wall-clock budget, but the final `val_mse` worsened from `6.828914e-01` to `6.841658e-01` and L-BFGS only made a tiny follow-up improvement, so the faster mixed-precision trajectory generalized worse than the float32 baseline.

- [x] **HYP-8.2: Float64 Precision for Hessians**
  - *Idea:* Call `dde.config.set_default_float("float64")`. Stiff PDEs suffer from FP32 rounding errors in `dde.grad.hessian`, which halts L-BFGS early. FP64 slows down iterations but can drastically improve mathematical precision and final val_mse.
  - *Outcome:* [DISCARD] | *Delta:* [+3.888e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-6.9 beta-biased static-collocation baseline, switched the DeepXDE default float to `float64`, promoted the Fourier coefficients and static anchors to float64, and evaluated the final field in `torch.float64` so the entire training and inference path used higher precision. Kaggle T4 stayed stable, but the cost was severe: peak VRAM rose to `3917.4 MB`, Adam only reached `1000` steps before the phase cutoff, total progress collapsed to `2438` steps, training stretched to `1838.0s`, and final `val_mse` regressed from `5.666258e-02` to `6.055001e-02`, so the extra Hessian precision was not worth the throughput loss under the current time budget.

## Category 9: 2026 Advanced Loss Balancing & Optimization
*Based on recent SciML literature, static loss weights fail on stiff PDEs because the network gets trapped in local saddle points between the boundary conditions and the PDE residual.*

- [x] **HYP-9.1: ReLoBRaLo (Relative Loss Balancing with Random Lookback)**
  - *Idea:* Implement a simplified version of ReLoBRaLo as a DeepXDE callback. At every $N$ epochs, dynamically adjust the loss weights ($w_{pde}, w_{ic}, w_{bc}$) based on the ratio of the current losses to the losses from a random previous epoch. This prevents the "gradient pathology" where the IC/BC losses dominate the highly non-linear breather PDE.
  - *Outcome:* [DISCARD] | *Delta:* [+1.927e-04 val_mse regression]
  - *Notes:* Starting from the kept HYP-11.3 breather-tuned Fourier baseline, added a lightweight ReLoBRaLo-style callback that watched the three active loss channels `[pde_u, pde_v, power]`, sampled a random reference point from the recent loss history every `500` logged steps, and updated the shared loss-weight list in place while preserving the total weight and clamping each channel to a bounded multiple of its initial value. Kaggle T4 stayed numerically stable, the adaptive weights moved sensibly instead of collapsing (`pde_u` and `pde_v` stayed near `3`, while the power prior varied roughly between `0.29` and `0.88`), and peak VRAM remained flat at `2133.0 MB`, but final `val_mse` still regressed slightly from `3.716117e-02` to `3.735390e-02`, so this simplified ReLoBRaLo proxy improved balance without beating the strong fixed-weight baseline.

- [x] **HYP-9.2: Curriculum Learning (Time-Marching Expansion)**
  - *Idea:* Breathers exhibit "propagation failure" because the network tries to learn the chaotic late-time oscillations before understanding the early-time formation. Write a callback that starts the training domain strictly at $t \in [t_0, t_0 + \Delta t]$ and gradually expands the upper time bound $t_{max}$ as the PDE residual drops below a threshold.
  - *Outcome:* [KEEP] | *Delta:* [-2.314e-03 val_mse improvement]
  - *Notes:* Starting from the kept HYP-11.3 breather-tuned Fourier baseline, replaced the one-shot full-time collocation set with a time-marching curriculum callback that rebuilt the same Gaussian-plus-uniform spatial anchors and `Beta(1, 3)` time sampler over staged horizons at `25%`, `50%`, `75%`, and `100%` of the domain. The scheduler waited for both a minimum amount of Adam progress and a drop in the logged training loss before expanding the horizon, then forced the full time range back on before L-BFGS so second-order refinement still saw the entire PDE domain. Kaggle T4 stayed stable, the curriculum expanded cleanly through all four stages, peak VRAM remained modest at `2133.1 MB`, and the run improved `val_mse` from `3.716117e-02` to `3.484739e-02`, making this the new best result so far.

- [ ] **HYP-9.3: Bregman Gradient Descent-Ascent (BGDA) Proxy**
  - *Idea:* To simulate the 2026 BGDA saddle-point optimization, alternate the optimization step: train the network parameters to *minimize* the loss for 5 steps, then train the loss weights ($w_{pde}, w_{ic}$) to *maximize* the loss for 1 step (using `dde.Variable`). This adversarial approach forces the network to focus on the hardest parts of the breather.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

## Category 10: 2026 Next-Gen Architectures (Transolver & PIKAN)
*Scaling laws show that MLPs struggle with multi-scale phenomena like breathers. Moving to dynamic activations and attention mechanisms drastically reduces the required parameter count, fitting perfectly within a 30-min budget.*

- [x] **HYP-10.1: PINNsFormer-Lite (Temporal Attention)**
  - *Idea:* Breathers oscillate periodically in time. Build a custom PyTorch network where the temporal input $t$ passes through a 1D Multi-Head Attention layer (or a simplified Transformer encoder) *before* concatenating with the spatial features $\Theta$ and passing to a standard MLP. This mimics the "Physics-Attention" of 2026 models.
  - *Outcome:* [CRASH] | *Delta:* [backend crash before val_mse]
  - *Notes:* Starting from the kept HYP-11.2 baseline, inserted a lightweight temporal-attention block into the MsFFN core by turning the deterministic time-harmonic pairs into tokens, applying `torch.nn.MultiheadAttention`, and concatenating the attended time summary back into the encoded feature vector. Kaggle T4 crashed immediately during the very first DeepXDE loss evaluation: PyTorch's fused scaled-dot-product attention backend does not expose the higher-order backward path required by the PDE's second derivatives, raising `RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented`. This means the idea is still conceptually untested in this codebase; it needs a manual softmax attention implementation or a forced non-fused attention backend before it can be evaluated fairly.

- [ ] **HYP-10.2: Wavelet-based PIKAN (Physics-Informed KAN)**
  - *Idea:* Implement a lightweight Kolmogorov-Arnold Network (KAN) where the edge activation functions are learnable wavelets (e.g., Morlet or Mexican Hat), rather than standard B-splines. Wavelets are mathematically superior for localizing the sharp peaks of a breather in the $\Theta$ domain. Keep the network extremely small (e.g., [2, 10, 10, 2]) to maximize speed.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

- [ ] **HYP-10.3: Complex-Valued Architecture with Phase Coupling**
  - *Idea:* The 1D LLE is a complex equation, but separating it into real ($u$) and imaginary ($v$) channels destroys the phase coupling in standard MLPs. Build a PyTorch module using `torch.complex64` weights. Apply complex-valued activations (e.g., Complex Tanh or modReLU) and output a single complex tensor, computing the residual natively in complex arithmetic before splitting into absolute errors for the loss.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* ...

## Category 11: Breather-Specific Physics & Sampling
*Breathers are localized in space but oscillate in time. Uniform sampling wastes 90% of compute on the flat CW (Continuous Wave) background.*

- [x] **HYP-11.1: Targeted Breather-Peak Sampling**
  - *Idea:* We know from the Initial Condition that the breather peak is located at $\Theta \approx 0$ (or the center of the domain). Override DeepXDE's sampling to draw 70% of the collocation points strictly in a narrow spatial band around the peak, and only 30% in the background.
  - *Outcome:* [DISCARD] | *Delta:* [+4.606e-04 val_mse regression]
  - *Notes:* Starting from the kept HYP-1.2 MsFFN-style baseline, replaced the winning `80%` Gaussian plus `20%` uniform `theta` sampler with a stricter targeted scheme: `70%` of collocation points drawn uniformly inside a periodic peak band of half-width `0.12 * (theta_max - theta_min)` around `theta_peak`, and the remaining `30%` sampled only from the complementary background, while preserving the kept `Beta(1.0, 3.0)` time bias and global-power prior. Kaggle T4 stayed stable, peak VRAM remained flat at `2104.2 MB`, and L-BFGS refined cleanly to `9994` steps, but final `val_mse` regressed from `4.465095e-02` to `4.511156e-02`, so the softer Gaussian sampler still provides the best center-versus-background balance.

- [x] **HYP-11.2: R3 Sampling (Retain-Resample-Release) Callback**
  - *Idea:* Implement the 2026 ICML standard R3 sampling. Create a custom callback that triggers every 5000 epochs: it evaluates the PDE residual on a dense grid, **Retains** the top 20% highest-error points, **Releases** (drops) the lowest-error points, and **Resamples** the rest randomly. Update `model.data.replace_points()`.
  - *Outcome:* [KEEP] | *Delta:* [-1.082e-03 val_mse improvement]
  - *Notes:* Starting from the kept HYP-9.2 curriculum-learning baseline, refactored the PDE code so the pointwise causal residual could be scored independently of the nonlocal global-power prior, then added an R3 callback that waited until the curriculum reached the full time domain and triggered once at Adam step `5000`. The callback scored all `30000` active PDE anchors in batches, retained the top `20%` highest-residual points, replaced the remaining `80%` with fresh samples from the same winning Gaussian-plus-uniform `theta` sampler and `Beta(1, 3)` time bias, and preserved that refreshed anchor set into the L-BFGS phase instead of recreating a plain final-stage curriculum batch. Kaggle T4 stayed stable, peak VRAM remained flat at `2133.1 MB`, total progress reached `9511` steps, and final `val_mse` improved from `3.484739e-02` to `3.376563e-02`, making this the new best result so far.

- [x] **HYP-11.3: Breather-Tuned Fourier Features**
  - *Idea:* Standard Fourier Features (`dde.nn.MsFFN`) use randomly initialized frequencies. For a breather, we want to force the network to "see" the specific spatial and temporal frequencies. Hardcode the Fourier feature mapping to include specific harmonic frequencies: $\sin(k \Theta)$ and $\cos(\omega_b t)$, where $\omega_b$ is an educated guess of the breather oscillation frequency.
  - *Outcome:* [KEEP] | *Delta:* [-7.490e-03 val_mse improvement]
  - *Notes:* Starting from the kept HYP-1.2 MsFFN-style baseline, retained the existing random two-scale Fourier bank but augmented it with a deterministic breather-tuned feature bank: periodic theta harmonics `k in {1,2,3,4,5}` aligned to the physical domain period plus explicit time harmonics at a guessed breather period of `0.9990` using multipliers `(1, 2)`. Kaggle T4 stayed stable, peak VRAM rose only modestly to `2133.0 MB`, parameter count increased to `76674`, and the stronger hybrid encoding improved final `val_mse` from `4.465095e-02` to `3.716117e-02`, making this the new best result on the current DeepXDE pipeline.

- [x] **HYP-11.4: Earlier Full-Domain R3 Refresh**
  - *Idea:* Keep the winning single-shot R3 callback, but trigger it earlier, immediately after the curriculum reaches the full time horizon, so Adam gets more time to adapt to the retained hard points before L-BFGS takes over.
  - *Outcome:* [DISCARD] | *Delta:* [+4.218e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-11.2 baseline, kept the same one-shot `20%` retain / `80%` resample R3 logic but moved the full-domain refresh earlier from Adam step `5000` to step `4500`, immediately after the curriculum reached its final stage. The callback itself worked as intended and retained similarly hard points (`retained_mean ≈ 1.87e-01`, `retained_max ≈ 9.90e-01`) with unchanged peak VRAM at `2133.1 MB`, but the optimization trajectory got noticeably worse: the best Adam checkpoint stayed at step `4000` before the refresh, total progress fell to `9414` steps, and final `val_mse` regressed from `3.376563e-02` to `3.798376e-02`. That suggests the anchor redistribution is helpful only after the optimizer has already spent longer fitting the full-domain batch.

- [x] **HYP-11.5: Gentler R3 Refresh with Higher Retain Fraction**
  - *Idea:* Keep the winning one-shot R3 timing at Adam step `5000`, but retain more of the matured full-domain anchor set, e.g. `30%` hard points instead of `20%`, so the refresh preserves global coverage while still injecting new difficult regions.
  - *Outcome:* [DISCARD] | *Delta:* [+8.407e-04 val_mse regression]
  - *Notes:* Starting from the kept HYP-11.2 baseline, kept the same one-shot full-domain R3 schedule at Adam step `5000` but raised the retain fraction from `20%` to `30%`, so `9000` current hard anchors were preserved and only `21000` points were resampled from the winning Gaussian-plus-uniform `theta` sampler with `Beta(1, 3)` time bias. Kaggle T4 stayed fully stable, peak VRAM remained flat at `2133.1 MB`, and the post-refresh optimization recovered better than the earlier-refresh variant, but it still did not beat the current winner: final `val_mse` regressed from `3.376563e-02` to `3.460632e-02` with `9490` total steps. The weaker residual concentration (`retained_mean = 1.524e-01` versus `1.859e-01` for HYP-11.2) suggests that preserving too much of the old anchor set dilutes the benefit of the refresh.

- [x] **HYP-11.6: Raw-Residual R3 Scoring**
  - *Idea:* Keep the winning one-shot R3 refresh schedule and training loss, but rank anchors for retain/resample using the raw PDE residual before causal time weighting. The training objective should still emphasize early-time stability, while the refresh can target hard late-time breather regions that the causal weight may otherwise hide.
  - *Outcome:* [DISCARD] | *Delta:* [+2.454e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-11.2 baseline, left the training objective unchanged but refactored the R3 scorer to rank anchors by the raw PDE residual norm instead of the causally weighted one. The refresh fired on schedule at Adam step `5000`, kept the same `20%` retain fraction, and selected much more extreme points (`retained_mean = 3.641e-01`, `retained_max = 3.653e+00`) than the winning causally scored variant, while peak VRAM remained flat at `2133.1 MB` and total progress reached `9513` steps. Even though the best Adam checkpoint shifted slightly later to step `6000`, final `val_mse` still regressed from `3.376563e-02` to `3.621968e-02`, so removing causal weighting from the anchor ranking appears to over-focus the refresh on hard late-time spikes at the expense of the more balanced collocation set.

- [x] **HYP-11.7: Complementary Late-Time R3 Resampling**
  - *Idea:* Keep the winning R3 timing, retain fraction, and causal scoring, but draw the resampled `80%` of points from a mildly late-time-biased distribution instead of the same early-time `Beta(1, 3)` used by the base anchors. This lets the retained hard points cover the early-time and causal-hot regions while the fresh points improve late-time breather coverage.
  - *Outcome:* [DISCARD] | *Delta:* [+2.395e-02 val_mse regression]
  - *Notes:* Starting from the kept HYP-11.2 baseline, preserved the winning one-shot R3 timing, `20%` retain fraction, and causal residual scoring, but changed only the resampled `80%` of points to use a complementary late-time time bias `Beta(2.0, 1.5)` instead of the baseline early-time `Beta(1, 3)`. The callback fired cleanly at Adam step `5000`, peak VRAM stayed flat at `2133.1 MB`, and the PDE losses collapsed to extremely small values through L-BFGS, but validation regressed catastrophically from `3.376563e-02` to `5.771856e-02` with only `9394` total steps. That combination of tiny training loss and much worse `val_mse` suggests the late-time refresh distribution over-specialized the collocation set and broke the broad coverage that made the original causal R3 refresh effective.

- [x] **HYP-11.8: Richer Breather Time Harmonics**
  - *Idea:* Keep the winning R3-plus-curriculum pipeline intact, but expand the deterministic breather-tuned time bank from `(1, 2)` to `(1, 2, 3)` so the model sees a slightly richer temporal prior without changing the random MsFFN scales or the collocation policy.
  - *Outcome:* [DISCARD] | *Delta:* [+3.589e-03 val_mse regression]
  - *Notes:* Starting from the kept HYP-11.2 baseline, left the R3 resampler, curriculum schedule, Gaussian-beta anchor policy, and MsFFN random scales unchanged, and only expanded the deterministic breather-tuned time harmonics from `(1, 2)` to `(1, 2, 3)`. Kaggle T4 stayed stable, total progress remained healthy at `9516` steps, and the extra harmonic only modestly increased parameter count to `76930` and peak VRAM to `2136.9 MB`, but final `val_mse` still regressed from `3.376563e-02` to `3.735495e-02`. The weaker retained residual scores during R3 (`retained_mean = 1.637e-01` versus `1.859e-01` for the winner) suggest the added temporal prior made the representation a bit too diffuse rather than sharpening the useful breather structure.
