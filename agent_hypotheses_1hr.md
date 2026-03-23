# Autoresearch PINN (DeepXDE) - 1-Hour Hypothesis Checklist

**AGENT INSTRUCTIONS:**
This is the active research roadmap for the **1-hour training budget on a Kaggle T4 GPU**. Use this file for all new experiments. Use `agent_hypotheses_30m.md` as the historical archive of the original 30-minute loop.
After each 1-hour Kaggle run finishes, update this file:
1. Change `[ ]` to `[x]`.
2. Fill in the `Outcome:` with either **KEEP**, **DISCARD**, or **CRASH**.
3. Fill in the `Delta:` with the improvement or regression in `val_mse`.
4. Add a brief `Notes:` explaining what happened and how it relates to the 30-minute evidence.
5. Log the run in `results_1hr.tsv`.

*DeepXDE Notes:*
- Custom PyTorch architectures must inherit from `dde.nn.pytorch.nn.NN` or `torch.nn.Module`.
- Use `dde.grad.jacobian` and `dde.grad.hessian` for PDE derivatives unless a hypothesis explicitly targets a spectral alternative.
- Preserve comparability against the 30-minute champion before stacking multiple new ideas.

---

## 30-Minute Summary

The 30-minute keep chain was:
`HYP-4.1 -> HYP-4.4 -> HYP-4.6 -> HYP-5.3 -> HYP-6.8 -> HYP-6.9 -> HYP-7.2 -> HYP-7.4 -> HYP-7.7 -> HYP-9.2 -> HYP-11.2 -> HYP-11.3 -> HYP-12.4`

The final 30-minute champion is the `HYP-12.4` line:
- exact Fourier hard-IC ansatz
- normalized chain-rule PDE residual
- MsFFN-style multi-scale Fourier encoder
- global power stabilization prior
- Gaussian-biased theta sampling plus `Beta(1, 3)` time bias
- time-marching curriculum
- breather-tuned deterministic Fourier features
- progressive one-shot R3 retention scaling

Recorded 30-minute best:
- `val_mse = 3.323696e-02`
- `peak_vram_mb = 2133.1`
- `training_seconds = 1767.0`

The main 30-minute bottleneck was throughput: many promising ideas stayed stable but lost because the winning full-domain R3 refresh fired only once, or failed to fire at all.

---

## Category 0: 1-Hour Baseline and Revalidation
The first priority is to measure what the kept 30-minute champion does when the budget doubles without changing the architecture.

- [x] **1HR-0.1: True 1-Hour Baseline Revalidation**
  - *Idea:* Rerun the unchanged 30-minute champion under the new 1-hour budget before changing anything else.
  - *Outcome:* [KEEP] | *Delta:* `-2.297020e-03 val_mse improvement`
  - *Notes:* Directly revalidated the kept `HYP-12.4` 30-minute champion under `TIME_BUDGET = 3600` with no architectural or sampler changes beyond the 1-hour migration. This established the first true 1-hour baseline and confirmed the main 30-minute hypothesis about throughput: the model stayed fully stable, peak VRAM remained flat at `2133.1 MB`, and progressive R3 finally fired twice, at steps `5000` and `10000`, instead of only once. That improved `val_mse` from the archived 30-minute best `3.323696e-02` to `3.093994e-02` with `16971` total steps and `3122.3s` of training, so the current SOTA genuinely benefits from the longer budget and should remain the active 1-hour baseline for future experiments.

## Category 6: Collocation and R3 at 1 Hour
The strongest 30-minute evidence says the best model benefits from aggressive full-domain R3, but the old budget only allowed one refresh.

- [ ] **1HR-6.1: Multi-Refresh Progressive R3**
  - *Idea:* Keep the current champion fixed and tune the schedule so progressive R3 can fire multiple times during the 1-hour run.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* This is the natural 1-hour extension of `HYP-11.2` and `HYP-12.4`. In 30 minutes only the first refresh at step `5000` mattered; with 1 hour, the main hypothesis is that later exploratory-to-refinement refreshes will finally have time to contribute.

- [ ] **1HR-6.2: Denser Collocation with Preserved R3**
  - *Idea:* Retest the denser collocation set, but only in a regime where R3 still has time to fire afterward.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* This revisits `HYP-11.11`, which regressed mainly because `45000` anchors slowed Adam enough that the winning R3 event never happened. With a 1-hour budget, the denser base set may become useful if the model still reaches multiple full-domain refreshes.

- [ ] **1HR-6.3: True RAR-Style R3 Refactor**
  - *Idea:* Refactor `R3Resampler` so discarded anchors are replaced by the hardest points from a large uniformly sampled candidate pool instead of by blindly redrawing from the static Gaussian-plus-Beta prior.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* This is a direct 1-hour follow-up to `HYP-11.2`, `HYP-12.4`, and the new `1HR-0.1` baseline. The 30-minute and baseline 1-hour R3 logic still injected biased early-time points during refreshes and completely relied on the static sampler for new anchors. The 1-hour hypothesis is that true residual-screened replacement, curriculum-aware time bounds, and a small amount of uniform noise will make repeated R3 refreshes more targeted without starving later-time dynamics.

## Category 7: Optimization and Dynamic Balancing at 1 Hour
Several optimization ideas were close or clearly stable, but they spent too much of the 30-minute budget before the model could exploit them.

- [ ] **1HR-7.1: BGDA Proxy Revisit**
  - *Idea:* Retest the lightweight BGDA-style loss-weight ascent block on top of the 1-hour baseline.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* This revisits `HYP-9.3`, which was a very close discard and stayed fully stable under the 30-minute loop. The 1-hour question is whether more post-refresh optimization time lets the temporary hardest-channel emphasis pay off instead of merely perturbing the baseline.

- [ ] **1HR-7.2: Persistent Adaptive Power Prior**
  - *Idea:* Retest adaptive weighting of the global power prior, but preserve the learned weight across the full 1-hour schedule instead of letting phase transitions effectively reset it.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* This revisits `HYP-13.6`, where the adaptive power weight rose sensibly during Adam but the 30-minute two-phase loop limited how much late-time benefit it could provide. With a longer run, the key question is whether persistent late-time balancing helps once the full horizon is open longer.

## Category 10: Advanced Architectures Worth Retesting at 1 Hour
Some architecture variants were stable and conceptually promising, but they lost mainly because they reduced throughput enough to miss the winning R3 moment.

- [ ] **1HR-10.1: Factorized Space-Time Gating Revisit**
  - *Idea:* Retest factorized theta/time gating on top of the current champion under the 1-hour budget.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* This revisits `HYP-10.7`, which was stable and relatively close but slowed Adam enough that the `5000`-step R3 refresh never fired in the 30-minute run. With more time, this is one of the clearest architecture retries.

- [ ] **1HR-10.2: Lightweight PIKAN Recipe Revisit**
  - *Idea:* Retest the lightweight adaptive-basis PIKAN-style head under a budget where it can actually reach the full curriculum and R3 stages.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* This connects `HYP-12.1` and the broader imported `HYP-14.3` recipe. The 30-minute version was stable but too slow, never unlocked late curriculum stages, and never reached the winning R3 event, so 1 hour is the first fair test of whether the extra basis flexibility is useful rather than merely expensive.

## Category 14: Full Recipe Imports for 1 Hour
These are full-procedure experiments that are more plausible once the loop has enough time for multiple training phases or multiple adaptive events.

- [ ] **1HR-14.1: TMA-PINN Recipe Revisit**
  - *Idea:* Retest the TMA-PINN two-stage adaptive recipe, but let the second stage run long enough for more than one meaningful focused event.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* This revisits `HYP-14.1`, which stayed stable but over-specialized the collocation set in a short run. The 1-hour hypothesis is that the focused second stage may work better when the model has more post-refresh optimization time and potentially more than one adaptive event.

- [ ] **1HR-14.2: Strong Causal PINN / bc-PINN Recipe**
  - *Idea:* Implement the imported strong-causality recipe as a whole training procedure rather than another local causal-weight tweak.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* This is the first fair test of imported `HYP-14.2`. It belongs in the 1-hour board because it is fundamentally a procedure-level schedule and would likely be too restrictive if squeezed into the old 30-minute loop.

- [ ] **1HR-14.3: PINO/FNO Hybrid Operator-Loss Recipe**
  - *Idea:* Test a hybrid operator-loss model with spectral operator layers plus PDE residual supervision.
  - *Outcome:* [ ] | *Delta:* [ ]
  - *Notes:* This corresponds to imported `HYP-14.4`. It is intentionally a stretch item: the 30-minute archive suggests operator-style models were stable but slower, so this belongs after the cheaper 1-hour retries above.
