# Crash Ledger

## 2026-03-21 - HYP-10.1 temporal attention core
- Commit: `285c01b`
- Retry commit: `941234a`
- Status: `FIXED`
- Summary: Lightweight `torch.nn.MultiheadAttention` inside the MsFFN core crashed before the first logged training step on Kaggle.
- Root cause: DeepXDE needs second-order derivatives of the network output with respect to the inputs, but PyTorch's fused scaled-dot-product attention path on the Kaggle T4 backend does not implement the required higher-order backward for `aten::_scaled_dot_product_efficient_attention_backward`.
- Evidence: `RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented`
- Resolution: Replaced `torch.nn.MultiheadAttention` with an explicit matmul-softmax-matmul attention block and reran from the kept `HYP-11.2` baseline. Kaggle completed without any higher-order autograd errors and produced `val_mse = 4.160770e-02`, so the crash is closed even though the experiment was discarded on accuracy.
- Next action: None. The backend crash is fixed; only model quality kept this hypothesis from winning.

## 2026-03-23 - HYP-10.2 wavelet PIKAN core
- Commit: `e77ae32`
- Status: `OPEN`
- Summary: A compact Morlet-wavelet KAN head on top of the winning MsFFN encoder crashed almost immediately on Kaggle.
- Root cause: The edge-wise wavelet tensors created a much larger higher-order autograd graph than the plain MLP head. DeepXDE hit the second-theta-derivative path in `compute_weighted_pde_residuals` and exhausted T4 memory before the first logged training step.
- Evidence: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 294.00 MiB ... 13.92 GiB is allocated by PyTorch`
- Resolution: Not fixed in this run. The experiment is being reverted to the kept `HYP-12.4` baseline.
- Next action: If retried, first remove `LayerNorm` from the wavelet head and/or reduce the wavelet edge tensor sizes so the Hessian path stays within T4 memory.
