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
