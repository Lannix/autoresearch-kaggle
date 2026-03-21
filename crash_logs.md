# Crash Ledger

## 2026-03-21 - HYP-10.1 temporal attention core
- Commit: `285c01b`
- Status: `OPEN`
- Summary: Lightweight `torch.nn.MultiheadAttention` inside the MsFFN core crashed before the first logged training step on Kaggle.
- Root cause: DeepXDE needs second-order derivatives of the network output with respect to the inputs, but PyTorch's fused scaled-dot-product attention path on the Kaggle T4 backend does not implement the required higher-order backward for `aten::_scaled_dot_product_efficient_attention_backward`.
- Evidence: `RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented`
- Next action: If this idea is retried, replace `torch.nn.MultiheadAttention` with an explicit matmul-softmax-matmul attention block or force a non-fused attention backend that supports higher-order autograd, then rerun from the kept `HYP-11.2` baseline.
