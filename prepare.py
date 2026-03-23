"""
Data preparation and evaluation module inside Kaggle Environment.
Provides isolated evaluation so the model doesn't train on ground truth data.
"""
import os
import torch
import numpy as np
from scipy.io import loadmat

# Internal time budget limit - 60 minutes
TIME_BUDGET = 3600

# Path to the dataset on Kaggle
DATA_PATH = "/kaggle/input/datasets/technolight/matlab-conditions/Breather.mat"

_GROUND_TRUTH = None

def _load_ground_truth():
    global _GROUND_TRUTH
    if _GROUND_TRUTH is not None:
        return _GROUND_TRUTH

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    mat = loadmat(DATA_PATH)
    
    # Extract coordinates
    t = np.squeeze(mat["t"]).astype(np.float64)
    theta = np.squeeze(mat["theta"]).astype(np.float64)
    
    # Find field matrix (usually 'B' or 'psi')
    field_key = "B" if "B" in mat else "psi"
    psi_ref = mat[field_key]

    # Check and fix dimensions if needed
    Nt, Nth = psi_ref.shape
    if (Nt != t.size) and (Nth == t.size) and (Nt == theta.size):
        psi_ref = psi_ref.T

    _GROUND_TRUTH = {
        "t": t, "theta": theta, "psi_ref": psi_ref,
        "zeta": 4.5, "f": 8**0.5
    }
    return _GROUND_TRUTH

def get_training_setup():
    """Returns only necessary physics params and ICs. NO interior truth data."""
    gt = _load_ground_truth()
    t, theta = gt["t"], gt["theta"]
    psi0 = gt["psi_ref"][0, :]
    
    return {
        "t_bounds": (float(t.min()), float(t.max())),
        "th_bounds": (float(theta.min()), float(theta.max())),
        "initial_conditions": {
            "t0": float(t.min()),
            "th0_arr": theta.reshape(-1, 1).astype(np.float32),
            "u0": np.real(psi0).reshape(-1, 1).astype(np.float32),
            "v0": np.imag(psi0).reshape(-1, 1).astype(np.float32)
        },
        "params": {"zeta": gt["zeta"], "f": gt["f"]}
    }

@torch.no_grad()
def evaluate_mse(model_uv_fn, device, dtype=torch.float32):
    """Evaluates true MSE against isolated ground truth without OOM issues."""
    gt = _load_ground_truth()
    t = gt["t"].astype(np.float32).reshape(-1, 1)
    theta = gt["theta"].astype(np.float32).reshape(-1, 1)
    psi_ref = gt["psi_ref"]
    
    TT, TH = np.meshgrid(t.squeeze(), theta.squeeze(), indexing="ij")
    t_val = torch.tensor(TT.reshape(-1, 1), device=device, dtype=dtype)
    th_val = torch.tensor(TH.reshape(-1, 1), device=device, dtype=dtype)
    
    dataset_size = t_val.shape[0]
    batch_size = 20000 
    u_preds, v_preds = [],[]
    
    for i in range(0, dataset_size, batch_size):
        t_b = t_val[i:i+batch_size]
        th_b = th_val[i:i+batch_size]
        u_b, v_b, _ = model_uv_fn(t_b, th_b, need_x=False)
        u_preds.append(u_b.cpu().numpy())
        v_preds.append(v_b.cpu().numpy())
        
    u_pred = np.concatenate(u_preds, axis=0).reshape(psi_ref.shape)
    v_pred = np.concatenate(v_preds, axis=0).reshape(psi_ref.shape)
    
    psi_pred = u_pred + 1j * v_pred
    error_field_sq = np.abs(psi_pred - psi_ref)**2
    return float(np.mean(error_field_sq))
