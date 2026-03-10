"""
Data preparation and evaluation module inside Kaggle Environment.
Provides isolated evaluation so the model doesn't train on ground truth data.
"""
import os
import torch
import numpy as np
from scipy.io import loadmat

# Internal time budget limit
TIME_BUDGET = 1140 

# Path to the dataset on Kaggle
DATA_PATH = "/kaggle/input/datasets/technolight/matlab-conditions/Field.mat"

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

# ==============================
# BEGIN TRAIN.PY
# ==============================

"""
Autoresearch PINN training script for LLE. Single-GPU, single-file.
Contains internal time-management to guarantee final evaluation before Kaggle timeout.
"""

import time
import traceback
import sys
import numpy as np
import torch
import torch.nn as nn


# ==========================================
# 1. Initialization and Setup
# ==========================================
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

print(f"[INFO] Using device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")

def t32(x):
    return torch.tensor(x, device=device, dtype=DTYPE) if not isinstance(x, torch.Tensor) else x.to(device=device, dtype=DTYPE)

t_start_training = time.time()

# Load isolated training setup from prepare.py
setup = get_training_setup()
t_min, t_max = setup["t_bounds"]
th_min, th_max = setup["th_bounds"]
zeta = setup["params"]["zeta"]
f = setup["params"]["f"]

ic = setup["initial_conditions"]
t0_t = t32(np.ones((ic["th0_arr"].shape[0], 1), dtype=np.float32) * ic["t0"])
th0_t = t32(ic["th0_arr"])
u0_t = t32(ic["u0"])
v0_t = t32(ic["v0"])

t_min_t, t_max_t = t32(t_min), t32(t_max)
th_min_t, th_max_t = t32(th_min), t32(th_max)

def norm_t(x): return 2.0 * (x - t_min_t) / (t_max_t - t_min_t + 1e-12) - 1.0
def norm_th(x): return 2.0 * (x - th_min_t) / (th_max_t - th_min_t + 1e-12) - 1.0

# ==========================================
# 2. Neural Network Architecture
# ==========================================
class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, width=128, depth=5):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

model = MLP(width=128, depth=5).to(device=device, dtype=DTYPE)

# ==========================================
# 3. Physics & Gradients
# ==========================================
def gradients(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

def model_uv(t_in, th_in, need_x=False):
    tn, thn = norm_t(t_in), norm_th(th_in)
    x = torch.cat([tn, thn], dim=1)
    if need_x: x.requires_grad_(True)
    uv = model(x)
    return uv[:, 0:1], uv[:, 1:2], x

dtn_dt_t = t32(2.0 / (t_max - t_min + 1e-12))
dthn_dth_t = t32(2.0 / (th_max - th_min + 1e-12))
zeta_t, f_t = t32(zeta), t32(f)

def lle_residual(t_in, th_in):
    u, v, x = model_uv(t_in, th_in, need_x=True)

    du_dx = gradients(u, x)
    dv_dx = gradients(v, x)
    u_tn, u_thn = du_dx[:, 0:1], du_dx[:, 1:2]
    v_tn, v_thn = dv_dx[:, 0:1], dv_dx[:, 1:2]

    u_thn2 = gradients(u_thn, x)[:, 1:2]
    v_thn2 = gradients(v_thn, x)[:, 1:2]

    u_t, v_t = u_tn * dtn_dt_t, v_tn * dtn_dt_t
    u_th2, v_th2 = u_thn2 * (dthn_dth_t ** 2), v_thn2 * (dthn_dth_t ** 2)

    S = u**2 + v**2
    rhs_re = -(u - zeta_t * v) - 0.5 * v_th2 - S * v + f_t
    rhs_im = -(v + zeta_t * u) + 0.5 * u_th2 + S * u

    return u_t - rhs_re, v_t - rhs_im

# ==========================================
# 4. Training Data Sampling
# ==========================================
N_f = 30_000
t_f_t = t32(np.random.uniform(t_min, t_max, size=(N_f, 1)))
th_f_t = t32(np.random.uniform(th_min, th_max, size=(N_f, 1)))

N_b = 6000
t_b_t = t32(np.random.uniform(t_min, t_max, size=(N_b, 1)))
thL_t = t32(np.ones((N_b, 1)) * th_min)
thR_t = t32(np.ones((N_b, 1)) * th_max)

# ==========================================
# 5. Loss Functions
# ==========================================
def loss_pde():
    res_re, res_im = lle_residual(t_f_t, th_f_t)
    return (res_re**2).mean() + (res_im**2).mean()

def loss_ic():
    u_pred, v_pred, _ = model_uv(t0_t, th0_t, need_x=False)
    return ((u_pred - u0_t)**2).mean() + ((v_pred - v0_t)**2).mean()

def loss_periodic():
    uL, vL, xL = model_uv(t_b_t, thL_t, need_x=True)
    uR, vR, xR = model_uv(t_b_t, thR_t, need_x=True)

    duL_dth = gradients(uL, xL)[:, 1:2] * dthn_dth_t
    dvL_dth = gradients(vL, xL)[:, 1:2] * dthn_dth_t
    duR_dth = gradients(uR, xR)[:, 1:2] * dthn_dth_t
    dvR_dth = gradients(vR, xR)[:, 1:2] * dthn_dth_t

    return ((uL - uR)**2).mean() + ((vL - vR)**2).mean() + ((duL_dth - duR_dth)**2).mean() + ((dvL_dth - dvR_dth)**2).mean()

# ==========================================
# 6. Training Loop (Time Constrained)
# ==========================================
w_pde, w_ic, w_bc = 3.0, 50.0, 5.0
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Reserve 45 seconds for final evaluation to avoid Kaggle timeout
EVAL_RESERVE = 45  
max_train_time = TIME_BUDGET - EVAL_RESERVE
adam_time_limit = max_train_time * 0.70  

print(f"[INFO] Starting training. Total budget: {TIME_BUDGET}s. Reserved for eval: {EVAL_RESERVE}s.")

step = 0
try:
    # ------------------ ADAM ------------------
    while (time.time() - t_start_training) < adam_time_limit:
        optimizer.zero_grad(set_to_none=True)
        loss = w_pde * loss_pde() + w_ic * loss_ic() + w_bc * loss_periodic()
        loss.backward()
        optimizer.step()
        
        if step % 200 == 0:
            elapsed = time.time() - t_start_training
            print(f"Adam step={step:5d} loss={loss.item():.3e} | elapsed={elapsed:.1f}s")
        step += 1

    # ------------------ L-BFGS ------------------
    print(f"\n[INFO] Switching to L-BFGS refinement. Elapsed: {time.time() - t_start_training:.1f}s")
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        loss = w_pde * loss_pde() + w_ic * loss_ic() + w_bc * loss_periodic()
        loss.backward()
        return loss

    while (time.time() - t_start_training) < max_train_time:
        prev_loss = closure().item()
        lbfgs.step(closure)
        new_loss = closure().item()
        
        print(f"L-BFGS loss: {new_loss:.3e} | elapsed={time.time() - t_start_training:.1f}s")
        
        # Stop early if converged
        if abs(prev_loss - new_loss) < 1e-8:
            print("[INFO] L-BFGS converged.")
            break

    total_training_time = time.time() - t_start_training

    # ==========================================
    # 7. Final Evaluation (Mandatory)
    # ==========================================
    print(f"\n[TIME UP / CONVERGED] Mandatory evaluation at {total_training_time:.1f}s ...")
    val_mse = evaluate_mse(model_uv, device, dtype=DTYPE)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    num_params = sum(p.numel() for p in model.parameters())

    print("\n---")
    print(f"val_mse:          {val_mse:.6e}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params:       {num_params}")
    print("---")

except Exception as e:
    # Catching exceptions to provide a visible Traceback for the agent before exiting
    print("\n[CRITICAL ERROR] Training crashed!")
    traceback.print_exc()
    sys.exit(1)