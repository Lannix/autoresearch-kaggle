"""
Autoresearch PINN training script for LLE. Single-GPU, single-file.
Executed directly on Kaggle T4 via launch.py
"""
import time
import numpy as np
import torch
import torch.nn as nn
from prepare import get_training_setup, evaluate_mse

# Strict time budget to safely exit and report metrics before Kaggle limit
TIME_BUDGET = 800  # ~13.3 minutes
t_start_training = time.time()

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

def t32(x):
    return torch.tensor(x, device=device, dtype=DTYPE) if not isinstance(x, torch.Tensor) else x.to(device=device, dtype=DTYPE)

# 1. Setup Data from prepare.py (isolated)
setup = get_training_setup()
t_min, t_max = setup["t_bounds"]
th_min, th_max = setup["th_bounds"]
zeta, f = setup["params"]["zeta"], setup["params"]["f"]

ic = setup["initial_conditions"]
t0_t = t32(np.ones((ic["th0_arr"].shape[0], 1), dtype=np.float32) * ic["t0"])
th0_t = t32(ic["th0_arr"])
u0_t, v0_t = t32(ic["u0"]), t32(ic["v0"])

t_min_t, t_max_t = t32(t_min), t32(t_max)
th_min_t, th_max_t = t32(th_min), t32(th_max)

def norm_t(x): return 2.0 * (x - t_min_t) / (t_max_t - t_min_t + 1e-12) - 1.0
def norm_th(x): return 2.0 * (x - th_min_t) / (th_max_t - th_min_t + 1e-12) - 1.0

# 2. PINN Architecture
class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, width=128, depth=5):
        super().__init__()
        layers =[nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers +=[nn.Linear(width, width), nn.Tanh()]
        layers +=[nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

model = MLP(width=128, depth=5).to(device=device, dtype=DTYPE)

# 3. Physics Residuals
def gradients(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

def model_uv(t_in, th_in, need_x=False):
    x = torch.cat([norm_t(t_in), norm_th(th_in)], dim=1)
    if need_x: x.requires_grad_(True)
    uv = model(x)
    return uv[:, 0:1], uv[:, 1:2], x

dtn_dt_t, dthn_dth_t = t32(2.0 / (t_max - t_min + 1e-12)), t32(2.0 / (th_max - th_min + 1e-12))
zeta_t, f_t = t32(zeta), t32(f)

def lle_residual(t_in, th_in):
    u, v, x = model_uv(t_in, th_in, need_x=True)
    du_dx = gradients(u, x)
    dv_dx = gradients(v, x)
    u_tn, u_thn = du_dx[:, 0:1], du_dx[:, 1:2]
    v_tn, v_thn = dv_dx[:, 0:1], dv_dx[:, 1:2]

    u_thn2, v_thn2 = gradients(u_thn, x)[:, 1:2], gradients(v_thn, x)[:, 1:2]

    u_t, v_t = u_tn * dtn_dt_t, v_tn * dtn_dt_t
    u_th2, v_th2 = u_thn2 * (dthn_dth_t ** 2), v_thn2 * (dthn_dth_t ** 2)

    S = u**2 + v**2
    rhs_re = -(u - zeta_t * v) - 0.5 * v_th2 - S * v + f_t
    rhs_im = -(v + zeta_t * u) + 0.5 * u_th2 + S * u

    return u_t - rhs_re, v_t - rhs_im

# 4. Data Sampling & Losses
N_f = 30_000
t_f_t = t32(np.random.uniform(t_min, t_max, size=(N_f, 1)))
th_f_t = t32(np.random.uniform(th_min, th_max, size=(N_f, 1)))

N_b = 6000
t_b_t = t32(np.random.uniform(t_min, t_max, size=(N_b, 1)))
thL_t, thR_t = t32(np.ones((N_b, 1)) * th_min), t32(np.ones((N_b, 1)) * th_max)

def loss_pde():
    res_re, res_im = lle_residual(t_f_t, th_f_t)
    return (res_re**2).mean() + (res_im**2).mean()

def loss_ic():
    u_p, v_p, _ = model_uv(t0_t, th0_t, need_x=False)
    return ((u_p - u0_t)**2).mean() + ((v_p - v0_t)**2).mean()

def loss_periodic():
    uL, vL, xL = model_uv(t_b_t, thL_t, need_x=True)
    uR, vR, xR = model_uv(t_b_t, thR_t, need_x=True)
    duL, dvL = gradients(uL, xL)[:, 1:2] * dthn_dth_t, gradients(vL, xL)[:, 1:2] * dthn_dth_t
    duR, dvR = gradients(uR, xR)[:, 1:2] * dthn_dth_t, gradients(vR, xR)[:, 1:2] * dthn_dth_t
    return ((uL - uR)**2).mean() + ((vL - vR)**2).mean() + ((duL - duR)**2).mean() + ((dvL - dvR)**2).mean()

# 5. Optimization Loop
w_pde, w_ic, w_bc = 3.0, 50.0, 5.0
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

step = 0
# Allocate ~70% of time for Adam
while time.time() - t_start_training < TIME_BUDGET * 0.7:
    optimizer.zero_grad(set_to_none=True)
    loss = w_pde * loss_pde() + w_ic * loss_ic() + w_bc * loss_periodic()
    loss.backward()
    optimizer.step()
    step += 1

# Allocate rest of time for L-BFGS
lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
def closure():
    lbfgs.zero_grad(set_to_none=True)
    loss = w_pde * loss_pde() + w_ic * loss_ic() + w_bc * loss_periodic()
    loss.backward()
    return loss

while time.time() - t_start_training < TIME_BUDGET - 10:
    prev_loss = closure().item()
    lbfgs.step(closure)
    if abs(prev_loss - closure().item()) < 1e-7:
        break

total_training_time = time.time() - t_start_training

# 6. Evaluation and Reporting
val_mse = evaluate_mse(model_uv, device, dtype=DTYPE)
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
num_params = sum(p.numel() for p in model.parameters())

print("---")
print(f"val_mse:          {val_mse:.6e}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params:       {num_params}")