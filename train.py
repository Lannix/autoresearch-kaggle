"""
Autoresearch PINN training script for LLE using DeepXDE.
Contains internal time-management to guarantee final evaluation before Kaggle timeout.
"""
import os
os.environ["DDE_BACKEND"] = "pytorch"

import time
import traceback
import sys
import numpy as np
import torch
import deepxde as dde

from prepare import TIME_BUDGET, get_training_setup, evaluate_mse

# ==========================================
# 1. Initialization and Setup
# ==========================================
dde.config.set_default_float("float32")
dde.config.set_random_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")

t_start_training = time.time()

# Load isolated training setup from prepare.py
setup = get_training_setup()
t_min, t_max = setup["t_bounds"]
th_min, th_max = setup["th_bounds"]
zeta = setup["params"]["zeta"]
f = setup["params"]["f"]

ic = setup["initial_conditions"]
t0 = ic["t0"]
th0_arr = ic["th0_arr"]
u0 = ic["u0"]
v0 = ic["v0"]

# ==========================================
# 2. DeepXDE Geometry and Domain
# ==========================================
geom = dde.geometry.Interval(th_min, th_max)
timedomain = dde.geometry.TimeDomain(t_min, t_max)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# ==========================================
# 3. Physics / LLE Residual
# ==========================================
def pde(x, y):
    # x is [theta, t]
    # y is [u, v]
    u, v = y[:, 0:1], y[:, 1:2]
    
    du_dt = dde.grad.jacobian(y, x, i=0, j=1)
    dv_dt = dde.grad.jacobian(y, x, i=1, j=1)
    
    du_dth2 = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dv_dth2 = dde.grad.hessian(y, x, component=1, i=0, j=0)
    
    S = u**2 + v**2
    
    res_u = du_dt - (-u + zeta * v - 0.5 * dv_dth2 - S * v + f)
    res_v = dv_dt - (-v - zeta * u + 0.5 * du_dth2 + S * u)
    
    return[res_u, res_v]

# ==========================================
# 4. Boundary and Initial Conditions
# ==========================================
def boundary(_, on_boundary):
    return on_boundary

# Periodic conditions for value and first spatial derivative
bc_u = dde.icbc.PeriodicBC(geomtime, 0, boundary, derivative_order=0, component=0)
bc_v = dde.icbc.PeriodicBC(geomtime, 0, boundary, derivative_order=0, component=1)
bc_u_x = dde.icbc.PeriodicBC(geomtime, 0, boundary, derivative_order=1, component=0)
bc_v_x = dde.icbc.PeriodicBC(geomtime, 0, boundary, derivative_order=1, component=1)

# Points initialization setup (hstack forms [theta, t] matching our geomtime)
ic_points = np.hstack((th0_arr, np.full_like(th0_arr, t0)))
ic_u = dde.icbc.PointSetBC(ic_points, u0, component=0)
ic_v = dde.icbc.PointSetBC(ic_points, v0, component=1)

data = dde.data.TimePDE(
    geomtime,
    pde,[bc_u, bc_v, bc_u_x, bc_v_x, ic_u, ic_v],
    num_domain=30000,
    num_boundary=6000,
    num_initial=th0_arr.shape[0],
    train_distribution="pseudo",
)

# ==========================================
# 5. Neural Network Architecture
# ==========================================
net = dde.nn.FNN([2] + [128] * 5 + [2], "tanh", "Glorot uniform")

def feature_transform(x):
    t_m = torch.tensor([th_min, t_min], device=x.device, dtype=x.dtype)
    t_M = torch.tensor([th_max, t_max], device=x.device, dtype=x.dtype)
    return 2.0 * (x - t_m) / (t_M - t_m + 1e-12) - 1.0

net.apply_feature_transform(feature_transform)
model = dde.Model(data, net)

# ==========================================
# 6. Training Setup
# ==========================================
EVAL_RESERVE = 45  
max_train_time = TIME_BUDGET - EVAL_RESERVE
adam_time_limit = max_train_time * 0.70  

print(f"[INFO] Starting training. Total budget: {TIME_BUDGET}s. Reserved for eval: {EVAL_RESERVE}s.")

class TimeBasedEarlyStopping(dde.callbacks.Callback):
    def __init__(self, max_duration):
        super().__init__()
        self.max_duration = max_duration
        self.start_time = time.time()
        
    def on_epoch_end(self):
        if time.time() - self.start_time > self.max_duration:
            self.model.stop_training = True

def model_uv(t_in, th_in, need_x=False):
    # DeepXDE inputs are [theta, t]
    x = torch.cat([th_in, t_in], dim=1)
    if need_x:
        x.requires_grad_(True)
    uv = net(x)
    return uv[:, 0:1], uv[:, 1:2], x

# Loss weights order corresponds to data array: 
#[pde_u, pde_v, bc_u, bc_v, bc_u_x, bc_v_x, ic_u, ic_v]
loss_weights =[3.0, 3.0, 5.0, 5.0, 5.0, 5.0, 50.0, 50.0]
model.compile("adam", lr=1e-3, loss_weights=loss_weights)

time_callback_adam = TimeBasedEarlyStopping(adam_time_limit)
resampler_callback = dde.callbacks.PDEPointResampler(period=500, pde_points=True, bc_points=False)

try:
    print("\n[INFO] Phase 1: Adam optimization")
    losshistory, train_state = model.train(
        iterations=100000,
        callbacks=[time_callback_adam, resampler_callback],
        display_every=1000,
    )
    
    print("\n[INFO] Phase 2: L-BFGS optimization")
    time_callback_lbfgs = TimeBasedEarlyStopping(max_train_time)
    time_callback_lbfgs.start_time = t_start_training  # base it on total elapsed time overall
    
    model.compile("L-BFGS", loss_weights=loss_weights)
    losshistory, train_state = model.train(callbacks=[time_callback_lbfgs], display_every=100)
    
    total_training_time = time.time() - t_start_training

    # ==========================================
    # 7. Final Evaluation
    # ==========================================
    print(f"\n[TIME UP / CONVERGED] Mandatory evaluation at {total_training_time:.1f}s ...")
    
    net.eval()
    val_mse = evaluate_mse(model_uv, device, dtype=torch.float32)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    num_params = sum(p.numel() for p in net.parameters())

    print("\n---")
    print(f"val_mse:          {val_mse:.6e}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {train_state.step}")
    print(f"num_params:       {num_params}")
    print("---")

except Exception as e:
    print("\n[CRITICAL ERROR] Training crashed!")
    traceback.print_exc()
    sys.exit(1)
