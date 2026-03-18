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
    time_frac = (x[:, 1:2] - t_min) / (t_max - t_min + 1e-12)
    causal_weight = torch.exp(-2.0 * time_frac)

    return [causal_weight * res_u, causal_weight * res_v]

# ==========================================
# 4. Initial Conditions
# ==========================================
# Build a stable (N, 2) coordinate matrix even if the IC arrays become 1D.
ic_points = np.column_stack(
    (
        np.asarray(th0_arr).reshape(-1),
        np.full(th0_arr.shape[0], t0, dtype=th0_arr.dtype),
    )
)
ic_u = dde.icbc.PointSetBC(ic_points, u0, component=0)
ic_v = dde.icbc.PointSetBC(ic_points, v0, component=1)

data = dde.data.TimePDE(
    geomtime,
    pde, [ic_u, ic_v],
    num_domain=30000,
    num_boundary=0,
    num_initial=0,
)

# ==========================================
# 5. Neural Network Architecture
# ==========================================
net = dde.nn.FNN([3] + [128] * 5 + [2], "tanh", "Glorot uniform")

def feature_transform(x):
    theta = x[:, 0:1]
    time_coord = x[:, 1:2]
    t_center = torch.tensor((t_min + t_max) * 0.5, device=x.device, dtype=x.dtype)
    t_scale = torch.tensor((t_max - t_min) * 0.5 + 1e-12, device=x.device, dtype=x.dtype)
    time_scaled = (time_coord - t_center) / t_scale
    return torch.cat((time_scaled, torch.cos(theta), torch.sin(theta)), dim=1)

net.apply_feature_transform(feature_transform)
model = dde.Model(data, net)

# ==========================================
# 6. Training Setup
# ==========================================
EVAL_RESERVE = 45  
max_train_time = TIME_BUDGET - EVAL_RESERVE
adam_time_limit = max_train_time * 0.60
lbfgs_total_iters = 5000
lbfgs_inner_iters = 250
rad_period = 2000
rad_batch_size = 2048
rad_candidate_multiplier = 2
rad_power = 1.0
rad_offset = 1.0

print(f"[INFO] Starting training. Total budget: {TIME_BUDGET}s. Reserved for eval: {EVAL_RESERVE}s.")

class TimeBasedEarlyStopping(dde.callbacks.Callback):
    def __init__(self, max_duration):
        super().__init__()
        self.max_duration = max_duration
        self.start_time = time.time()
        
    def on_epoch_end(self):
        if time.time() - self.start_time > self.max_duration:
            self.model.stop_training = True


def configure_pytorch_lbfgs(total_iters, inner_iters):
    # DeepXDE caches PyTorch's per-step L-BFGS budget separately from maxiter.
    dde.optimizers.config.set_LBFGS_options(maxiter=total_iters)
    lbfgs_options = dde.optimizers.config.LBFGS_options
    inner_iters = min(inner_iters, total_iters)
    lbfgs_options["iter_per_step"] = inner_iters
    lbfgs_options["fun_per_step"] = max(
        1,
        lbfgs_options["maxfun"] * inner_iters // max(1, total_iters),
    )


def residual_scores_for_points(points_np, batch_size):
    scores = []
    for start in range(0, points_np.shape[0], batch_size):
        stop = start + batch_size
        x_batch = torch.tensor(
            points_np[start:stop],
            device=device,
            dtype=torch.float32,
        ).requires_grad_(True)
        residual_terms = pde(x_batch, net(x_batch))
        score = torch.zeros(x_batch.shape[0], device=device, dtype=torch.float32)
        for term in residual_terms:
            score = score + term[:, 0] ** 2
        scores.append(torch.sqrt(score + 1e-12).detach().cpu().numpy())
        del x_batch, residual_terms, score
    return np.concatenate(scores, axis=0)


class RADSampler(dde.callbacks.Callback):
    def __init__(self, period, batch_size, candidate_multiplier, power, offset):
        super().__init__()
        self.period = period
        self.batch_size = batch_size
        self.candidate_multiplier = candidate_multiplier
        self.power = power
        self.offset = offset

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step == 0 or step % self.period != 0:
            return

        current_points = np.asarray(self.model.data.train_x_all, dtype=np.float32)
        num_points = current_points.shape[0]
        candidate_count = self.candidate_multiplier * num_points
        candidate_points = geomtime.random_points(candidate_count, random="pseudo").astype(np.float32)
        residual_scores = residual_scores_for_points(candidate_points, self.batch_size)

        scaled_scores = np.power(np.maximum(residual_scores, 1e-12), self.power)
        weights = scaled_scores / (scaled_scores.mean() + 1e-12) + self.offset
        probabilities = weights / weights.sum()
        sampled_idx = np.random.choice(candidate_count, size=num_points, replace=False, p=probabilities)
        updated_points = candidate_points[sampled_idx]
        np.random.shuffle(updated_points)

        self.model.data.replace_with_anchors(updated_points)
        self.model.data.test_x = None
        self.model.data.test_y = None
        self.model.data.test_aux_vars = None
        self.model.train_state.set_data_train(
            self.model.data.train_x,
            self.model.data.train_y,
            self.model.data.train_aux_vars,
        )
        self.model.train_state.set_data_test(*self.model.data.test())

        sampled_scores = residual_scores[sampled_idx]
        print(
            f"[INFO] RAD resample at step {step}: candidates={candidate_count}, "
            f"mean_residual={residual_scores.mean():.3e}, sampled_mean={sampled_scores.mean():.3e}"
        )


def model_uv(t_in, th_in, need_x=False):
    # DeepXDE inputs are [theta, t]
    x = torch.cat((th_in, t_in), dim=1)
    if need_x:
        x = x.requires_grad_(True)
    else:
        x = x.detach()
    uv = net(x)
    return uv[:, 0:1], uv[:, 1:2], x

# Loss weights order corresponds to data array:
#[pde_u, pde_v, ic_u, ic_v]
loss_weights =[3.0, 3.0, 50.0, 50.0]
model.compile("adam", lr=1e-3, loss_weights=loss_weights)

time_callback_adam = TimeBasedEarlyStopping(adam_time_limit)
rad_callback = RADSampler(
    rad_period,
    rad_batch_size,
    rad_candidate_multiplier,
    rad_power,
    rad_offset,
)

try:
    print("\n[INFO] Phase 1: Adam optimization")
    losshistory, train_state = model.train(
        iterations=100000,
        callbacks=[time_callback_adam, rad_callback],
        display_every=1000,
    )
    
    print("\n[INFO] Phase 2: L-BFGS optimization")
    time_callback_lbfgs = TimeBasedEarlyStopping(max_train_time)
    time_callback_lbfgs.start_time = t_start_training  # base it on total elapsed time overall

    # Give L-BFGS more of the budget, but keep each PyTorch step short enough for callbacks.
    configure_pytorch_lbfgs(lbfgs_total_iters, lbfgs_inner_iters)
    model.compile("L-BFGS", loss_weights=loss_weights)
    losshistory, train_state = model.train(
        iterations=10000,
        callbacks=[time_callback_lbfgs],
        display_every=10,
    )
    
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
