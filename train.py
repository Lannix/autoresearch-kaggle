"""
Autoresearch PINN training script for LLE using DeepXDE.
Contains internal time-management to guarantee final evaluation before Kaggle timeout.
"""
import os
os.environ["DDE_BACKEND"] = "pytorch"

import time
import traceback
import sys
import math
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

theta_samples = np.asarray(th0_arr, dtype=np.float64).reshape(-1)
u0_samples = np.asarray(u0, dtype=np.float64).reshape(-1)
v0_samples = np.asarray(v0, dtype=np.float64).reshape(-1)
theta_step = float(np.median(np.diff(theta_samples)))
theta_period = float(theta_step * theta_samples.size)
theta_origin = float(theta_samples[0])


def build_real_fourier_coeffs(values):
    coeffs = np.fft.rfft(values) / values.size
    cos_coeffs = 2.0 * coeffs.real
    sin_coeffs = -2.0 * coeffs.imag
    cos_coeffs[0] = coeffs[0].real
    sin_coeffs[0] = 0.0
    if values.size % 2 == 0:
        cos_coeffs[-1] = coeffs[-1].real
        sin_coeffs[-1] = 0.0
    return cos_coeffs.astype(np.float32), sin_coeffs.astype(np.float32)


u0_cos_coeffs, u0_sin_coeffs = build_real_fourier_coeffs(u0_samples)
v0_cos_coeffs, v0_sin_coeffs = build_real_fourier_coeffs(v0_samples)
fourier_modes = np.arange(u0_cos_coeffs.size, dtype=np.float32)

# ==========================================
# 2. DeepXDE Geometry and Domain
# ==========================================
geom = dde.geometry.Interval(th_min, th_max)
timedomain = dde.geometry.TimeDomain(t_min, t_max)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

HYBRID_RAR_BASE_POINTS = 20000
HYBRID_RAR_ADAPTIVE_POINTS = 10000
HYBRID_RAR_PERIOD = 2500
HYBRID_RAR_CANDIDATES = 20000
HYBRID_RAR_REFRESH_COUNT = 1000
HYBRID_RAR_MAX_UPDATES = 3
HYBRID_RAR_SCORE_BATCH = 2048


def sample_interior_points(count, random):
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    return geomtime.random_points(count, random=random).astype(np.float32)


base_collocation_points = sample_interior_points(HYBRID_RAR_BASE_POINTS, random="Hammersley")
adaptive_collocation_points = sample_interior_points(HYBRID_RAR_ADAPTIVE_POINTS, random="pseudo")
initial_collocation_points = np.vstack(
    (base_collocation_points, adaptive_collocation_points)
).astype(np.float32)

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

data = dde.data.TimePDE(
    geomtime,
    pde,
    [],
    num_domain=0,
    num_boundary=0,
    num_initial=0,
    anchors=initial_collocation_points,
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

ic_fourier_cache = {}


def get_ic_fourier_tensors(device, dtype):
    key = (device.type, device.index, str(dtype))
    if key not in ic_fourier_cache:
        ic_fourier_cache[key] = {
            "modes": torch.as_tensor(fourier_modes, device=device, dtype=dtype).view(1, -1),
            "u_cos": torch.as_tensor(u0_cos_coeffs, device=device, dtype=dtype).view(-1, 1),
            "u_sin": torch.as_tensor(u0_sin_coeffs, device=device, dtype=dtype).view(-1, 1),
            "v_cos": torch.as_tensor(v0_cos_coeffs, device=device, dtype=dtype).view(-1, 1),
            "v_sin": torch.as_tensor(v0_sin_coeffs, device=device, dtype=dtype).view(-1, 1),
            "theta_origin": torch.tensor(theta_origin, device=device, dtype=dtype),
            "theta_period": torch.tensor(theta_period, device=device, dtype=dtype),
            "two_pi": torch.tensor(2.0 * math.pi, device=device, dtype=dtype),
            "t0": torch.tensor(t0, device=device, dtype=dtype),
        }
    return ic_fourier_cache[key]


def reconstruct_fourier_signal(theta, cos_coeffs, sin_coeffs, coeffs):
    theta_rel = theta - coeffs["theta_origin"]
    angles = coeffs["two_pi"] * theta_rel * coeffs["modes"] / coeffs["theta_period"]
    return torch.cos(angles) @ cos_coeffs + torch.sin(angles) @ sin_coeffs


def output_transform(x, y):
    theta = x[:, 0:1]
    time_coord = x[:, 1:2]
    coeffs = get_ic_fourier_tensors(x.device, x.dtype)
    u_exact = reconstruct_fourier_signal(theta, coeffs["u_cos"], coeffs["u_sin"], coeffs)
    v_exact = reconstruct_fourier_signal(theta, coeffs["v_cos"], coeffs["v_sin"], coeffs)
    growth = 1.0 - torch.exp(-5.0 * torch.clamp(time_coord - coeffs["t0"], min=0.0))
    return torch.cat((u_exact, v_exact), dim=1) + growth * y

net.apply_feature_transform(feature_transform)
net.apply_output_transform(output_transform)
model = dde.Model(data, net)

# ==========================================
# 6. Training Setup
# ==========================================
EVAL_RESERVE = 45  
max_train_time = TIME_BUDGET - EVAL_RESERVE
adam_time_limit = max_train_time * 0.60
lbfgs_total_iters = 5000
lbfgs_inner_iters = 250

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

def model_uv(t_in, th_in, need_x=False):
    # DeepXDE inputs are [theta, t]
    x = torch.cat((th_in, t_in), dim=1)
    if need_x:
        x = x.requires_grad_(True)
    else:
        x = x.detach()
    uv = net(x)
    return uv[:, 0:1], uv[:, 1:2], x


def causal_residual_scores(points, batch_size):
    was_training = net.training
    net.train(mode=False)
    scores = []

    with torch.enable_grad():
        for start in range(0, points.shape[0], batch_size):
            batch = points[start : start + batch_size]
            x = torch.as_tensor(batch, device=device, dtype=torch.float32)
            x.requires_grad_(True)

            uv = net(x)
            u, v = uv[:, 0:1], uv[:, 1:2]

            grad_u = torch.autograd.grad(
                u,
                x,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True,
            )[0]
            grad_v = torch.autograd.grad(
                v,
                x,
                grad_outputs=torch.ones_like(v),
                create_graph=True,
                retain_graph=True,
            )[0]

            du_dt = grad_u[:, 1:2]
            dv_dt = grad_v[:, 1:2]
            du_dth = grad_u[:, 0:1]
            dv_dth = grad_v[:, 0:1]

            du_dth2 = torch.autograd.grad(
                du_dth,
                x,
                grad_outputs=torch.ones_like(du_dth),
                create_graph=False,
                retain_graph=True,
            )[0][:, 0:1]
            dv_dth2 = torch.autograd.grad(
                dv_dth,
                x,
                grad_outputs=torch.ones_like(dv_dth),
                create_graph=False,
                retain_graph=False,
            )[0][:, 0:1]

            intensity = u.square() + v.square()
            res_u = du_dt - (-u + zeta * v - 0.5 * dv_dth2 - intensity * v + f)
            res_v = dv_dt - (-v - zeta * u + 0.5 * du_dth2 + intensity * u)
            time_frac = (x[:, 1:2] - t_min) / (t_max - t_min + 1e-12)
            causal_weight = torch.exp(-2.0 * time_frac)
            residual_sq = (causal_weight * res_u).square() + (causal_weight * res_v).square()
            scores.append(residual_sq.detach().cpu().numpy().reshape(-1))

            del x, uv, u, v, grad_u, grad_v, du_dt, dv_dt, du_dth, dv_dth, du_dth2, dv_dth2
            del intensity, res_u, res_v, time_frac, causal_weight, residual_sq

    net.train(mode=was_training)
    return np.concatenate(scores, axis=0)


def select_top_points(points, scores, count):
    if count <= 0:
        return np.empty((0, points.shape[1]), dtype=points.dtype), np.empty((0,), dtype=scores.dtype)
    if count >= points.shape[0]:
        order = np.argsort(scores)[::-1]
        return points[order], scores[order]
    idx = np.argpartition(scores, -count)[-count:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return points[idx], scores[idx]


class HybridRARCallback(dde.callbacks.Callback):
    def __init__(
        self,
        base_points,
        adaptive_points,
        period,
        candidate_count,
        refresh_count,
        max_updates,
        score_batch,
    ):
        super().__init__()
        self.base_points = np.asarray(base_points, dtype=np.float32)
        self.adaptive_points = np.asarray(adaptive_points, dtype=np.float32)
        self.period = period
        self.candidate_count = candidate_count
        self.refresh_count = refresh_count
        self.max_updates = max_updates
        self.score_batch = score_batch
        self.updates_completed = 0
        self.epochs_since_refresh = 0

    def on_train_begin(self):
        print(
            "[INFO] Hybrid RAR enabled with "
            f"{self.base_points.shape[0]} static base points and "
            f"{self.adaptive_points.shape[0]} adaptive points."
        )

    def on_epoch_end(self):
        if self.updates_completed >= self.max_updates:
            return

        self.epochs_since_refresh += 1
        if self.epochs_since_refresh < self.period:
            return
        self.epochs_since_refresh = 0

        retain_count = max(0, self.adaptive_points.shape[0] - self.refresh_count)
        adaptive_scores = causal_residual_scores(self.adaptive_points, self.score_batch)
        retained_points, retained_scores = select_top_points(
            self.adaptive_points, adaptive_scores, retain_count
        )

        candidate_points = sample_interior_points(self.candidate_count, random="pseudo")
        candidate_scores = causal_residual_scores(candidate_points, self.score_batch)
        injected_points, injected_scores = select_top_points(
            candidate_points, candidate_scores, self.refresh_count
        )

        self.adaptive_points = np.vstack((retained_points, injected_points)).astype(np.float32)
        refreshed_points = np.vstack((self.base_points, self.adaptive_points)).astype(np.float32)
        self.model.data.replace_with_anchors(refreshed_points)
        self.model.data.test_x = None
        self.model.data.test_y = None
        self.model.data.test_aux_vars = None
        self.model.train_state.set_data_train(
            self.model.data.train_x,
            self.model.data.train_y,
            self.model.data.train_aux_vars,
        )
        self.model.train_state.set_data_test(*self.model.data.test())

        self.updates_completed += 1
        print(
            "[INFO] Hybrid RAR refresh "
            f"{self.updates_completed}/{self.max_updates}: "
            f"retained mean residual {retained_scores.mean():.3e}, "
            f"injected mean residual {injected_scores.mean():.3e}, "
            f"candidate mean residual {candidate_scores.mean():.3e}."
        )

# Loss weights order corresponds to the PDE residual outputs.
loss_weights =[3.0, 3.0]
model.compile("adam", lr=1e-3, loss_weights=loss_weights)

time_callback_adam = TimeBasedEarlyStopping(adam_time_limit)
hybrid_rar_callback = HybridRARCallback(
    base_points=base_collocation_points,
    adaptive_points=adaptive_collocation_points,
    period=HYBRID_RAR_PERIOD,
    candidate_count=HYBRID_RAR_CANDIDATES,
    refresh_count=HYBRID_RAR_REFRESH_COUNT,
    max_updates=HYBRID_RAR_MAX_UPDATES,
    score_batch=HYBRID_RAR_SCORE_BATCH,
)

try:
    print("\n[INFO] Phase 1: Adam optimization")
    losshistory, train_state = model.train(
        iterations=100000,
        callbacks=[time_callback_adam, hybrid_rar_callback],
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
