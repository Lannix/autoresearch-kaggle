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
theta_peak = float(theta_samples[int(np.argmax(u0_samples**2 + v0_samples**2))])


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

theta_center = float((th_min + th_max) * 0.5)
theta_half_span = float((th_max - th_min) * 0.5 + 1e-12)
time_center = float((t_min + t_max) * 0.5)
time_half_span = float((t_max - t_min) * 0.5 + 1e-12)
theta_norm_scale = float(1.0 / theta_half_span)
time_norm_scale = float(1.0 / time_half_span)
num_domain_points = 30000
gaussian_collocation_fraction = 0.80
gaussian_collocation_sigma = 0.15 * (th_max - th_min)
time_bias_beta_a = 1.0
time_bias_beta_b = 3.0
anchor_batch_count = 10
fixed_anchor_fraction = 0.20
total_anchor_points = num_domain_points * anchor_batch_count
fixed_anchor_points = int(round(total_anchor_points * fixed_anchor_fraction))
fixed_anchor_points -= fixed_anchor_points % anchor_batch_count
dynamic_anchor_points = total_anchor_points - fixed_anchor_points
fixed_batch_size = fixed_anchor_points // anchor_batch_count
dynamic_batch_size = dynamic_anchor_points // anchor_batch_count
rar_period = 600
rar_score_batch_size = 2048
rar_candidate_points = dynamic_batch_size

# ==========================================
# 3. Neural Network Architecture
# ==========================================
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


def wrap_theta_to_domain(theta):
    domain_width = th_max - th_min
    return ((theta - th_min) % domain_width) + th_min


def build_gaussian_biased_collocation_points(num_points, announce=True, label="Static Gaussian-biased collocation"):
    gaussian_count = int(round(num_points * gaussian_collocation_fraction))
    uniform_count = num_points - gaussian_count

    theta_gaussian = np.random.normal(
        loc=theta_peak,
        scale=gaussian_collocation_sigma,
        size=(gaussian_count, 1),
    ).astype(np.float32)
    theta_gaussian = wrap_theta_to_domain(theta_gaussian).astype(np.float32)

    theta_uniform = np.random.uniform(
        th_min,
        th_max,
        size=(uniform_count, 1),
    ).astype(np.float32)
    theta_samples_biased = np.vstack((theta_gaussian, theta_uniform)).astype(np.float32)
    np.random.shuffle(theta_samples_biased)

    time_samples_biased = np.random.beta(
        time_bias_beta_a,
        time_bias_beta_b,
        size=(num_points, 1),
    ).astype(np.float32)
    time_samples_biased = (
        t_min + (t_max - t_min) * time_samples_biased
    ).astype(np.float32)
    np.random.shuffle(time_samples_biased)

    collocation_points = np.hstack((theta_samples_biased, time_samples_biased)).astype(np.float32)
    if announce:
        print(
            f"[INFO] {label}: "
            f"{gaussian_count} Gaussian + {uniform_count} uniform theta samples, "
            f"theta_peak={theta_peak:.4f}, sigma={gaussian_collocation_sigma:.4f}, "
            f"time_beta=({time_bias_beta_a:.1f}, {time_bias_beta_b:.1f})"
        )
    return collocation_points


def split_anchor_batches(points, batch_size):
    return [
        np.array(chunk, dtype=np.float32, copy=True)
        for chunk in np.split(points, points.shape[0] // batch_size)
    ]


def assemble_anchor_batch(batch_index, fixed_batches, dynamic_batches):
    batch_points = np.vstack((fixed_batches[batch_index], dynamic_batches[batch_index])).astype(np.float32)
    np.random.shuffle(batch_points)
    return batch_points


class NormalizedChainRuleNet(dde.nn.NN):
    def __init__(self):
        super().__init__()
        self.core = dde.nn.FNN([2] + [128] * 5 + [2], "tanh", "Glorot uniform")
        self.regularizer = self.core.regularizer
        self.last_x_norm = None

    def normalize_inputs(self, x):
        theta = x[:, 0:1]
        time_coord = x[:, 1:2]
        theta_norm = (theta - theta_center) * theta_norm_scale
        time_norm = (time_coord - time_center) * time_norm_scale
        return torch.cat((theta_norm, time_norm), dim=1)

    def denormalize_inputs(self, x_norm):
        theta = x_norm[:, 0:1] / theta_norm_scale + theta_center
        time_coord = x_norm[:, 1:2] / time_norm_scale + time_center
        return theta, time_coord

    def forward_from_normalized(self, x_norm):
        raw = self.core(x_norm)
        theta, time_coord = self.denormalize_inputs(x_norm)
        coeffs = get_ic_fourier_tensors(x_norm.device, x_norm.dtype)
        u_exact = reconstruct_fourier_signal(theta, coeffs["u_cos"], coeffs["u_sin"], coeffs)
        v_exact = reconstruct_fourier_signal(theta, coeffs["v_cos"], coeffs["v_sin"], coeffs)
        growth = 1.0 - torch.exp(-5.0 * torch.clamp(time_coord - coeffs["t0"], min=0.0))
        return torch.cat((u_exact, v_exact), dim=1) + growth * raw

    def forward(self, inputs):
        self.last_x_norm = self.normalize_inputs(inputs)
        return self.forward_from_normalized(self.last_x_norm)


net = NormalizedChainRuleNet()
fixed_anchor_pool = build_gaussian_biased_collocation_points(
    fixed_anchor_points,
    announce=False,
)
dynamic_anchor_pool = build_gaussian_biased_collocation_points(
    dynamic_anchor_points,
    announce=False,
)
fixed_anchor_batches = split_anchor_batches(fixed_anchor_pool, fixed_batch_size)
dynamic_anchor_batches = split_anchor_batches(dynamic_anchor_pool, dynamic_batch_size)
custom_collocation_points = assemble_anchor_batch(0, fixed_anchor_batches, dynamic_anchor_batches)
print(
    "[INFO] Batched RAR collocation pool: "
    f"total_points={total_anchor_points}, batches={anchor_batch_count}, "
    f"active_batch={num_domain_points}, fixed={fixed_anchor_points} ({fixed_batch_size}/batch), "
    f"dynamic={dynamic_anchor_points} ({dynamic_batch_size}/batch), "
    f"theta_peak={theta_peak:.4f}, sigma={gaussian_collocation_sigma:.4f}, "
    f"time_beta=({time_bias_beta_a:.1f}, {time_bias_beta_b:.1f}), "
    f"rar_period={rar_period}, candidate_points={rar_candidate_points}"
)

# ==========================================
# 4. Physics / LLE Residual
# ==========================================
def pde(x, y):
    x_norm = net.last_x_norm
    if x_norm is None or x_norm.shape[0] != x.shape[0]:
        x_norm = net.normalize_inputs(x)
        y = net.forward_from_normalized(x_norm)

    u, v = y[:, 0:1], y[:, 1:2]

    grad_u_norm = torch.autograd.grad(
        u,
        x_norm,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_v_norm = torch.autograd.grad(
        v,
        x_norm,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        retain_graph=True,
    )[0]

    du_dth2_norm = torch.autograd.grad(
        grad_u_norm[:, 0:1],
        x_norm,
        grad_outputs=torch.ones_like(grad_u_norm[:, 0:1]),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    dv_dth2_norm = torch.autograd.grad(
        grad_v_norm[:, 0:1],
        x_norm,
        grad_outputs=torch.ones_like(grad_v_norm[:, 0:1]),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]

    du_dt = grad_u_norm[:, 1:2] * time_norm_scale
    dv_dt = grad_v_norm[:, 1:2] * time_norm_scale
    du_dth2 = du_dth2_norm * (theta_norm_scale ** 2)
    dv_dth2 = dv_dth2_norm * (theta_norm_scale ** 2)

    intensity = u.square() + v.square()
    res_u = du_dt - (-u + zeta * v - 0.5 * dv_dth2 - intensity * v + f)
    res_v = dv_dt - (-v - zeta * u + 0.5 * du_dth2 + intensity * u)
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
    anchors=custom_collocation_points,
)
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


def residual_scores_for_points(points_np, batch_size):
    original_requires_grad = [param.requires_grad for param in net.parameters()]
    for param in net.parameters():
        param.requires_grad_(False)

    scores = []
    try:
        for start in range(0, points_np.shape[0], batch_size):
            stop = start + batch_size
            x_batch = torch.tensor(
                points_np[start:stop],
                device=device,
                dtype=torch.float32,
            ).requires_grad_(True)
            residual_terms = pde(x_batch, net(x_batch))
            score = torch.zeros(x_batch.shape[0], device=device, dtype=torch.float32)
            for term in residual_terms[:2]:
                score = score + term[:, 0].square()
            scores.append(torch.sqrt(score + 1e-12).detach().cpu().numpy())
            del x_batch, residual_terms, score
        return np.concatenate(scores, axis=0)
    finally:
        for param, requires_grad in zip(net.parameters(), original_requires_grad):
            param.requires_grad_(requires_grad)


class BatchedRARRotator(dde.callbacks.Callback):
    def __init__(
        self,
        period,
        fixed_batches,
        dynamic_batches,
        candidate_points,
        score_batch_size,
    ):
        super().__init__()
        self.period = period
        self.fixed_batches = fixed_batches
        self.dynamic_batches = dynamic_batches
        self.candidate_points = candidate_points
        self.score_batch_size = score_batch_size
        self.active_batch_index = 0

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step == 0 or step % self.period != 0:
            return

        current_dynamic = np.asarray(
            self.dynamic_batches[self.active_batch_index],
            dtype=np.float32,
        )
        candidate_points = build_gaussian_biased_collocation_points(
            self.candidate_points,
            announce=False,
        )

        current_scores = residual_scores_for_points(current_dynamic, self.score_batch_size)
        candidate_scores = residual_scores_for_points(candidate_points, self.score_batch_size)

        union_points = np.vstack((current_dynamic, candidate_points)).astype(np.float32)
        union_scores = np.concatenate((current_scores, candidate_scores), axis=0)
        top_indices = np.argpartition(union_scores, -dynamic_batch_size)[-dynamic_batch_size:]
        updated_dynamic = union_points[top_indices].astype(np.float32)
        np.random.shuffle(updated_dynamic)
        self.dynamic_batches[self.active_batch_index] = updated_dynamic

        updated_mean = float(union_scores[top_indices].mean())
        current_mean = float(current_scores.mean())
        candidate_mean = float(candidate_scores.mean())

        self.active_batch_index = (self.active_batch_index + 1) % len(self.dynamic_batches)
        next_batch = assemble_anchor_batch(
            self.active_batch_index,
            self.fixed_batches,
            self.dynamic_batches,
        )

        self.model.data.replace_with_anchors(next_batch)
        self.model.data.test_x = None
        self.model.data.test_y = None
        self.model.data.test_aux_vars = None
        self.model.train_state.set_data_train(
            self.model.data.train_x,
            self.model.data.train_y,
            self.model.data.train_aux_vars,
        )
        self.model.train_state.set_data_test(*self.model.data.test())

        print(
            f"[INFO] Batched RAR update at step {step}: "
            f"updated_batch={self.active_batch_index - 1 if self.active_batch_index > 0 else len(self.dynamic_batches) - 1}, "
            f"current_mean={current_mean:.3e}, candidate_mean={candidate_mean:.3e}, "
            f"selected_mean={updated_mean:.3e}, next_batch={self.active_batch_index}"
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

# Loss weights order corresponds to the PDE residual outputs.
loss_weights =[3.0, 3.0]
model.compile("adam", lr=1e-3, loss_weights=loss_weights)

time_callback_adam = TimeBasedEarlyStopping(adam_time_limit)
rar_callback = BatchedRARRotator(
    rar_period,
    fixed_anchor_batches,
    dynamic_anchor_batches,
    rar_candidate_points,
    rar_score_batch_size,
)

try:
    print("\n[INFO] Phase 1: Adam optimization")
    losshistory, train_state = model.train(
        iterations=100000,
        callbacks=[time_callback_adam, rar_callback],
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
