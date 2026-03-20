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

# ==========================================
# 3. Neural Network Architecture
# ==========================================
ic_fourier_cache = {}
power_stabilization_cache = {}
power_stabilization_theta_count = 512
power_stabilization_time_count = 6
power_stabilization_start_frac = 0.80


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


def build_power_stabilization_grid(theta_count, time_count):
    theta_points = np.linspace(
        th_min,
        th_max,
        theta_count,
        endpoint=False,
        dtype=np.float32,
    ).reshape(-1, 1)
    time_points = np.linspace(
        t_min + power_stabilization_start_frac * (t_max - t_min),
        t_max,
        time_count,
        dtype=np.float32,
    ).reshape(-1, 1)
    theta_grid = np.tile(theta_points[None, :, :], (time_count, 1, 1))
    time_grid = np.tile(time_points[:, None, :], (1, theta_count, 1))
    points = np.concatenate((theta_grid, time_grid), axis=2).reshape(-1, 2)
    pair_times = time_points[1:]
    pair_weights = ((pair_times - t_min) / (t_max - t_min + 1e-12)) ** 4.0
    return points.astype(np.float32), pair_weights.astype(np.float32)


def get_power_stabilization_tensors(device, dtype):
    key = (device.type, device.index, str(dtype))
    if key not in power_stabilization_cache:
        points, pair_weights = build_power_stabilization_grid(
            power_stabilization_theta_count,
            power_stabilization_time_count,
        )
        power_stabilization_cache[key] = {
            "points": torch.as_tensor(points, device=device, dtype=dtype),
            "pair_weights": torch.as_tensor(pair_weights, device=device, dtype=dtype),
            "delta_t": torch.tensor(
                (1.0 - power_stabilization_start_frac) * (t_max - t_min) / max(1, power_stabilization_time_count - 1),
                device=device,
                dtype=dtype,
            ),
        }
    return power_stabilization_cache[key]

def reconstruct_fourier_signal(theta, cos_coeffs, sin_coeffs, coeffs):
    theta_rel = theta - coeffs["theta_origin"]
    angles = coeffs["two_pi"] * theta_rel * coeffs["modes"] / coeffs["theta_period"]
    return torch.cos(angles) @ cos_coeffs + torch.sin(angles) @ sin_coeffs


def wrap_theta_to_domain(theta):
    domain_width = th_max - th_min
    return ((theta - th_min) % domain_width) + th_min


def build_gaussian_biased_collocation_points(num_points):
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
    print(
        "[INFO] Static Gaussian-biased collocation: "
        f"{gaussian_count} Gaussian + {uniform_count} uniform theta samples, "
        f"theta_peak={theta_peak:.4f}, sigma={gaussian_collocation_sigma:.4f}, "
        f"time_beta=({time_bias_beta_a:.1f}, {time_bias_beta_b:.1f})"
    )
    return collocation_points


class MultiScaleFourierCore(torch.nn.Module):
    def __init__(
        self,
        input_dim=2,
        hidden_dim=128,
        num_hidden_layers=5,
        output_dim=2,
        sigmas=(1.0, 10.0),
        features_per_scale=16,
    ):
        super().__init__()
        self.sigmas = tuple(float(sigma) for sigma in sigmas)
        self.features_per_scale = int(features_per_scale)

        for idx, sigma in enumerate(self.sigmas):
            projection = torch.randn(self.features_per_scale, input_dim) * sigma
            self.register_buffer(f"projection_{idx}", projection)

        encoded_dim = input_dim + 2 * self.features_per_scale * len(self.sigmas)
        layer_dims = [encoded_dim] + [hidden_dim] * num_hidden_layers + [output_dim]
        layers = []
        for in_dim, out_dim in zip(layer_dims[:-2], layer_dims[1:-1]):
            layers.append(torch.nn.Linear(in_dim, out_dim))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(layer_dims[-2], layer_dims[-1]))
        self.network = torch.nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.network:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def encode(self, x):
        encoded = [x]
        for idx in range(len(self.sigmas)):
            projection = getattr(self, f"projection_{idx}")
            angles = 2.0 * math.pi * (x @ projection.T)
            encoded.append(torch.sin(angles))
            encoded.append(torch.cos(angles))
        return torch.cat(encoded, dim=1)

    def forward(self, x):
        return self.network(self.encode(x))


class ComplexLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.complex64)
        )
        self.bias = torch.nn.Parameter(
            torch.empty(out_features, dtype=torch.complex64)
        )
        self.reset_parameters()

    def reset_parameters(self):
        weight_real = torch.empty(self.weight.shape, dtype=torch.float32)
        weight_imag = torch.empty(self.weight.shape, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(weight_real)
        torch.nn.init.xavier_uniform_(weight_imag)
        with torch.no_grad():
            self.weight.copy_(torch.complex(weight_real, weight_imag) * 0.5)
            self.bias.zero_()

    def forward(self, x):
        return x @ self.weight.T + self.bias


class ComplexMultiScaleFourierCore(torch.nn.Module):
    def __init__(
        self,
        input_dim=2,
        hidden_dim=96,
        num_hidden_layers=4,
        sigmas=(1.0, 10.0),
        features_per_scale=16,
    ):
        super().__init__()
        self.sigmas = tuple(float(sigma) for sigma in sigmas)
        self.features_per_scale = int(features_per_scale)

        for idx, sigma in enumerate(self.sigmas):
            projection = torch.randn(self.features_per_scale, input_dim) * sigma
            self.register_buffer(f"projection_{idx}", projection)

        encoded_dim = input_dim + 2 * self.features_per_scale * len(self.sigmas)
        self.hidden_layers = torch.nn.ModuleList()
        in_dim = encoded_dim
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(ComplexLinear(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.output_layer = ComplexLinear(hidden_dim, 1)

    def encode(self, x):
        encoded = [x]
        for idx in range(len(self.sigmas)):
            projection = getattr(self, f"projection_{idx}")
            angles = 2.0 * math.pi * (x @ projection.T)
            encoded.append(torch.sin(angles))
            encoded.append(torch.cos(angles))
        encoded_real = torch.cat(encoded, dim=1)
        return torch.complex(encoded_real, torch.zeros_like(encoded_real))

    def forward(self, x):
        z = self.encode(x)
        for layer in self.hidden_layers:
            z = torch.tanh(layer(z))
        return self.output_layer(z)


class NormalizedChainRuleNet(dde.nn.NN):
    def __init__(self):
        super().__init__()
        self.core = ComplexMultiScaleFourierCore(
            sigmas=(1.0, 10.0),
            features_per_scale=16,
        )
        self.regularizer = None
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
        raw_complex = self.core(x_norm)
        theta, time_coord = self.denormalize_inputs(x_norm)
        coeffs = get_ic_fourier_tensors(x_norm.device, x_norm.dtype)
        u_exact = reconstruct_fourier_signal(theta, coeffs["u_cos"], coeffs["u_sin"], coeffs)
        v_exact = reconstruct_fourier_signal(theta, coeffs["v_cos"], coeffs["v_sin"], coeffs)
        growth = 1.0 - torch.exp(-5.0 * torch.clamp(time_coord - coeffs["t0"], min=0.0))
        psi_exact = torch.complex(u_exact, v_exact)
        psi = psi_exact + growth * raw_complex
        return torch.cat((psi.real, psi.imag), dim=1)

    def forward(self, inputs):
        self.last_x_norm = self.normalize_inputs(inputs)
        return self.forward_from_normalized(self.last_x_norm)


net = NormalizedChainRuleNet()
custom_collocation_points = build_gaussian_biased_collocation_points(num_domain_points)
print(
    "[INFO] CVPINN core: "
    "complex_weights=True, hidden_dim=96, num_hidden_layers=4, "
    "sigmas=(1.0, 10.0), features_per_scale=16"
)
print(
    "[INFO] Global power prior: "
    f"theta_points={power_stabilization_theta_count}, "
    f"time_points={power_stabilization_time_count}, "
    f"late_start_frac={power_stabilization_start_frac:.2f}"
)


def global_power_stabilization_loss(device, dtype):
    tensors = get_power_stabilization_tensors(device, dtype)
    uv = net.forward_from_normalized(net.normalize_inputs(tensors["points"]))
    intensity = uv[:, 0:1].square() + uv[:, 1:2].square()
    power_by_time = intensity.view(
        power_stabilization_time_count,
        power_stabilization_theta_count,
        1,
    ).mean(dim=1) * theta_period
    power_dt = (power_by_time[1:] - power_by_time[:-1]) / (tensors["delta_t"] + 1e-12)
    return tensors["pair_weights"] * power_dt

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

    complex_dtype = torch.complex64 if x.dtype == torch.float32 else torch.complex128
    psi = torch.complex(u, v)
    psi_dt = torch.complex(du_dt, dv_dt)
    psi_dth2 = torch.complex(du_dth2, dv_dth2)
    linear_coeff = torch.tensor(complex(-1.0, -float(zeta)), device=x.device, dtype=complex_dtype)
    nonlinear_coeff = torch.tensor(1j, device=x.device, dtype=complex_dtype)
    dispersion_coeff = torch.tensor(0.5j, device=x.device, dtype=complex_dtype)
    drive = torch.tensor(float(f), device=x.device, dtype=complex_dtype)
    res_complex = psi_dt - (
        linear_coeff * psi
        + dispersion_coeff * psi_dth2
        + nonlinear_coeff * psi.abs().square() * psi
        + drive
    )
    res_u = res_complex.real
    res_v = res_complex.imag
    time_frac = (x[:, 1:2] - t_min) / (t_max - t_min + 1e-12)
    causal_weight = torch.exp(-2.0 * time_frac)
    power_stabilization = global_power_stabilization_loss(x.device, x.dtype)

    return [causal_weight * res_u, causal_weight * res_v, power_stabilization]


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

def model_uv(t_in, th_in, need_x=False):
    # DeepXDE inputs are [theta, t]
    x = torch.cat((th_in, t_in), dim=1)
    if need_x:
        x = x.requires_grad_(True)
    else:
        x = x.detach()
    uv = net(x)
    return uv[:, 0:1], uv[:, 1:2], x

# Loss weights order corresponds to [pde_u, pde_v, global_power_dt].
loss_weights =[3.0, 3.0, 0.5]
model.compile("adam", lr=1e-3, loss_weights=loss_weights)

time_callback_adam = TimeBasedEarlyStopping(adam_time_limit)

try:
    print("\n[INFO] Phase 1: Adam optimization")
    losshistory, train_state = model.train(iterations=100000, callbacks=[time_callback_adam], display_every=1000)
    
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
