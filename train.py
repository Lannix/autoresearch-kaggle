"""
Autoresearch PINN training script for LLE using DeepXDE.
Contains internal time-management to guarantee final evaluation before Kaggle timeout.[DeepXDE Note for Beginners]:
Physics-Informed Neural Networks (PINNs) don't just learn from data; they learn to satisfy 
a governing differential equation (PDE). This script solves the 1D Lugiato-Lefever Equation (LLE).
Instead of using standard DeepXDE syntax for Initial Conditions (IC) and Boundary Conditions (BC), 
this script uses "Hard Constraints". The network architecture itself mathematically guarantees 
the ICs and periodic BCs are satisfied, leaving the optimizer to focus 100% on the PDE residual.
"""
import os
# Force DeepXDE to use the PyTorch backend
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
# Use float32 for speed on Kaggle T4 GPUs.
dde.config.set_default_float("float32")
dde.config.set_random_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")

t_start_training = time.time()

# Load isolated training setup from prepare.py. 
# This ensures we don't accidentally leak the interior ground truth dataset during training.
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

# Extract grid info from initial conditions
theta_samples = np.asarray(th0_arr, dtype=np.float64).reshape(-1)
u0_samples = np.asarray(u0, dtype=np.float64).reshape(-1)
v0_samples = np.asarray(v0, dtype=np.float64).reshape(-1)
theta_step = float(np.median(np.diff(theta_samples)))
theta_period = float(theta_step * theta_samples.size)
theta_origin = float(theta_samples[0])
theta_peak = float(theta_samples[int(np.argmax(u0_samples**2 + v0_samples**2))])


def build_real_fourier_coeffs(values):
    """
    Computes Fourier coefficients of the Initial Conditions.
    This is used later to construct an analytical function that exactly reconstructs 
    the t=0 state, acting as a "Hard Constraint" for the PINN.
    """
    coeffs = np.fft.rfft(values) / values.size
    cos_coeffs = 2.0 * coeffs.real
    sin_coeffs = -2.0 * coeffs.imag
    cos_coeffs[0] = coeffs[0].real
    sin_coeffs[0] = 0.0
    if values.size % 2 == 0:
        cos_coeffs[-1] = coeffs[-1].real
        sin_coeffs[-1] = 0.0
    return cos_coeffs.astype(np.float32), sin_coeffs.astype(np.float32)

# Pre-calculate Fourier modes for exact IC reconstruction
u0_cos_coeffs, u0_sin_coeffs = build_real_fourier_coeffs(u0_samples)
v0_cos_coeffs, v0_sin_coeffs = build_real_fourier_coeffs(v0_samples)
fourier_modes = np.arange(u0_cos_coeffs.size, dtype=np.float32)

# ==========================================
# 2. DeepXDE Geometry and Domain
# ==========================================
# [DeepXDE Note for Beginners]: We define the spatial (Interval) and temporal (TimeDomain) domains, 
# then multiply them to get the spatio-temporal domain (GeometryXTime).
geom = dde.geometry.Interval(th_min, th_max)
timedomain = dde.geometry.TimeDomain(t_min, t_max)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Normalization variables (Neural Networks learn best when inputs are roughly in [-1, 1])
theta_center = float((th_min + th_max) * 0.5)
theta_half_span = float((th_max - th_min) * 0.5 + 1e-12)
time_center = float((t_min + t_max) * 0.5)
time_half_span = float((t_max - t_min) * 0.5 + 1e-12)
theta_norm_scale = float(1.0 / theta_half_span)
time_norm_scale = float(1.0 / time_half_span)

# Hyperparameters discovered by the autonomous swarm
num_domain_points = 30000
gaussian_collocation_fraction = 0.80
gaussian_collocation_sigma = 0.15 * (th_max - th_min)
time_bias_beta_a = 1.0
time_bias_beta_b = 3.0
msffn_sigmas = (1.0, 10.0)
msffn_features_per_scale = 16
breather_theta_harmonics = (1, 2, 3, 4, 5)
breather_period_guess = 0.9990
breather_time_harmonics = (1.0, 2.0)
curriculum_stage_upper_fracs = (0.25, 0.50, 0.75, 1.00)
curriculum_stage_loss_thresholds = (0.25, 0.10, 0.045)
curriculum_min_stage_steps = 1000
r3_period = 5000
r3_retain_fraction = 0.20
r3_score_batch_size = 1024

# ==========================================
# 3. Neural Network Architecture
# ==========================================
ic_fourier_cache = {}
power_stabilization_cache = {}
power_stabilization_theta_count = 512
power_stabilization_time_count = 6
power_stabilization_start_frac = 0.80
curriculum_time_upper = float(t_max)


def get_ic_fourier_tensors(device, dtype):
    """Caches fourier tensors on the correct PyTorch device to avoid host-to-device bottlenecks."""
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


def build_power_stabilization_grid(theta_count, time_count, time_upper=None):
    """Creates a regular grid to calculate the physical 'power conservation' prior."""
    time_upper = float(t_max if time_upper is None else time_upper)
    theta_points = np.linspace(
        th_min, th_max, theta_count, endpoint=False, dtype=np.float32
    ).reshape(-1, 1)
    time_points = np.linspace(
        t_min + power_stabilization_start_frac * (time_upper - t_min),
        time_upper, time_count, dtype=np.float32
    ).reshape(-1, 1)
    
    theta_grid = np.tile(theta_points[None, :, :], (time_count, 1, 1))
    time_grid = np.tile(time_points[:, None, :], (1, theta_count, 1))
    points = np.concatenate((theta_grid, time_grid), axis=2).reshape(-1, 2)
    
    pair_times = time_points[1:]
    pair_weights = ((pair_times - t_min) / (t_max - t_min + 1e-12)) ** 4.0
    return points.astype(np.float32), pair_weights.astype(np.float32)


def get_power_stabilization_tensors(device, dtype, time_upper=None):
    time_upper = float(t_max if time_upper is None else time_upper)
    key = (device.type, device.index, str(dtype), round(time_upper, 6))
    if key not in power_stabilization_cache:
        points, pair_weights = build_power_stabilization_grid(
            power_stabilization_theta_count,
            power_stabilization_time_count,
            time_upper=time_upper,
        )
        power_stabilization_cache[key] = {
            "points": torch.as_tensor(points, device=device, dtype=dtype),
            "pair_weights": torch.as_tensor(pair_weights, device=device, dtype=dtype),
            "delta_t": torch.tensor(
                (1.0 - power_stabilization_start_frac) * (time_upper - t_min) / max(1, power_stabilization_time_count - 1),
                device=device, dtype=dtype,
            ),
        }
    return power_stabilization_cache[key]


def reconstruct_fourier_signal(theta, cos_coeffs, sin_coeffs, coeffs):
    """Analytically reconstructs the Initial Condition from the precomputed Fourier series."""
    theta_rel = theta - coeffs["theta_origin"]
    angles = coeffs["two_pi"] * theta_rel * coeffs["modes"] / coeffs["theta_period"]
    return torch.cos(angles) @ cos_coeffs + torch.sin(angles) @ sin_coeffs


def wrap_theta_to_domain(theta):
    domain_width = th_max - th_min
    return ((theta - th_min) % domain_width) + th_min


def build_gaussian_biased_collocation_points(num_points, time_upper=None, log_prefix="Static"):
    """[DeepXDE Note for Beginners]: DeepXDE usually samples collocation points uniformly. 
    However, for the non-linear LLE PDE, the dynamics are highly localized around an optical 'breather' peak. 
    This custom sampler biases the spatial points (Gaussian around the peak) and time points (Beta distribution) 
    to force the network to focus on the hardest regions.
    """
    time_upper = float(t_max if time_upper is None else time_upper)
    gaussian_count = int(round(num_points * gaussian_collocation_fraction))
    uniform_count = num_points - gaussian_count

    # Spatially biased (focuses on the optical peak)
    theta_gaussian = np.random.normal(
        loc=theta_peak,
        scale=gaussian_collocation_sigma,
        size=(gaussian_count, 1),
    ).astype(np.float32)
    theta_gaussian = wrap_theta_to_domain(theta_gaussian).astype(np.float32)

    theta_uniform = np.random.uniform(th_min, th_max, size=(uniform_count, 1)).astype(np.float32)
    theta_samples_biased = np.vstack((theta_gaussian, theta_uniform)).astype(np.float32)
    np.random.shuffle(theta_samples_biased)

    # Temporally biased (causality - focuses more on earlier times to propagate physics forward correctly)
    time_samples_biased = np.random.beta(
        time_bias_beta_a,
        time_bias_beta_b,
        size=(num_points, 1),
    ).astype(np.float32)
    time_samples_biased = (t_min + (time_upper - t_min) * time_samples_biased).astype(np.float32)
    np.random.shuffle(time_samples_biased)

    collocation_points = np.hstack((theta_samples_biased, time_samples_biased)).astype(np.float32)
    print(
        f"[INFO] {log_prefix} Gaussian-biased collocation: "
        f"{gaussian_count} Gaussian + {uniform_count} uniform theta samples, "
        f"theta_peak={theta_peak:.4f}, sigma={gaussian_collocation_sigma:.4f}, "
        f"time_beta=({time_bias_beta_a:.1f}, {time_bias_beta_b:.1f}), "
        f"time_upper={time_upper:.4f}"
    )
    return collocation_points


class MultiScaleFourierCore(torch.nn.Module):
    """
    Standard PINNs suffer from 'spectral bias' (they struggle to learn high frequencies).
    This core projects the inputs into a series of Sine/Cosine waves at different scales (sigmas)
    and specific guessed frequencies (harmonics) to give the network a "head start".
    """
    def __init__(self, input_dim=2, hidden_dim=128, num_hidden_layers=5, output_dim=2, 
                 sigmas=(1.0, 10.0), features_per_scale=16):
        super().__init__()
        self.sigmas = tuple(float(sigma) for sigma in sigmas)
        self.features_per_scale = int(features_per_scale)
        self.theta_harmonics = tuple(int(mode) for mode in breather_theta_harmonics)
        self.time_harmonics = tuple(float(mode) for mode in breather_time_harmonics)

        for idx, sigma in enumerate(self.sigmas):
            projection = torch.randn(self.features_per_scale, input_dim) * sigma
            self.register_buffer(f"projection_{idx}", projection)

        self.register_buffer(
            "det_theta_modes",
            torch.tensor(self.theta_harmonics, dtype=torch.float32).view(1, -1),
        )
        self.register_buffer(
            "det_time_omegas",
            torch.tensor([(2.0 * math.pi / breather_period_guess) * mode for mode in self.time_harmonics],
                dtype=torch.float32,
            ).view(1, -1),
        )

        encoded_dim = (
            input_dim
            + 2 * self.features_per_scale * len(self.sigmas)
            + 2 * len(self.theta_harmonics)
            + 2 * len(self.time_harmonics)
        )
        
        # Build the standard MLP on top of the Fourier features
        layer_dims =[encoded_dim] + [hidden_dim] * num_hidden_layers + [output_dim]
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

        theta = x[:, 0:1] / theta_norm_scale + theta_center
        time_coord = x[:, 1:2] / time_norm_scale + time_center
        theta_modes = self.det_theta_modes.to(device=x.device, dtype=x.dtype)
        time_omegas = self.det_time_omegas.to(device=x.device, dtype=x.dtype)
        
        theta_angles = 2.0 * math.pi * (theta - theta_origin) * theta_modes / theta_period
        time_angles = (time_coord - t0) * time_omegas
        encoded.append(torch.sin(theta_angles))
        encoded.append(torch.cos(theta_angles))
        encoded.append(torch.sin(time_angles))
        encoded.append(torch.cos(time_angles))
        return torch.cat(encoded, dim=1)

    def forward(self, x):
        return self.network(self.encode(x))


class NormalizedChainRuleNet(dde.nn.NN):
    """
    [DeepXDE Note for Beginners]: DeepXDE expects the network to subclass `dde.nn.NN`
    and implement `forward(self, inputs)`.
    
    This architecture utilizes a "Hard Initial Condition Constraint". 
    Instead of calculating a loss against t=0, the model's output is structured as:
    Output(th, t) = IC_Exact(th) + (1 - exp(-5*t)) * NeuralNet(th, t)
    Notice that when t=0, the exponent term becomes 0, and the exact IC is recovered perfectly.
    """
    def __init__(self):
        super().__init__()
        self.core = MultiScaleFourierCore(
            sigmas=msffn_sigmas,
            features_per_scale=msffn_features_per_scale,
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
        raw = self.core(x_norm)
        theta, time_coord = self.denormalize_inputs(x_norm)
        coeffs = get_ic_fourier_tensors(x_norm.device, x_norm.dtype)
        
        # Hard IC Ansatz
        u_exact = reconstruct_fourier_signal(theta, coeffs["u_cos"], coeffs["u_sin"], coeffs)
        v_exact = reconstruct_fourier_signal(theta, coeffs["v_cos"], coeffs["v_sin"], coeffs)
        growth = 1.0 - torch.exp(-5.0 * torch.clamp(time_coord - coeffs["t0"], min=0.0))
        return torch.cat((u_exact, v_exact), dim=1) + growth * raw

    def forward(self, inputs):
        # We save normalized inputs so we can calculate exact derivatives via the chain rule later
        self.last_x_norm = self.normalize_inputs(inputs)
        return self.forward_from_normalized(self.last_x_norm)


net = NormalizedChainRuleNet()
curriculum_time_upper = t_min + curriculum_stage_upper_fracs[0] * (t_max - t_min)
custom_collocation_points = build_gaussian_biased_collocation_points(
    num_domain_points,
    time_upper=curriculum_time_upper,
    log_prefix="Curriculum stage 1/4",
)

print(
    "[INFO] MsFFN-style core: "
    f"sigmas={msffn_sigmas}, "
    f"features_per_scale={msffn_features_per_scale}"
)
print(
    "[INFO] Breather-tuned Fourier features: "
    f"theta_modes={breather_theta_harmonics}, "
    f"time_period_guess={breather_period_guess:.4f}, "
    f"time_harmonics={breather_time_harmonics}"
)
print(
    "[INFO] Global power prior: "
    f"theta_points={power_stabilization_theta_count}, "
    f"time_points={power_stabilization_time_count}, "
    f"late_start_frac={power_stabilization_start_frac:.2f}"
)

def global_power_stabilization_loss(device, dtype):
    """
    Computes a physics prior: the integral of power (intensity) across the spatial domain 
    should change consistently over time according to the physics. 
    This operates entirely separate from the collocation points.
    """
    tensors = get_power_stabilization_tensors(device, dtype, time_upper=curriculum_time_upper)
    uv = net.forward_from_normalized(net.normalize_inputs(tensors["points"]))
    intensity = uv[:, 0:1].square() + uv[:, 1:2].square()
    
    # Integrate power over spatial domain
    power_by_time = intensity.view(
        power_stabilization_time_count,
        power_stabilization_theta_count,
        1,
    ).mean(dim=1) * theta_period
    
    # Calculate difference over time (derivative of power)
    power_dt = (power_by_time[1:] - power_by_time[:-1]) / (tensors["delta_t"] + 1e-12)
    return tensors["pair_weights"] * power_dt

# ==========================================
# 4. Physics / LLE Residual
# ==========================================
def compute_weighted_pde_residuals(x, y=None):
    """[DeepXDE Note for Beginners]: To evaluate the PDE, we must compute gradients of the network output (y)
    with respect to the input coordinates (x) using torch.autograd.
    
    CRITICAL: Because we normalized our inputs inside the network, PyTorch's `autograd.grad` will compute
    derivatives with respect to `x_norm`. To get the true physical derivatives (du/dx), we must multiply
    by the scaling factors via the Chain Rule!
    """
    x_norm = net.last_x_norm
    if y is None or x_norm is None or x_norm.shape[0] != x.shape[0]:
        x_norm = net.normalize_inputs(x)
        y = net.forward_from_normalized(x_norm)

    u, v = y[:, 0:1], y[:, 1:2]

    # First derivatives
    grad_u_norm = torch.autograd.grad(
        u, x_norm, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True,
    )[0]
    grad_v_norm = torch.autograd.grad(
        v, x_norm, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True,
    )[0]

    # Second spatial derivatives (needed for the diffusion term)
    du_dth2_norm = torch.autograd.grad(
        grad_u_norm[:, 0:1], x_norm, grad_outputs=torch.ones_like(grad_u_norm[:, 0:1]),
        create_graph=True, retain_graph=True,
    )[0][:, 0:1]
    dv_dth2_norm = torch.autograd.grad(
        grad_v_norm[:, 0:1], x_norm, grad_outputs=torch.ones_like(grad_v_norm[:, 0:1]),
        create_graph=True, retain_graph=True,
    )[0][:, 0:1]

    # Chain rule scaling to convert back to physical domain derivatives
    du_dt = grad_u_norm[:, 1:2] * time_norm_scale
    dv_dt = grad_v_norm[:, 1:2] * time_norm_scale
    du_dth2 = du_dth2_norm * (theta_norm_scale ** 2)
    dv_dth2 = dv_dth2_norm * (theta_norm_scale ** 2)

    # The actual physical residuals of the Lugiato-Lefever Equation (LLE)
    intensity = u.square() + v.square()
    res_u = du_dt - (-u + zeta * v - 0.5 * dv_dth2 - intensity * v + f)
    res_v = dv_dt - (-v - zeta * u + 0.5 * du_dth2 + intensity * u)
    
    # Causal weighting penalizes early-time errors more strictly than late-time errors
    time_frac = (x[:, 1:2] - t_min) / (t_max - t_min + 1e-12)
    causal_weight = torch.exp(-2.0 * time_frac)
    return causal_weight * res_u, causal_weight * res_v


def pde(x, y):
    """
    [DeepXDE Note for Beginners]: The `pde` function is passed to the DeepXDE Data object.
    It expects lists of PDE residuals. The DeepXDE optimizer will try to drive all returned
    residuals to zero.
    """
    weighted_res_u, weighted_res_v = compute_weighted_pde_residuals(x, y)
    power_stabilization = global_power_stabilization_loss(x.device, x.dtype)
    return[weighted_res_u, weighted_res_v, power_stabilization]


# [DeepXDE Note for Beginners]: `dde.data.TimePDE` combines the geometry, the PDE formulation, and boundary conditions.
# Because we used "Hard Constraints" directly in `NormalizedChainRuleNet`, the boundary/initial condition list is empty[].
# We set `num_domain=0` because we inject our own custom `anchors` (collocation points) instead of 
# letting DeepXDE sample them randomly.
data = dde.data.TimePDE(
    geomtime,
    pde,[],
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
    """
    [DeepXDE Note for Beginners]: Custom Callbacks function like Keras hooks (`on_epoch_end`).
    This callback strictly interrupts training if we approach the Kaggle timeout.
    """
    def __init__(self, max_duration):
        super().__init__()
        self.max_duration = max_duration
        self.start_time = time.time()
        
    def on_epoch_end(self):
        if time.time() - self.start_time > self.max_duration:
            self.model.stop_training = True


class TimeCurriculumScheduler(dde.callbacks.Callback):
    """
    Gradually expands the maximum training time bounds (t_max) as the loss crosses predefined thresholds.
    Training PINNs incrementally over time avoids "getting stuck" in early bad local minima.
    """
    def __init__(self, stage_upper_fracs, stage_loss_thresholds, min_stage_steps=1000):
        super().__init__()
        self.stage_upper_fracs = tuple(float(v) for v in stage_upper_fracs)
        self.stage_loss_thresholds = tuple(float(v) for v in stage_loss_thresholds)
        self.min_stage_steps = int(min_stage_steps)

    def on_train_begin(self):
        self.stage_index = 0
        self.last_seen_logged_step = -1
        self.stage_start_step = 0
        self.apply_stage(self.stage_index)

    def on_epoch_end(self):
        if self.stage_index >= len(self.stage_upper_fracs) - 1:
            return
        if not self.model.losshistory.steps:
            return

        step = int(self.model.losshistory.steps[-1])
        if step == self.last_seen_logged_step:
            return
        self.last_seen_logged_step = step

        if step - self.stage_start_step < self.min_stage_steps:
            return

        loss_train = self.model.losshistory.loss_train[-1]
        total_loss = float(np.sum(loss_train))
        threshold = self.stage_loss_thresholds[self.stage_index]
        if total_loss > threshold:
            return

        self.stage_index += 1
        self.stage_start_step = step
        self.apply_stage(self.stage_index)

    def apply_stage(self, stage_index):
        global curriculum_time_upper
        stage_frac = self.stage_upper_fracs[stage_index]
        curriculum_time_upper = t_min + stage_frac * (t_max - t_min)
        
        # Build new points encompassing the extended time domain
        new_anchors = build_gaussian_biased_collocation_points(
            num_domain_points,
            time_upper=curriculum_time_upper,
            log_prefix=f"Curriculum stage {stage_index + 1}/{len(self.stage_upper_fracs)}",
        )
        # Update DeepXDE dataset mid-training
        self.model.data.replace_with_anchors(new_anchors)
        self.model.data.test_x, self.model.data.test_y, self.model.data.test_aux_vars = None, None, None
        self.model.train_state.set_data_test(*self.model.data.test())
        print(
            "[INFO] Curriculum expansion: "
            f"stage={stage_index + 1}/{len(self.stage_upper_fracs)}, "
            f"time_upper={curriculum_time_upper:.4f}"
        )

    def set_final_stage(self, preserve_current_anchors=False):
        final_stage_index = len(self.stage_upper_fracs) - 1
        self.stage_index = final_stage_index
        if preserve_current_anchors and abs(curriculum_time_upper - t_max) < 1e-6:
            return
        self.apply_stage(self.stage_index)


def configure_pytorch_lbfgs(total_iters, inner_iters):
    """
    [DeepXDE Note for Beginners]: L-BFGS is a second-order optimizer crucial for PINNs.
    Unlike standard PyTorch where you pass arguments during instantiation, DeepXDE configures 
    it globally via its `config` settings object.
    """
    dde.optimizers.config.set_LBFGS_options(maxiter=total_iters)
    lbfgs_options = dde.optimizers.config.LBFGS_options
    inner_iters = min(inner_iters, total_iters)
    lbfgs_options["iter_per_step"] = inner_iters
    lbfgs_options["fun_per_step"] = max(
        1,
        lbfgs_options["maxfun"] * inner_iters // max(1, total_iters),
    )


def residual_scores_for_points(points_np, batch_size):
    """Calculates the physical PDE error at a given batch of points without tracking gradients."""
    original_requires_grad =[param.requires_grad for param in net.parameters()]
    for param in net.parameters():
        param.requires_grad_(False)

    scores =[]
    try:
        for start in range(0, points_np.shape[0], batch_size):
            stop = start + batch_size
            x_batch = torch.tensor(
                points_np[start:stop],
                device=device,
                dtype=torch.float32,
            ).requires_grad_(True)
            weighted_res_u, weighted_res_v = compute_weighted_pde_residuals(x_batch)
            score = torch.sqrt(
                weighted_res_u[:, 0].square() + weighted_res_v[:, 0].square() + 1e-12
            )
            scores.append(score.detach().cpu().numpy())
            del x_batch, weighted_res_u, weighted_res_v, score
        return np.concatenate(scores, axis=0)
    finally:
        for param, requires_grad in zip(net.parameters(), original_requires_grad):
            param.requires_grad_(requires_grad)


class R3Resampler(dde.callbacks.Callback):
    """
    R3 Resampler (Residual-Based Adaptive Refinement).
    Periodically evaluates the current collocation points, discards those with low PDE error,
    and replaces them with newly sampled points to force the network to focus on harder regions.
    """
    def __init__(self, period, retain_fraction, score_batch_size):
        super().__init__()
        self.period = int(period)
        self.retain_fraction = float(retain_fraction)
        self.score_batch_size = int(score_batch_size)
        self.has_updated = False

    def on_epoch_end(self):
        if curriculum_time_upper < t_max - 1e-6:
            return

        step = int(self.model.train_state.step)
        if step == 0 or step % self.period != 0:
            return

        current_points = np.asarray(self.model.data.train_x_all, dtype=np.float32)
        if current_points.shape[0] == 0:
            return

        # Find points with highest PDE errors
        residual_scores = residual_scores_for_points(current_points, self.score_batch_size)
        retain_count = max(1, int(round(current_points.shape[0] * self.retain_fraction)))
        retain_indices = np.argpartition(residual_scores, -retain_count)[-retain_count:]
        
        retained_points = current_points[retain_indices].astype(np.float32)
        retained_scores = residual_scores[retain_indices]

        # Generate fresh replacement points
        resampled_count = current_points.shape[0] - retain_count
        refreshed_points = build_gaussian_biased_collocation_points(
            resampled_count,
            time_upper=curriculum_time_upper,
            log_prefix=f"R3 refresh step {step}",
        )
        updated_points = np.vstack((retained_points, refreshed_points)).astype(np.float32)
        np.random.shuffle(updated_points)

        # Inject the refined points back into DeepXDE
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
        self.has_updated = True

        print(
            f"[INFO] R3 refresh at step {step}: retained {retain_count}/{current_points.shape[0]} "
            f"points ({100.0 * retain_count / current_points.shape[0]:.1f}%), "
            f"retained_mean={float(retained_scores.mean()):.3e}, "
            f"retained_max={float(retained_scores.max()):.3e}"
        )


def model_uv(t_in, th_in, need_x=False):
    """Helper formatting output wrapper strictly used by `prepare.evaluate_mse`."""
    # DeepXDE inputs are concatenated [theta, time]
    x = torch.cat((th_in, t_in), dim=1)
    if need_x:
        x = x.requires_grad_(True)
    else:
        x = x.detach()
    uv = net(x)
    return uv[:, 0:1], uv[:, 1:2], x

# Loss weights order corresponds directly to [res_u, res_v, power_stabilization] returned by `pde`.
loss_weights =[3.0, 3.0, 0.5]
curriculum_callback = TimeCurriculumScheduler(
    stage_upper_fracs=curriculum_stage_upper_fracs,
    stage_loss_thresholds=curriculum_stage_loss_thresholds,
    min_stage_steps=curriculum_min_stage_steps,
)
r3_callback = R3Resampler(
    period=r3_period,
    retain_fraction=r3_retain_fraction,
    score_batch_size=r3_score_batch_size,
)

# Phase 1: Adam is great for fast navigation of the initial loss landscape
model.compile("adam", lr=1e-3, loss_weights=loss_weights)
time_callback_adam = TimeBasedEarlyStopping(adam_time_limit)

try:
    print("\n[INFO] Phase 1: Adam optimization")
    losshistory, train_state = model.train(
        iterations=100000,
        callbacks=[time_callback_adam, curriculum_callback, r3_callback],
        display_every=1000,
    )
    
    # Phase 2: L-BFGS uses a Hessian approximation to polish the physics solution precisely to zero
    print("\n[INFO] Phase 2: L-BFGS optimization")
    time_callback_lbfgs = TimeBasedEarlyStopping(max_train_time)
    time_callback_lbfgs.start_time = t_start_training  # Base it on total elapsed time overall
    curriculum_callback.set_final_stage(preserve_current_anchors=r3_callback.has_updated)

    # Give L-BFGS more of the budget, but keep each PyTorch step short enough for callbacks to run.
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
    
    # Isolate training grid from ground truth evaluation
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
