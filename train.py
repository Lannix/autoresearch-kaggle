"""
Autoresearch PINN training script for LLE using DeepXDE.
Contains internal time-management to guarantee final evaluation before Kaggle timeout.

[DeepXDE Note for Beginners]:
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
from dataclasses import dataclass
import numpy as np
import torch
import deepxde as dde

# TIME_BUDGET: total allowed wall-clock training time on Kaggle.
# get_training_setup: loads PDE metadata and the exact initial-condition slice.
# evaluate_mse: computes the hidden validation score after training finishes.
from prepare import TIME_BUDGET, get_training_setup, evaluate_mse

# ==========================================
# 1. Initialization and Setup
# ==========================================
# Use float32 for speed on Kaggle T4 GPUs.
dde.config.set_default_float("float32")
# Fix all random generators so different experiments are comparable.
dde.config.set_random_seed(42)

# device: active PyTorch device for tensors and the neural network.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")

# t_start_training: wall-clock reference time used for time-budget control.
t_start_training = time.time()

# Load isolated training setup from prepare.py. 
# This ensures we don't accidentally leak the interior ground truth dataset during training.
# setup: safe metadata bundle containing domain bounds, PDE parameters, and the exact IC slice.
setup = get_training_setup()
# t_min/t_max: physical time interval of the PDE domain.
t_min, t_max = setup["t_bounds"]
# th_min/th_max: spatial theta interval of the periodic domain.
th_min, th_max = setup["th_bounds"]
# zeta: cavity detuning parameter in the LLE.
zeta = setup["params"]["zeta"]
# f: external pump / drive strength in the LLE.
f = setup["params"]["f"]

# ic: dictionary containing the exact initial-condition slice at t=t0.
ic = setup["initial_conditions"]
# t0: physical time where the exact initial condition is known.
t0 = ic["t0"]
# th0_arr: theta coordinates of the known initial-condition slice.
th0_arr = ic["th0_arr"]
# u0/v0: real and imaginary parts of the initial condition.
u0 = ic["u0"]
v0 = ic["v0"]

# Extract grid info from initial conditions
# theta_samples: flattened theta grid used for Fourier reconstruction.
theta_samples = np.asarray(th0_arr, dtype=np.float64).reshape(-1)
# u0_samples/v0_samples: flattened initial-condition values.
u0_samples = np.asarray(u0, dtype=np.float64).reshape(-1)
v0_samples = np.asarray(v0, dtype=np.float64).reshape(-1)
# theta_step: approximate theta spacing of the IC grid.
theta_step = float(np.median(np.diff(theta_samples)))
# theta_period: full periodic length of the theta domain.
theta_period = float(theta_step * theta_samples.size)
# theta_origin: reference theta used by the Fourier series.
theta_origin = float(theta_samples[0])
# theta_peak: location of the strongest initial intensity peak.
theta_peak = float(theta_samples[int(np.argmax(u0_samples**2 + v0_samples**2))])


def build_real_fourier_coeffs(values):
    """
    Computes Fourier coefficients of the Initial Conditions.
    This is used later to construct an analytical function that exactly reconstructs 
    the t=0 state, acting as a "Hard Constraint" for the PINN.
    """
    # coeffs: complex FFT coefficients of the real-valued signal.
    coeffs = np.fft.rfft(values) / values.size
    # cos_coeffs/sin_coeffs: real Fourier-series coefficients used for exact reconstruction.
    cos_coeffs = 2.0 * coeffs.real
    sin_coeffs = -2.0 * coeffs.imag
    cos_coeffs[0] = coeffs[0].real
    sin_coeffs[0] = 0.0
    if values.size % 2 == 0:
        cos_coeffs[-1] = coeffs[-1].real
        sin_coeffs[-1] = 0.0
    return cos_coeffs.astype(np.float32), sin_coeffs.astype(np.float32)


@dataclass(frozen=True)
class DomainConfig:
    """Physical domain constants and normalization factors used across the script."""

    t_min: float
    t_max: float
    th_min: float
    th_max: float
    zeta: float
    f: float
    t0: float
    theta_origin: float
    theta_period: float
    theta_peak: float
    theta_center: float
    theta_half_span: float
    time_center: float
    time_half_span: float
    theta_norm_scale: float
    time_norm_scale: float

    @property
    def theta_span(self):
        """Full physical width of the periodic theta domain."""
        return self.th_max - self.th_min

    @property
    def time_span(self):
        """Full physical width of the time domain."""
        return self.t_max - self.t_min


@dataclass(frozen=True)
class SamplerConfig:
    """Controls how PDE collocation points are sampled."""

    num_domain_points: int
    gaussian_fraction: float
    gaussian_sigma: float
    time_beta_a: float
    time_beta_b: float


@dataclass(frozen=True)
class FeatureConfig:
    """Controls the MsFFN-style random and deterministic Fourier feature bank."""

    sigmas: tuple[float, ...]
    features_per_scale: int
    theta_harmonics: tuple[int, ...]
    period_guess: float
    time_harmonics: tuple[float, ...]


@dataclass(frozen=True)
class PowerPriorConfig:
    """Controls the late-time global-power stabilization prior."""

    theta_count: int
    time_count: int
    start_frac: float


@dataclass(frozen=True)
class CurriculumConfig:
    """Controls the staged time-domain expansion schedule."""

    upper_fracs: tuple[float, ...]
    loss_thresholds: tuple[float, ...]
    min_stage_steps: int


@dataclass(frozen=True)
class OptimizationConfig:
    """Controls optimizer budgets, L-BFGS settings, and R3 behavior."""

    eval_reserve: int
    adam_fraction: float
    lbfgs_total_iters: int
    lbfgs_inner_iters: int
    r3_period: int
    r3_retain_fraction: float
    r3_max_retain_fraction: float
    r3_retain_growth_per_refresh: float
    r3_score_batch_size: int
    loss_weights: tuple[float, float, float]

# Pre-calculate Fourier modes for exact IC reconstruction
# u0_* / v0_*: Fourier coefficients of the exact initial condition.
u0_cos_coeffs, u0_sin_coeffs = build_real_fourier_coeffs(u0_samples)
v0_cos_coeffs, v0_sin_coeffs = build_real_fourier_coeffs(v0_samples)
# fourier_modes: integer mode indices 0,1,2,... for the Fourier series.
fourier_modes = np.arange(u0_cos_coeffs.size, dtype=np.float32)

# domain: one object that groups the physical bounds, PDE constants, and normalization factors.
domain = DomainConfig(
    t_min=t_min,
    t_max=t_max,
    th_min=th_min,
    th_max=th_max,
    zeta=zeta,
    f=f,
    t0=t0,
    theta_origin=theta_origin,
    theta_period=theta_period,
    theta_peak=theta_peak,
    theta_center=float((th_min + th_max) * 0.5),
    theta_half_span=float((th_max - th_min) * 0.5 + 1e-12),
    time_center=float((t_min + t_max) * 0.5),
    time_half_span=float((t_max - t_min) * 0.5 + 1e-12),
    theta_norm_scale=float(1.0 / (float((th_max - th_min) * 0.5) + 1e-12)),
    time_norm_scale=float(1.0 / (float((t_max - t_min) * 0.5) + 1e-12)),
)

# ==========================================
# 2. DeepXDE Geometry and Domain
# ==========================================
# [DeepXDE Note for Beginners]: We define the spatial (Interval) and temporal (TimeDomain) domains, 
# then multiply them to get the spatio-temporal domain (GeometryXTime).
# geom: 1D periodic spatial interval in theta.
geom = dde.geometry.Interval(domain.th_min, domain.th_max)
# timedomain: 1D time interval for the PDE.
timedomain = dde.geometry.TimeDomain(domain.t_min, domain.t_max)
# geomtime: full spatio-temporal domain object used by DeepXDE.
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# sampler_config: controls the custom Gaussian/Beta collocation sampler.
sampler_config = SamplerConfig(
    num_domain_points=30000,
    gaussian_fraction=0.80,
    gaussian_sigma=0.15 * domain.theta_span,
    time_beta_a=1.0,
    time_beta_b=3.0,
)

# feature_config: controls the random MsFFN features plus deterministic breather harmonics.
feature_config = FeatureConfig(
    sigmas=(1.0, 10.0),
    features_per_scale=16,
    theta_harmonics=(1, 2, 3, 4, 5),
    period_guess=0.9990,
    time_harmonics=(1.0, 2.0),
)

# power_prior_config: controls how the late-time global-power prior is discretized.
power_prior_config = PowerPriorConfig(
    theta_count=512,
    time_count=6,
    start_frac=0.80,
)

# curriculum_config: controls when and how the visible time horizon expands.
curriculum_config = CurriculumConfig(
    upper_fracs=(0.25, 0.50, 0.75, 1.00),
    loss_thresholds=(0.25, 0.10, 0.045),
    min_stage_steps=1000,
)

# optimization_config: optimizer timing, L-BFGS settings, and R3 refresh behavior.
optimization_config = OptimizationConfig(
    eval_reserve=45,
    adam_fraction=0.60,
    lbfgs_total_iters=5000,
    lbfgs_inner_iters=250,
    r3_period=5000,
    r3_retain_fraction=0.10,
    r3_max_retain_fraction=0.30,
    r3_retain_growth_per_refresh=0.10,
    r3_score_batch_size=1024,
    loss_weights=(3.0, 3.0, 0.5),
)

# ==========================================
# 3. Neural Network Architecture
# ==========================================
# ic_fourier_cache: cached IC tensors keyed by device/dtype.
ic_fourier_cache = {}
# power_stabilization_cache: cached tensors for the nonlocal power prior.
power_stabilization_cache = {}
# curriculum_time_upper: current maximum time visible to the curriculum sampler.
curriculum_time_upper = float(domain.t_max)


def get_ic_fourier_tensors(device, dtype):
    """Caches fourier tensors on the correct PyTorch device to avoid host-to-device bottlenecks."""
    # key: unique identifier for a specific device/dtype combination.
    key = (device.type, device.index, str(dtype))
    if key not in ic_fourier_cache:
        ic_fourier_cache[key] = {
            # modes: Fourier mode indices as a row vector.
            "modes": torch.as_tensor(fourier_modes, device=device, dtype=dtype).view(1, -1),
            # u_cos/u_sin/v_cos/v_sin: exact IC coefficients on the target device.
            "u_cos": torch.as_tensor(u0_cos_coeffs, device=device, dtype=dtype).view(-1, 1),
            "u_sin": torch.as_tensor(u0_sin_coeffs, device=device, dtype=dtype).view(-1, 1),
            "v_cos": torch.as_tensor(v0_cos_coeffs, device=device, dtype=dtype).view(-1, 1),
            "v_sin": torch.as_tensor(v0_sin_coeffs, device=device, dtype=dtype).view(-1, 1),
            # theta_origin/theta_period/two_pi/t0: scalar constants needed by the ansatz.
            "theta_origin": torch.tensor(domain.theta_origin, device=device, dtype=dtype),
            "theta_period": torch.tensor(domain.theta_period, device=device, dtype=dtype),
            "two_pi": torch.tensor(2.0 * math.pi, device=device, dtype=dtype),
            "t0": torch.tensor(domain.t0, device=device, dtype=dtype),
        }
    return ic_fourier_cache[key]


def build_power_stabilization_grid(theta_count, time_count, time_upper=None):
    """Creates a regular grid to calculate the physical 'power conservation' prior."""
    # time_upper: current curriculum-dependent upper time bound.
    time_upper = float(domain.t_max if time_upper is None else time_upper)
    # theta_points/time_points: 1D grids used to build the nonlocal power prior.
    theta_points = np.linspace(
        domain.th_min, domain.th_max, theta_count, endpoint=False, dtype=np.float32
    ).reshape(-1, 1)
    time_points = np.linspace(
        domain.t_min + power_prior_config.start_frac * (time_upper - domain.t_min),
        time_upper, time_count, dtype=np.float32
    ).reshape(-1, 1)
    
    # theta_grid/time_grid: broadcasted 2D mesh over theta and time.
    theta_grid = np.tile(theta_points[None, :, :], (time_count, 1, 1))
    time_grid = np.tile(time_points[:, None, :], (1, theta_count, 1))
    # points: flattened [theta, time] coordinates fed through the network.
    points = np.concatenate((theta_grid, time_grid), axis=2).reshape(-1, 2)

    # pair_times: all time slices except the first, used for finite differences.
    pair_times = time_points[1:]
    # pair_weights: late-time emphasis weights for the power derivative penalty.
    pair_weights = ((pair_times - domain.t_min) / (domain.time_span + 1e-12)) ** 4.0
    return points.astype(np.float32), pair_weights.astype(np.float32)


def get_power_stabilization_tensors(device, dtype, time_upper=None):
    # time_upper: current curriculum-dependent upper time bound.
    time_upper = float(domain.t_max if time_upper is None else time_upper)
    # key: cache key that changes with device, dtype, or time horizon.
    key = (device.type, device.index, str(dtype), round(time_upper, 6))
    if key not in power_stabilization_cache:
        # points/pair_weights: numpy arrays generated for the power prior.
        points, pair_weights = build_power_stabilization_grid(
            power_prior_config.theta_count,
            power_prior_config.time_count,
            time_upper=time_upper,
        )
        power_stabilization_cache[key] = {
            # points: flattened evaluation coordinates for the power prior.
            "points": torch.as_tensor(points, device=device, dtype=dtype),
            # pair_weights: weights applied to each finite-difference time pair.
            "pair_weights": torch.as_tensor(pair_weights, device=device, dtype=dtype),
            # delta_t: spacing between consecutive time slices in the power grid.
            "delta_t": torch.tensor(
                (1.0 - power_prior_config.start_frac) * (time_upper - domain.t_min) / max(1, power_prior_config.time_count - 1),
                device=device, dtype=dtype,
            ),
        }
    return power_stabilization_cache[key]


def reconstruct_fourier_signal(theta, cos_coeffs, sin_coeffs, coeffs):
    """Analytically reconstructs the Initial Condition from the precomputed Fourier series."""
    # theta_rel: theta measured relative to the Fourier reference origin.
    theta_rel = theta - coeffs["theta_origin"]
    # angles: Fourier phases k * 2pi * (theta - origin) / period.
    angles = coeffs["two_pi"] * theta_rel * coeffs["modes"] / coeffs["theta_period"]
    return torch.cos(angles) @ cos_coeffs + torch.sin(angles) @ sin_coeffs


def wrap_theta_to_domain(theta):
    # domain_width: length of the periodic theta interval.
    domain_width = domain.theta_span
    return ((theta - domain.th_min) % domain_width) + domain.th_min


def build_gaussian_biased_collocation_points(num_points, time_upper=None, log_prefix="Static"):
    """[DeepXDE Note for Beginners]: DeepXDE usually samples collocation points uniformly. 
    However, for the non-linear LLE PDE, the dynamics are highly localized around an optical 'breather' peak. 
    This custom sampler biases the spatial points (Gaussian around the peak) and time points (Beta distribution) 
    to force the network to focus on the hardest regions.
    """
    # time_upper: current curriculum-dependent time ceiling for sampling.
    time_upper = float(domain.t_max if time_upper is None else time_upper)
    # gaussian_count/uniform_count: split between peak-focused and global theta samples.
    gaussian_count = int(round(num_points * sampler_config.gaussian_fraction))
    uniform_count = num_points - gaussian_count

    # Spatially biased (focuses on the optical peak)
    # theta_gaussian: theta samples concentrated near the current breather peak.
    theta_gaussian = np.random.normal(
        loc=domain.theta_peak,
        scale=sampler_config.gaussian_sigma,
        size=(gaussian_count, 1),
    ).astype(np.float32)
    theta_gaussian = wrap_theta_to_domain(theta_gaussian).astype(np.float32)

    # theta_uniform: uniform theta samples that preserve global coverage.
    theta_uniform = np.random.uniform(domain.th_min, domain.th_max, size=(uniform_count, 1)).astype(np.float32)
    # theta_samples_biased: final mixed theta pool after shuffling.
    theta_samples_biased = np.vstack((theta_gaussian, theta_uniform)).astype(np.float32)
    np.random.shuffle(theta_samples_biased)

    # Temporally biased (causality - focuses more on earlier times to propagate physics forward correctly)
    # time_samples_biased: Beta-distributed times emphasizing the early transient.
    time_samples_biased = np.random.beta(
        sampler_config.time_beta_a,
        sampler_config.time_beta_b,
        size=(num_points, 1),
    ).astype(np.float32)
    time_samples_biased = (domain.t_min + (time_upper - domain.t_min) * time_samples_biased).astype(np.float32)
    np.random.shuffle(time_samples_biased)

    # collocation_points: final [theta, time] anchors passed to DeepXDE.
    collocation_points = np.hstack((theta_samples_biased, time_samples_biased)).astype(np.float32)
    print(
        f"[INFO] {log_prefix} Gaussian-biased collocation: "
        f"{gaussian_count} Gaussian + {uniform_count} uniform theta samples, "
        f"theta_peak={domain.theta_peak:.4f}, sigma={sampler_config.gaussian_sigma:.4f}, "
        f"time_beta=({sampler_config.time_beta_a:.1f}, {sampler_config.time_beta_b:.1f}), "
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
        # sigmas: random Fourier-feature scales.
        self.sigmas = tuple(float(sigma) for sigma in sigmas)
        # features_per_scale: number of random projections at each sigma.
        self.features_per_scale = int(features_per_scale)
        # theta_harmonics/time_harmonics: deterministic breather-specific feature frequencies.
        self.theta_harmonics = tuple(int(mode) for mode in feature_config.theta_harmonics)
        self.time_harmonics = tuple(float(mode) for mode in feature_config.time_harmonics)

        for idx, sigma in enumerate(self.sigmas):
            # projection: random matrix mapping inputs into Fourier angles.
            projection = torch.randn(self.features_per_scale, input_dim) * sigma
            self.register_buffer(f"projection_{idx}", projection)

        # det_theta_modes: deterministic spatial harmonics added as prior knowledge.
        self.register_buffer(
            "det_theta_modes",
            torch.tensor(self.theta_harmonics, dtype=torch.float32).view(1, -1),
        )
        # det_time_omegas: angular frequencies derived from the guessed breather period.
        self.register_buffer(
            "det_time_omegas",
            torch.tensor([(2.0 * math.pi / feature_config.period_guess) * mode for mode in self.time_harmonics],
                dtype=torch.float32,
            ).view(1, -1),
        )

        # encoded_dim: width of the concatenated feature vector after all encodings.
        encoded_dim = (
            input_dim
            + 2 * self.features_per_scale * len(self.sigmas)
            + 2 * len(self.theta_harmonics)
            + 2 * len(self.time_harmonics)
        )
        
        # Build the standard MLP on top of the Fourier features
        # layer_dims: widths of every linear layer in the MLP head.
        layer_dims =[encoded_dim] + [hidden_dim] * num_hidden_layers + [output_dim]
        # layers: Python list used to build the final torch.nn.Sequential block.
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
        # encoded: list of feature blocks that will be concatenated into one vector.
        encoded = [x]
        for idx in range(len(self.sigmas)):
            # projection: one random Fourier projection matrix.
            projection = getattr(self, f"projection_{idx}")
            # angles: Fourier phases produced by the random projection.
            angles = 2.0 * math.pi * (x @ projection.T)
            encoded.append(torch.sin(angles))
            encoded.append(torch.cos(angles))

        # theta/time_coord: convert normalized inputs back to physical coordinates.
        theta = x[:, 0:1] / domain.theta_norm_scale + domain.theta_center
        time_coord = x[:, 1:2] / domain.time_norm_scale + domain.time_center
        # theta_modes/time_omegas: deterministic harmonic tensors on the current device/dtype.
        theta_modes = self.det_theta_modes.to(device=x.device, dtype=x.dtype)
        time_omegas = self.det_time_omegas.to(device=x.device, dtype=x.dtype)
        
        # theta_angles/time_angles: phases for the deterministic breather-tuned harmonics.
        theta_angles = 2.0 * math.pi * (theta - domain.theta_origin) * theta_modes / domain.theta_period
        time_angles = (time_coord - domain.t0) * time_omegas
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
        # slab_split_frac: normalized location of the time-slab interface.
        self.slab_split_frac = 0.50
        # slab_split_time: physical time where the early and late subnetworks hand off.
        self.slab_split_time = domain.t_min + self.slab_split_frac * domain.time_span
        # slab_split_time_norm: same split location expressed in normalized network coordinates.
        self.slab_split_time_norm = (self.slab_split_time - domain.time_center) * domain.time_norm_scale
        # core_early/core_late: separate subnetworks for the early and late time slabs.
        # Each slab is slightly narrower than the monolithic winner so the total runtime stays manageable.
        self.core_early = MultiScaleFourierCore(
            hidden_dim=96,
            num_hidden_layers=4,
            sigmas=feature_config.sigmas,
            features_per_scale=feature_config.features_per_scale,
        )
        self.core_late = MultiScaleFourierCore(
            hidden_dim=96,
            num_hidden_layers=4,
            sigmas=feature_config.sigmas,
            features_per_scale=feature_config.features_per_scale,
        )
        # regularizer: DeepXDE compatibility field; unused in this script.
        self.regularizer = None
        # last_x_norm: cached normalized inputs used later for chain-rule derivatives.
        self.last_x_norm = None

    def normalize_inputs(self, x):
        # theta/time_coord: physical coordinates extracted from DeepXDE's [theta, time] input order.
        theta = x[:, 0:1]
        time_coord = x[:, 1:2]
        # theta_norm/time_norm: coordinates rescaled into the neural-network domain.
        theta_norm = (theta - domain.theta_center) * domain.theta_norm_scale
        time_norm = (time_coord - domain.time_center) * domain.time_norm_scale
        return torch.cat((theta_norm, time_norm), dim=1)

    def denormalize_inputs(self, x_norm):
        # theta/time_coord: normalized coordinates mapped back to physical units.
        theta = x_norm[:, 0:1] / domain.theta_norm_scale + domain.theta_center
        time_coord = x_norm[:, 1:2] / domain.time_norm_scale + domain.time_center
        return theta, time_coord

    def forward_with_core(self, x_norm, core):
        # raw: unconstrained network correction predicted by the selected slab subnetwork.
        raw = core(x_norm)
        # theta/time_coord: physical coordinates needed by the hard-constraint ansatz.
        theta, time_coord = self.denormalize_inputs(x_norm)
        # coeffs: cached Fourier tensors for exact IC reconstruction.
        coeffs = get_ic_fourier_tensors(x_norm.device, x_norm.dtype)
        
        # Hard IC Ansatz
        # u_exact/v_exact: exact real and imaginary initial-condition signals.
        u_exact = reconstruct_fourier_signal(theta, coeffs["u_cos"], coeffs["u_sin"], coeffs)
        v_exact = reconstruct_fourier_signal(theta, coeffs["v_cos"], coeffs["v_sin"], coeffs)
        # growth: time gate that is zero at t=t0 and gradually unlocks the NN correction.
        growth = 1.0 - torch.exp(-5.0 * torch.clamp(time_coord - coeffs["t0"], min=0.0))
        return torch.cat((u_exact, v_exact), dim=1) + growth * raw

    def forward_from_normalized(self, x_norm):
        # early_uv: solution produced by the early-slab subnetwork on the current coordinates.
        early_uv = self.forward_with_core(x_norm, self.core_early)
        # time_coord: physical time used to decide which slab owns each point.
        _, time_coord = self.denormalize_inputs(x_norm)
        # late_mask: points assigned to the late subnetwork.
        late_mask = time_coord[:, 0] > self.slab_split_time
        if not torch.any(late_mask):
            return early_uv

        # x_norm_late: normalized coordinates belonging to the late slab.
        x_norm_late = x_norm[late_mask]
        # split_time_norm: constant normalized time coordinate at the slab interface.
        split_time_norm = torch.full_like(x_norm_late[:, 1:2], self.slab_split_time_norm)
        # x_split_late: same late-slab theta locations but evaluated exactly at the slab interface.
        x_split_late = torch.cat((x_norm_late[:, 0:1], split_time_norm), dim=1)
        # handoff_uv: early-slab solution passed forward to initialize the late slab.
        handoff_uv = self.forward_with_core(x_split_late, self.core_early)
        # late_raw: late-slab correction predicted after the interface.
        late_raw = self.core_late(x_norm_late)
        # time_late: physical late-slab times used by the second gate.
        time_late = time_coord[late_mask]
        # late_growth: second hard gate that is zero exactly at the slab interface.
        late_growth = 1.0 - torch.exp(-5.0 * torch.clamp(time_late - self.slab_split_time, min=0.0))
        # late_uv: late-slab solution that starts from the handed-off interface state.
        late_uv = handoff_uv + late_growth * late_raw

        # Stitch the slab predictions back together.
        stitched_uv = early_uv.clone()
        stitched_uv[late_mask] = late_uv
        return stitched_uv

    def forward(self, inputs):
        # We save normalized inputs so we can calculate exact derivatives via the chain rule later
        self.last_x_norm = self.normalize_inputs(inputs)
        return self.forward_from_normalized(self.last_x_norm)


net = NormalizedChainRuleNet()
# curriculum_time_upper: first curriculum stage before training begins.
curriculum_time_upper = domain.t_min + curriculum_config.upper_fracs[0] * domain.time_span
# custom_collocation_points: first batch of PDE anchors injected into DeepXDE.
custom_collocation_points = build_gaussian_biased_collocation_points(
    sampler_config.num_domain_points,
    time_upper=curriculum_time_upper,
    log_prefix="Curriculum stage 1/4",
)

print(
    "[INFO] XPINN-style two-slab core: "
    f"sigmas={feature_config.sigmas}, "
    f"features_per_scale={feature_config.features_per_scale}, "
    f"slab_split_frac={0.50:.2f}, slab_hidden=96, slab_layers=4"
)
print(
    "[INFO] Breather-tuned Fourier features: "
    f"theta_modes={feature_config.theta_harmonics}, "
    f"time_period_guess={feature_config.period_guess:.4f}, "
    f"time_harmonics={feature_config.time_harmonics}"
)
print(
    "[INFO] Global power prior: "
    f"theta_points={power_prior_config.theta_count}, "
    f"time_points={power_prior_config.time_count}, "
    f"late_start_frac={power_prior_config.start_frac:.2f}"
)
print(
    "[INFO] Progressive R3 retention: "
    f"start={optimization_config.r3_retain_fraction:.2f}, "
    f"max={optimization_config.r3_max_retain_fraction:.2f}, "
    f"growth={optimization_config.r3_retain_growth_per_refresh:.2f}"
)

def global_power_stabilization_loss(device, dtype):
    """
    Computes a physics prior: the integral of power (intensity) across the spatial domain 
    should change consistently over time according to the physics. 
    This operates entirely separate from the collocation points.
    """
    # tensors: cached coordinates, weights, and time spacing for the power prior.
    tensors = get_power_stabilization_tensors(device, dtype, time_upper=curriculum_time_upper)
    # uv: model prediction evaluated on the nonlocal power grid.
    uv = net.forward_from_normalized(net.normalize_inputs(tensors["points"]))
    # intensity: local field power |psi|^2 = u^2 + v^2.
    intensity = uv[:, 0:1].square() + uv[:, 1:2].square()
    
    # Integrate power over spatial domain
    # power_by_time: estimated total cavity power for each sampled late-time slice.
    power_by_time = intensity.view(
        power_prior_config.time_count,
        power_prior_config.theta_count,
        1,
    ).mean(dim=1) * domain.theta_period
    
    # Calculate difference over time (derivative of power)
    # power_dt: finite-difference estimate of how total power changes over time.
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
    # x_norm: normalized coordinates used internally by the network.
    x_norm = net.last_x_norm
    if y is None or x_norm is None or x_norm.shape[0] != x.shape[0]:
        x_norm = net.normalize_inputs(x)
        y = net.forward_from_normalized(x_norm)

    # u/v: predicted real and imaginary fields at the requested points.
    u, v = y[:, 0:1], y[:, 1:2]

    # First derivatives
    # grad_u_norm/grad_v_norm: first derivatives with respect to normalized coordinates.
    grad_u_norm = torch.autograd.grad(
        u, x_norm, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True,
    )[0]
    grad_v_norm = torch.autograd.grad(
        v, x_norm, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True,
    )[0]

    # Second spatial derivatives (needed for the diffusion term)
    # du_dth2_norm/dv_dth2_norm: second theta-derivatives in normalized coordinates.
    du_dth2_norm = torch.autograd.grad(
        grad_u_norm[:, 0:1], x_norm, grad_outputs=torch.ones_like(grad_u_norm[:, 0:1]),
        create_graph=True, retain_graph=True,
    )[0][:, 0:1]
    dv_dth2_norm = torch.autograd.grad(
        grad_v_norm[:, 0:1], x_norm, grad_outputs=torch.ones_like(grad_v_norm[:, 0:1]),
        create_graph=True, retain_graph=True,
    )[0][:, 0:1]

    # Chain rule scaling to convert back to physical domain derivatives
    # du_dt/dv_dt: true time derivatives in physical coordinates.
    du_dt = grad_u_norm[:, 1:2] * domain.time_norm_scale
    dv_dt = grad_v_norm[:, 1:2] * domain.time_norm_scale
    # du_dth2/dv_dth2: true second theta-derivatives in physical coordinates.
    du_dth2 = du_dth2_norm * (domain.theta_norm_scale ** 2)
    dv_dth2 = dv_dth2_norm * (domain.theta_norm_scale ** 2)

    # The actual physical residuals of the Lugiato-Lefever Equation (LLE)
    # intensity: local optical intensity |psi|^2.
    intensity = u.square() + v.square()
    # res_u/res_v: real and imaginary PDE mismatches that training tries to drive to zero.
    res_u = du_dt - (-u + domain.zeta * v - 0.5 * dv_dth2 - intensity * v + domain.f)
    res_v = dv_dt - (-v - domain.zeta * u + 0.5 * du_dth2 + intensity * u)
    
    # Causal weighting penalizes early-time errors more strictly than late-time errors
    # time_frac: physical time normalized into [0, 1].
    time_frac = (x[:, 1:2] - domain.t_min) / (domain.time_span + 1e-12)
    # causal_weight: early-time emphasis factor applied to the PDE residual.
    causal_weight = torch.exp(-2.0 * time_frac)
    return causal_weight * res_u, causal_weight * res_v


def pde(x, y):
    """
    [DeepXDE Note for Beginners]: The `pde` function is passed to the DeepXDE Data object.
    It expects lists of PDE residuals. The DeepXDE optimizer will try to drive all returned
    residuals to zero.
    """
    # weighted_res_u/weighted_res_v: pointwise PDE residual channels.
    weighted_res_u, weighted_res_v = compute_weighted_pde_residuals(x, y)
    # power_stabilization: extra nonlocal prior that stabilizes late-time dynamics.
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
# model: central DeepXDE object that combines the dataset and the neural network.
model = dde.Model(data, net)

# ==========================================
# 6. Training Setup
# ==========================================
# max_train_time: time actually available for Adam + L-BFGS after reserving evaluation time.
max_train_time = TIME_BUDGET - optimization_config.eval_reserve
# adam_time_limit: fraction of the budget reserved for Adam exploration.
adam_time_limit = max_train_time * optimization_config.adam_fraction

print(
    f"[INFO] Starting training. Total budget: {TIME_BUDGET}s. "
    f"Reserved for eval: {optimization_config.eval_reserve}s."
)


def replace_model_anchors(model, anchors, sync_train_state=False):
    """Replace DeepXDE anchors and refresh cached train/test datasets consistently."""
    model.data.replace_with_anchors(np.asarray(anchors, dtype=np.float32))
    model.data.test_x = None
    model.data.test_y = None
    model.data.test_aux_vars = None
    if sync_train_state:
        model.train_state.set_data_train(
            model.data.train_x,
            model.data.train_y,
            model.data.train_aux_vars,
        )
    model.train_state.set_data_test(*model.data.test())


class TimeBasedEarlyStopping(dde.callbacks.Callback):
    """
    [DeepXDE Note for Beginners]: Custom Callbacks function like Keras hooks (`on_epoch_end`).
    This callback strictly interrupts training if we approach the Kaggle timeout.
    """
    def __init__(self, max_duration):
        super().__init__()
        # max_duration: allowed training time in seconds for this phase.
        self.max_duration = max_duration
        # start_time: timestamp used to compute elapsed wall-clock time.
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
        # stage_upper_fracs: fractions of the full time domain visible at each stage.
        self.stage_upper_fracs = tuple(float(v) for v in stage_upper_fracs)
        # stage_loss_thresholds: total-loss cutoffs that unlock the next stage.
        self.stage_loss_thresholds = tuple(float(v) for v in stage_loss_thresholds)
        # min_stage_steps: minimum logged steps before a stage change is allowed.
        self.min_stage_steps = int(min_stage_steps)

    def on_train_begin(self):
        # stage_index: current curriculum stage number.
        self.stage_index = 0
        # last_seen_logged_step: prevents duplicate processing of the same logged step.
        self.last_seen_logged_step = -1
        # stage_start_step: step where the current stage began.
        self.stage_start_step = 0
        self.apply_stage(self.stage_index)

    def on_epoch_end(self):
        if self.stage_index >= len(self.stage_upper_fracs) - 1:
            return
        if not self.model.losshistory.steps:
            return

        # step: latest logged DeepXDE step.
        step = int(self.model.losshistory.steps[-1])
        if step == self.last_seen_logged_step:
            return
        self.last_seen_logged_step = step

        if step - self.stage_start_step < self.min_stage_steps:
            return

        # loss_train: latest vector of loss channels logged by DeepXDE.
        loss_train = self.model.losshistory.loss_train[-1]
        # total_loss: scalar used to decide whether the current stage is ready to expand.
        total_loss = float(np.sum(loss_train))
        # threshold: stage-specific target loss.
        threshold = self.stage_loss_thresholds[self.stage_index]
        if total_loss > threshold:
            return

        self.stage_index += 1
        self.stage_start_step = step
        self.apply_stage(self.stage_index)

    def apply_stage(self, stage_index):
        global curriculum_time_upper
        # stage_frac: fraction of the full time window visible at this stage.
        stage_frac = self.stage_upper_fracs[stage_index]
        curriculum_time_upper = domain.t_min + stage_frac * domain.time_span
        
        # Build new points encompassing the extended time domain
        # new_anchors: fresh PDE anchors spanning the expanded time window.
        new_anchors = build_gaussian_biased_collocation_points(
            sampler_config.num_domain_points,
            time_upper=curriculum_time_upper,
            log_prefix=f"Curriculum stage {stage_index + 1}/{len(self.stage_upper_fracs)}",
        )
        # Update DeepXDE dataset mid-training
        replace_model_anchors(self.model, new_anchors)
        print(
            "[INFO] Curriculum expansion: "
            f"stage={stage_index + 1}/{len(self.stage_upper_fracs)}, "
            f"time_upper={curriculum_time_upper:.4f}"
        )

    def set_final_stage(self, preserve_current_anchors=False):
        final_stage_index = len(self.stage_upper_fracs) - 1
        self.stage_index = final_stage_index
        if preserve_current_anchors and abs(curriculum_time_upper - domain.t_max) < 1e-6:
            return
        self.apply_stage(self.stage_index)


def configure_pytorch_lbfgs(total_iters, inner_iters):
    """
    [DeepXDE Note for Beginners]: L-BFGS is a second-order optimizer crucial for PINNs.
    Unlike standard PyTorch where you pass arguments during instantiation, DeepXDE configures 
    it globally via its `config` settings object.
    """
    dde.optimizers.config.set_LBFGS_options(maxiter=total_iters)
    # lbfgs_options: DeepXDE's global dictionary for PyTorch L-BFGS settings.
    lbfgs_options = dde.optimizers.config.LBFGS_options
    inner_iters = min(inner_iters, total_iters)
    lbfgs_options["iter_per_step"] = inner_iters
    lbfgs_options["fun_per_step"] = max(
        1,
        lbfgs_options["maxfun"] * inner_iters // max(1, total_iters),
    )


def residual_scores_for_points(points_np, batch_size):
    """Calculates the physical PDE error at a given batch of points without tracking gradients."""
    # original_requires_grad: remembers the current grad state of every parameter.
    original_requires_grad =[param.requires_grad for param in net.parameters()]
    for param in net.parameters():
        param.requires_grad_(False)

    # scores: list of numpy chunks that will be concatenated into one score vector.
    scores =[]
    try:
        for start in range(0, points_np.shape[0], batch_size):
            # stop: exclusive end index of this mini-batch.
            stop = start + batch_size
            # x_batch: current mini-batch of points as a differentiable torch tensor.
            x_batch = torch.tensor(
                points_np[start:stop],
                device=device,
                dtype=torch.float32,
            ).requires_grad_(True)
            # weighted_res_u/weighted_res_v: PDE residual channels on this mini-batch.
            weighted_res_u, weighted_res_v = compute_weighted_pde_residuals(x_batch)
            # score: single residual magnitude per point used for R3 ranking.
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
    def __init__(self, period, retain_fraction, max_retain_fraction, retain_growth_per_refresh, score_batch_size):
        super().__init__()
        # period: training-step interval between possible R3 refreshes.
        self.period = int(period)
        # retain_fraction: starting fraction of the hardest anchors kept during the first refresh.
        self.retain_fraction = float(retain_fraction)
        # max_retain_fraction: upper limit for later, more exploitative R3 refreshes.
        self.max_retain_fraction = float(max_retain_fraction)
        # retain_growth_per_refresh: how much the keep fraction grows after each completed refresh.
        self.retain_growth_per_refresh = float(retain_growth_per_refresh)
        # score_batch_size: mini-batch size used to evaluate residual scores.
        self.score_batch_size = int(score_batch_size)
        # has_updated: remembers whether the anchors were refreshed at least once.
        self.has_updated = False
        # refresh_count: number of successful R3 updates already performed.
        self.refresh_count = 0

    def on_epoch_end(self):
        if curriculum_time_upper < domain.t_max - 1e-6:
            return

        # step: current DeepXDE optimization step.
        step = int(self.model.train_state.step)
        if step == 0 or step % self.period != 0:
            return

        # current_points: active collocation anchors currently stored in DeepXDE.
        current_points = np.asarray(self.model.data.train_x_all, dtype=np.float32)
        if current_points.shape[0] == 0:
            return

        # Find points with highest PDE errors
        # residual_scores: one scalar difficulty score per active anchor.
        residual_scores = residual_scores_for_points(current_points, self.score_batch_size)
        # current_retain_fraction: progressively shifts from exploration to exploitation.
        current_retain_fraction = min(
            self.max_retain_fraction,
            self.retain_fraction + self.refresh_count * self.retain_growth_per_refresh,
        )
        # retain_count: number of hardest anchors preserved during the refresh.
        retain_count = max(1, int(round(current_points.shape[0] * current_retain_fraction)))
        # retain_indices: indices of the highest-scoring anchors.
        retain_indices = np.argpartition(residual_scores, -retain_count)[-retain_count:]
        
        # retained_points: hardest anchors kept from the current set.
        retained_points = current_points[retain_indices].astype(np.float32)
        # retained_scores: their residual scores, used for logging.
        retained_scores = residual_scores[retain_indices]

        # Generate fresh replacement points
        # resampled_count: number of anchors replaced with new samples.
        resampled_count = current_points.shape[0] - retain_count
        # refreshed_points: new anchors sampled from the winning biased sampler.
        refreshed_points = build_gaussian_biased_collocation_points(
            resampled_count,
            time_upper=curriculum_time_upper,
            log_prefix=f"R3 refresh step {step}",
        )
        # updated_points: final anchor set after retain + resample.
        updated_points = np.vstack((retained_points, refreshed_points)).astype(np.float32)
        np.random.shuffle(updated_points)

        # Inject the refined points back into DeepXDE
        replace_model_anchors(self.model, updated_points, sync_train_state=True)
        self.has_updated = True
        self.refresh_count += 1

        print(
            f"[INFO] R3 refresh at step {step}: retained {retain_count}/{current_points.shape[0]} "
            f"points ({100.0 * retain_count / current_points.shape[0]:.1f}%), "
            f"retain_fraction={current_retain_fraction:.2f}, "
            f"retained_mean={float(retained_scores.mean()):.3e}, "
            f"retained_max={float(retained_scores.max()):.3e}"
        )


def model_uv(t_in, th_in, need_x=False):
    """Helper formatting output wrapper strictly used by `prepare.evaluate_mse`."""
    # DeepXDE inputs are concatenated [theta, time]
    # x: coordinates packed in the exact order expected by the network.
    x = torch.cat((th_in, t_in), dim=1)
    if need_x:
        x = x.requires_grad_(True)
    else:
        x = x.detach()
    uv = net(x)
    return uv[:, 0:1], uv[:, 1:2], x

# Loss weights order corresponds directly to [res_u, res_v, power_stabilization] returned by `pde`.
# loss_weights: relative importance of real PDE, imaginary PDE, and power prior losses.
loss_weights = list(optimization_config.loss_weights)
# curriculum_callback: expands the visible time horizon as training improves.
curriculum_callback = TimeCurriculumScheduler(
    stage_upper_fracs=curriculum_config.upper_fracs,
    stage_loss_thresholds=curriculum_config.loss_thresholds,
    min_stage_steps=curriculum_config.min_stage_steps,
)
# r3_callback: retains hard points and resamples easier ones during Adam.
r3_callback = R3Resampler(
    period=optimization_config.r3_period,
    retain_fraction=optimization_config.r3_retain_fraction,
    max_retain_fraction=optimization_config.r3_max_retain_fraction,
    retain_growth_per_refresh=optimization_config.r3_retain_growth_per_refresh,
    score_batch_size=optimization_config.r3_score_batch_size,
)

# Phase 1: Adam is great for fast navigation of the initial loss landscape
model.compile("adam", lr=1e-3, loss_weights=loss_weights)
# time_callback_adam: hard wall-clock stop for the Adam phase.
time_callback_adam = TimeBasedEarlyStopping(adam_time_limit)

try:
    print("\n[INFO] Phase 1: Adam optimization")
    # losshistory: DeepXDE object storing the logged loss values.
    # train_state: DeepXDE object storing the latest training step and state.
    losshistory, train_state = model.train(
        iterations=100000,
        callbacks=[time_callback_adam, curriculum_callback, r3_callback],
        display_every=1000,
    )
    
    # Phase 2: L-BFGS uses a Hessian approximation to polish the physics solution precisely to zero
    print("\n[INFO] Phase 2: L-BFGS optimization")
    # time_callback_lbfgs: hard wall-clock stop for the L-BFGS phase.
    time_callback_lbfgs = TimeBasedEarlyStopping(max_train_time)
    time_callback_lbfgs.start_time = t_start_training  # Base it on total elapsed time overall
    curriculum_callback.set_final_stage(preserve_current_anchors=r3_callback.has_updated)

    # Give L-BFGS more of the budget, but keep each PyTorch step short enough for callbacks to run.
    configure_pytorch_lbfgs(
        optimization_config.lbfgs_total_iters,
        optimization_config.lbfgs_inner_iters,
    )
    model.compile("L-BFGS", loss_weights=loss_weights)
    losshistory, train_state = model.train(
        iterations=10000,
        callbacks=[time_callback_lbfgs],
        display_every=10,
    )
    
    # total_training_time: elapsed time spent in Adam + L-BFGS before final evaluation.
    total_training_time = time.time() - t_start_training

    # ==========================================
    # 7. Final Evaluation
    # ==========================================
    print(f"\n[TIME UP / CONVERGED] Mandatory evaluation at {total_training_time:.1f}s ...")
    
    # Isolate training grid from ground truth evaluation
    net.eval()
    # val_mse: hidden validation metric used to rank experiments.
    val_mse = evaluate_mse(model_uv, device, dtype=torch.float32)
    # peak_vram_mb: maximum GPU memory allocated during the run.
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    # num_params: total number of trainable parameters in the network.
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
