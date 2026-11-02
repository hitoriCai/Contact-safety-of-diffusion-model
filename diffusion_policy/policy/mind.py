"""MIND safety filter for the PushT low-dimensional policy.

This module implements the online mixed-integer problem in Section 4.3 of
``relu.pdf``.  A separately trained ReLU NNDM maps

    [18-D keypoint state, optional measured 2-D pusher position,
     previous 2-D action, current 2-D action]

to an 18-D state increment.  The ReLU graph is encoded exactly with binary
variables and interval-derived Big-M constants.  Safety-index synthesis (SIS)
is deliberately outside the scope of this module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence
import threading

import numpy as np
import torch
import torch.nn as nn

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # Keep diffusion-only imports usable without Gurobi.
    gp = None
    GRB = None


class ReLUDynamicsMLP(nn.Module):
    """Checkpoint-compatible 22/24 -> ReLU -> ReLU -> 18 dynamics MLP."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        if len(hidden_dims) != 2:
            raise ValueError("MIND currently requires exactly two ReLU hidden layers.")
        self.net = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dims[0])),
            nn.ReLU(),
            nn.Linear(int(hidden_dims[0]), int(hidden_dims[1])),
            nn.ReLU(),
            nn.Linear(int(hidden_dims[1]), output_dim),
        )


class ControlAffineDynamicsMLP(nn.Module):
    """Checkpoint-compatible augmented-state affine dynamics network.

    It predicts ``delta_z = f(z) + G(z) v`` for
    ``z=[18 T keypoints, 2 actual pusher coordinates]`` and ``v=u-p``.
    The model is nonlinear in state but affine in the current control, which
    makes each fixed-state MIND projection a linear program.
    """

    def __init__(self, state_dim=20, hidden_dims=(96, 96), control_dim=2):
        super().__init__()
        layers = []
        input_dim = int(state_dim)
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, int(hidden_dim)), nn.ReLU()))
            input_dim = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)
        self.drift_head = nn.Linear(input_dim, int(state_dim))
        self.control_head = nn.Linear(
            input_dim, int(state_dim) * int(control_dim))


@dataclass(frozen=True)
class MindConfig:
    enabled: bool = True
    checkpoint_path: str = "diffusion_policy/data/dynamics_model_3layer_64.pth"
    control_dt_sec: float = 0.1
    gamma: float = 1.0e-3
    safe_alpha: float = 1.0
    recovery_alpha: float = 0.01
    # Default recovery is nonworsening at every projected stage: an unsafe T
    # may stay put or move outward, but may not move farther inward. Set this
    # to True only to require an additional strict outward decrease at the
    # terminal stage of a multi-action chunk.
    recovery_terminal_step_only: bool = False
    # Enter recovery before the true clearance boundary, and leave only after
    # a larger clearance is restored. Zero margins recover the legacy logic.
    recovery_trigger_margin_px: float = 0.0
    recovery_release_margin_px: float = 0.0
    safe_slack_weight: float = 60.0
    recovery_slack_weight: float = 30.0
    terminal_slack_weight: float = 300.0
    # Multiplies the one-step L1 distance from the diffusion action. Keep 1.0
    # for historical evaluations; larger values accept more bounded CBF slack.
    tracking_cost_multiplier: float = 1.0
    clearance_scale: float = 1.0
    # Asymmetric end-region proxies from actual NNDM keypoints: left/right
    # bar midpoints (1,3)/(2,3), plus the lower region centroid (0,4,3).
    block_clearance_radius: float = 50.0
    center_keypoint_indices: Sequence[int] = (1, 2, 4)
    safety_keypoint_groups: Sequence[Sequence[int]] = ((1, 3), (2, 3), (0, 4, 3))
    safety_proxy_radii: Sequence[float] = (35.0, 35.0, 50.0)
    # The policy projects four diffusion actions one at a time, then executes
    # that four-action chunk open-loop.  This is deliberately *not* a joint
    # MPC: each one-step MIP receives the preceding projected action and an
    # NNDM-predicted state as its initial condition.
    execution_action_steps: int = 4
    # False preserves the established sequence of independent one-step MIPs.
    # True jointly optimizes the complete execution chunk with project_horizon.
    joint_mpc_enabled: bool = False
    max_action_delta_px: float = 25.0
    # Retained only so old Hydra/checkpoint configs still deserialize.  The
    # active controller uses ``project_sequence`` rather than a joint horizon.
    mip_horizon: int = 1
    safety_directions: int = 16
    parallel_workers: int = 1
    output_mode: str = "delta"
    objective_norm: str = "l1"
    action_search_radius: float = 1.0
    mip_gap: float = 0.0
    time_limit_sec: float = 5.0
    node_limit: int = 20_000
    fail_safe: str = "agent_hold"
    # Opt-in experiment: forbid CBF violations in every main-task one-step
    # solve. The original soft-CBF evaluation keeps the default False.
    hard_cbf_enabled: bool = False
    # Optional finite allowance for each one-step CBF inequality, in px^2.
    # None preserves the historical unbounded soft CBF when hard_cbf_enabled
    # is False. A positive cap must not be combined with hard CBF.
    safety_slack_cap_px2: Optional[float] = None
    print_status: bool = True
    verbose: bool = False

    @classmethod
    def from_mapping(cls, values: Optional[Mapping]) -> "MindConfig":
        return cls(**({} if values is None else dict(values)))


def _checkpoint_array(checkpoint: Mapping, *names: str) -> np.ndarray:
    for name in names:
        if name in checkpoint:
            value = checkpoint[name]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float64).reshape(-1)
    raise KeyError(f"Dynamics checkpoint is missing all of these keys: {names}.")


def load_dynamics_checkpoint(path: str):
    """Load the standalone NNDM and its training normalization metadata."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_absolute() and not checkpoint_path.is_file():
        checkpoint_path = Path(__file__).resolve().parents[2] / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MIND dynamics checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0 has no weights_only argument.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise KeyError("Dynamics checkpoint needs 'model_state_dict' or 'state_dict'.")

    # v4.0's checkpoint contains a state-dependent control-affine model rather
    # than the legacy 22->64->64->18 action-coupled ReLU NNDM.
    model_config = checkpoint.get("model_config", {})
    if int(model_config.get("state_dim", -1)) == 20:
        hidden_dims = tuple(int(value) for value in model_config["hidden_dims"])
        control_dim = int(model_config.get("control_dim", 2))
        if control_dim != 2:
            raise ValueError("Control-affine MIND requires a 2-D PushT control.")
        model = ControlAffineDynamicsMLP(20, hidden_dims, control_dim)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        state_mean = _checkpoint_array(checkpoint, "state_mean")
        state_std = _checkpoint_array(checkpoint, "state_std")
        if state_mean.size != 20 or state_std.size != 20:
            raise ValueError("Control-affine state normalization must contain 20 values.")
        if np.any(state_std <= 0):
            raise ValueError("Control-affine state normalization must be strictly positive.")
        return model, {
            "model_type": "control_affine",
            "state_mean": state_mean,
            "state_std": state_std,
        }

    # Accept both the training script's net.0/net.2/net.4 names and a bare
    # Sequential checkpoint whose keys are prefixed by module/model.
    linear_weights = []
    for key, tensor in state_dict.items():
        if key.endswith("weight") and torch.is_tensor(tensor) and tensor.ndim == 2:
            linear_weights.append((key, tensor))
    if len(linear_weights) != 3:
        raise ValueError(
            "MIND requires a three-Linear-layer checkpoint (22/24->64->64->18); "
            f"found {len(linear_weights)} matrix weights."
        )
    # Recover feed-forward order from dimensions instead of relying on state
    # dictionary insertion order or a particular module-name prefix.
    first = [item for item in linear_weights if item[1].shape[1] in (22, 24)]
    last = [item for item in linear_weights if item[1].shape[0] == 18]
    if len(first) != 1 or len(last) != 1 or first[0][0] == last[0][0]:
        raise ValueError("Could not identify unique 22/24-input and 18-output Linear layers.")
    middle = [item for item in linear_weights if item[0] not in {first[0][0], last[0][0]}]
    linear_weights = [first[0], middle[0], last[0]]
    weight_keys = [item[0] for item in linear_weights]
    shapes = [tuple(item[1].shape) for item in linear_weights]
    input_dim = shapes[0][1]
    hidden_dims = (shapes[0][0], shapes[1][0])
    output_dim = shapes[2][0]
    if shapes[1][1] != hidden_dims[0] or shapes[2][1] != hidden_dims[1]:
        raise ValueError(f"Dynamics checkpoint Linear dimensions do not chain: {shapes}.")
    if input_dim not in (22, 24) or output_dim != 18:
        raise ValueError(
            f"Expected a 22/24-input, 18-output NNDM, got {input_dim}-input/{output_dim}-output."
        )
    if hidden_dims != (64, 64):
        raise ValueError(
            f"Expected NNDM hidden dimensions (64, 64), got {hidden_dims}."
        )

    model = ReLUDynamicsMLP(input_dim, hidden_dims, output_dim)
    target_keys = ["net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias",
                   "net.4.weight", "net.4.bias"]
    source_prefixes = [key[:-len("weight")] for key in weight_keys]
    remapped = {}
    for layer_index, (target_weight, target_bias) in enumerate(zip(target_keys[::2], target_keys[1::2])):
        prefix = source_prefixes[layer_index]
        source_weight = prefix + "weight"
        source_bias = prefix + "bias"
        if source_bias not in state_dict:
            raise KeyError(f"Dynamics checkpoint is missing bias '{source_bias}'.")
        remapped[target_weight] = state_dict[source_weight]
        remapped[target_bias] = state_dict[source_bias]
    checkpoint_norm = checkpoint.get("normalizer", checkpoint)
    if input_dim == 24:
        if checkpoint.get("output_mode") not in {
                "standardized T keypoint delta 18",
                "standardized_normalized_keypoint_delta"}:
            raise ValueError("24D NNDM must predict standardized normalized keypoint deltas.")
        delta_mean = _checkpoint_array(checkpoint_norm, "delta_mean")
        delta_std = _checkpoint_array(checkpoint_norm, "delta_std")
        if delta_mean.size != 18 or delta_std.size != 18 or np.any(delta_std <= 0):
            raise ValueError("24D NNDM delta normalization must have 18 positive scales.")
        # Fold the output transformation into the final affine layer. All
        # existing MILP/interval code then sees raw normalized keypoint deltas.
        scale = torch.as_tensor(delta_std, dtype=remapped["net.4.weight"].dtype)
        offset = torch.as_tensor(delta_mean, dtype=remapped["net.4.bias"].dtype)
        remapped["net.4.weight"] = remapped["net.4.weight"] * scale[:, None]
        remapped["net.4.bias"] = remapped["net.4.bias"] * scale + offset
    model.load_state_dict(remapped, strict=True)
    model.eval()

    keypoint_min = _checkpoint_array(checkpoint_norm, "keypoint_min", "keypoints_min")
    try:
        keypoint_range = _checkpoint_array(checkpoint_norm, "keypoint_range", "keypoints_range")
    except KeyError:
        keypoint_max = _checkpoint_array(checkpoint_norm, "keypoint_max", "keypoints_max")
        keypoint_range = keypoint_max - keypoint_min
    action_min = _checkpoint_array(checkpoint_norm, "action_min")
    try:
        action_range = _checkpoint_array(checkpoint_norm, "action_range")
    except KeyError:
        action_max = _checkpoint_array(checkpoint_norm, "action_max")
        action_range = action_max - action_min

    if keypoint_min.size != 18 or keypoint_range.size != 18:
        raise ValueError("NNDM keypoint normalization must contain 18 values.")
    if action_min.size != 2 or action_range.size != 2:
        raise ValueError("NNDM action normalization must contain 2 values.")
    if np.any(keypoint_range <= 0) or np.any(action_range <= 0):
        raise ValueError("NNDM normalization ranges must be strictly positive.")
    normalization = {
        "keypoint_min": keypoint_min,
        "keypoint_range": keypoint_range,
        "action_min": action_min,
        "action_range": action_range,
    }
    if input_dim == 24:
        normalization["agent_min"] = _checkpoint_array(checkpoint_norm, "agent_min")
        normalization["agent_range"] = _checkpoint_array(checkpoint_norm, "agent_range")
        if normalization["agent_min"].size != 2 or normalization["agent_range"].size != 2:
            raise ValueError("24D NNDM pusher normalization must have two values.")
        if np.any(normalization["agent_range"] <= 0):
            raise ValueError("24D NNDM pusher normalization range must be positive.")
    return model, normalization


class MindSafetyFilter:
    """Exact-ReLU DCBF filter for one-step closed-loop projection."""

    def __init__(self, dynamics_model: nn.Module, normalization: Mapping, config: MindConfig):
        self.config = config
        self.keypoint_min = np.asarray(normalization["keypoint_min"], dtype=np.float64)
        self.keypoint_range = np.asarray(normalization["keypoint_range"], dtype=np.float64)
        self.action_min = np.asarray(normalization["action_min"], dtype=np.float64)
        self.action_range = np.asarray(normalization["action_range"], dtype=np.float64)
        self.agent_min = np.asarray(normalization.get("agent_min", [0., 0.]), dtype=np.float64)
        self.agent_range = np.asarray(normalization.get("agent_range", [1., 1.]), dtype=np.float64)
        self._measured_agent_norm = {}
        self._previous_action_norm = {}
        self._early_recovery_active = {}
        self._last_solver_status = {}
        self._last_solver_status_sequence = {}
        self._diagnostics_lock = threading.Lock()
        linears = [module for module in dynamics_model.net if isinstance(module, nn.Linear)]
        relus = [module for module in dynamics_model.net if isinstance(module, nn.ReLU)]
        if len(linears) != 3 or len(relus) != 2:
            raise ValueError("MIND NNDM must be Linear-ReLU-Linear-ReLU-Linear.")
        self.weights = [layer.weight.detach().cpu().numpy().astype(np.float64) for layer in linears]
        self.biases = [layer.bias.detach().cpu().numpy().astype(np.float64) for layer in linears]
        if self.weights[0].shape[1] not in (22, 24) or self.weights[-1].shape[0] != 18:
            raise ValueError("MIND NNDM must map 22/24 inputs to 18 outputs.")
        if config.output_mode not in {"delta", "next_state"}:
            raise ValueError("mind.output_mode must be 'delta' or 'next_state'.")
        if config.objective_norm != "l1":
            raise ValueError("Only the MILP-compatible L1 objective is implemented.")
        if config.fail_safe not in {"raise", "agent_hold"}:
            raise ValueError("mind.fail_safe must be raise or agent_hold.")
        if len(config.center_keypoint_indices) == 0:
            raise ValueError("mind.center_keypoint_indices cannot be empty.")
        if any(index < 0 or index >= 9 for index in config.center_keypoint_indices):
            raise ValueError("mind.center_keypoint_indices must be in [0, 8].")
        if config.execution_action_steps < 1:
            raise ValueError(
                "mind.execution_action_steps must be positive.")
        if config.max_action_delta_px <= 0:
            raise ValueError("mind.max_action_delta_px must be positive.")
        if config.mip_horizon < 1:
            raise ValueError("mind.mip_horizon must be positive.")
        if config.safety_directions < 4:
            raise ValueError("mind.safety_directions must be at least 4.")
        if config.parallel_workers < 1:
            raise ValueError("mind.parallel_workers must be positive.")
        if config.control_dt_sec <= 0 or config.gamma < 0:
            raise ValueError("mind.control_dt_sec must be positive and gamma nonnegative.")
        if not 0 < config.safe_alpha <= 1:
            raise ValueError("mind.safe_alpha must be in (0, 1].")
        if not 0 < config.recovery_alpha <= 1:
            raise ValueError("mind.recovery_alpha must be in (0, 1].")
        if config.recovery_trigger_margin_px < 0:
            raise ValueError(
                "mind.recovery_trigger_margin_px must be nonnegative.")
        if config.recovery_release_margin_px < \
                config.recovery_trigger_margin_px:
            raise ValueError(
                "mind.recovery_release_margin_px must be at least "
                "mind.recovery_trigger_margin_px.")
        if config.safe_slack_weight <= 0 or config.recovery_slack_weight <= 0:
            raise ValueError("mind slack weights must be positive.")
        if (not np.isfinite(config.tracking_cost_multiplier)
                or config.tracking_cost_multiplier <= 0):
            raise ValueError("mind.tracking_cost_multiplier must be finite and positive.")
        if config.safety_slack_cap_px2 is not None:
            if (config.hard_cbf_enabled
                    or not np.isfinite(config.safety_slack_cap_px2)
                    or config.safety_slack_cap_px2 <= 0):
                raise ValueError("mind.safety_slack_cap_px2 requires a finite positive soft-CBF cap.")
        if len(config.safety_keypoint_groups) == 0:
            raise ValueError("mind.safety_keypoint_groups cannot be empty.")
        if len(config.safety_keypoint_groups) != len(config.safety_proxy_radii):
            raise ValueError(
                "mind safety group/radius counts must match.")
        for group in config.safety_keypoint_groups:
            if len(group) == 0 or any(index < 0 or index >= 9 for index in group):
                raise ValueError(
                    "each mind safety keypoint group must contain indices in [0, 8].")
        if any(radius < 0 for radius in config.safety_proxy_radii):
            raise ValueError("mind.safety_proxy_radii must be nonnegative.")
        if not 0 < config.action_search_radius <= 1:
            raise ValueError("mind.action_search_radius must be in (0, 1].")
        self.reset()

    def __getstate__(self):
        """Allow EMA policy deepcopy without attempting to copy a thread lock."""
        state = self.__dict__.copy()
        state.pop("_diagnostics_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._diagnostics_lock = threading.Lock()

    def reset(self) -> None:
        self._previous_action_norm.clear()
        self._measured_agent_norm.clear()
        self._early_recovery_active.clear()
        self._last_solver_status.clear()
        self._last_solver_status_sequence.clear()
        self._last_visualization_data = {}
        self._solve_count = 0
        self._positive_slack_count = 0
        self._no_solution_count = 0
        self._total_safety_slack = 0.0
        self._max_safety_slack = 0.0

    def diagnostics_summary(self):
        """Return aggregate soft-DCBF diagnostics since the latest reset."""
        with self._diagnostics_lock:
            return {
                "solve_count": int(self._solve_count),
                "positive_slack_count": int(self._positive_slack_count),
                "no_solution_count": int(self._no_solution_count),
                "mean_safety_slack": (
                    self._total_safety_slack / self._solve_count
                    if self._solve_count else 0.0),
                "max_safety_slack": float(self._max_safety_slack),
            }

    def last_solver_status(self, stream_id=0):
        """Return the aggregate status for the most recent action chunk."""
        return self._last_solver_status.get(stream_id)

    def last_solver_status_sequence(self, stream_id=0):
        """Return one Gurobi status per projected action in the latest chunk."""
        return self._last_solver_status_sequence.get(stream_id, ())

    def last_visualization_data(self, stream_id=0):
        """Return one-step MIND/NNDM diagnostics for an evaluation overlay."""
        value = self._last_visualization_data.get(stream_id)
        return None if value is None else dict(value)

    def _store_visualization_data(
            self, stream_id, state_norm, previous_action_norm, action_ref,
            action_used, action_anchor, obstacle_positions, obstacle_radii,
            diagnostics):
        """Record the exact numerical NNDM prediction used for MIND step 1."""
        used_norm = np.clip(
            (np.asarray(action_used, dtype=np.float64) - self.action_min)
            / self.action_range, 0.0, 1.0)
        next_state_norm = self._predict_next_state_norm(
            state_norm, previous_action_norm, used_norm,
            self._measured_agent_norm.get(stream_id))
        indices = self.config.safety_keypoint_groups[0]
        center_now = self._center_raw(state_norm, indices)
        center_next = self._center_raw(next_state_norm, indices)
        obstacle = np.asarray(obstacle_positions[0], dtype=np.float64)
        safe_radius = (
            float(obstacle_radii[0]) * self.config.clearance_scale
            + float(self.config.safety_proxy_radii[0]))
        distance_now = float(np.linalg.norm(center_now - obstacle))
        distance_next = float(np.linalg.norm(center_next - obstacle))
        self._last_visualization_data[stream_id] = {
            'raw_action': np.asarray(action_ref, dtype=np.float64).tolist(),
            'used_action': np.asarray(action_used, dtype=np.float64).tolist(),
            'action_anchor': np.asarray(
                action_anchor, dtype=np.float64).tolist(),
            'center_now': center_now.tolist(),
            'center_next': center_next.tolist(),
            'safe_radius': float(safe_radius),
            'distance_now': distance_now,
            'distance_next': distance_next,
            'phi_now': float(safe_radius ** 2 - distance_now ** 2),
            'phi_next': float(safe_radius ** 2 - distance_next ** 2),
            'action_l1_change': float(np.abs(
                np.asarray(action_used) - np.asarray(action_ref)).sum()),
            'action_delta_from_anchor_linf': float(np.abs(
                np.asarray(action_used) - np.asarray(action_anchor)).max()),
            'status': str(diagnostics['status']),
            'solutions': int(diagnostics['solutions']),
            'runtime_sec': float(diagnostics['runtime_sec']),
            'nodes': int(diagnostics['nodes']),
            'safety_slack': (
                None if diagnostics['safety_slack'] is None
                else float(diagnostics['safety_slack'])),
        }

    def _predict_next_state_norm(self, state_norm, previous_action_norm,
                                 current_action_norm, measured_agent_norm=None):
        """Run the NNDM numerically to propagate a projected MPC state."""
        fixed = [state_norm]
        if self.weights[0].shape[1] == 24:
            if measured_agent_norm is None:
                raise ValueError("24D NNDM needs the measured pusher position.")
            fixed.append(np.asarray(measured_agent_norm, dtype=np.float64).reshape(2))
        fixed.extend((previous_action_norm, current_action_norm))
        value = np.concatenate(fixed)
        for layer_index, (weight, bias) in enumerate(
                zip(self.weights, self.biases)):
            value = weight @ value + bias
            if layer_index < len(self.weights) - 1:
                value = np.maximum(value, 0.0)
        if self.config.output_mode == "delta":
            return state_norm + value
        return value

    @staticmethod
    def _affine_bounds(weight, bias, lower, upper):
        positive = np.maximum(weight, 0.0)
        negative = np.minimum(weight, 0.0)
        return (
            positive @ lower + negative @ upper + bias,
            positive @ upper + negative @ lower + bias,
        )

    def _preactivation_bounds(self, input_lower, input_upper):
        lower, upper = input_lower, input_upper
        bounds = []
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            pre_lower, pre_upper = self._affine_bounds(weight, bias, lower, upper)
            bounds.append((pre_lower, pre_upper))
            if layer_index < len(self.weights) - 1:
                lower = np.maximum(pre_lower, 0.0)
                upper = np.maximum(pre_upper, 0.0)
            else:
                lower, upper = pre_lower, pre_upper
        return bounds

    @staticmethod
    def _add_relu(model, affine, lower: float, upper: float, name: str):
        if upper <= 0.0:
            return model.addVar(lb=0.0, ub=0.0, name=name)
        output = model.addVar(lb=0.0, ub=max(upper, 0.0), name=name)
        if lower >= 0.0:
            model.addConstr(output == affine, name=f"{name}_active")
            return output
        preactivation = model.addVar(lb=lower, ub=upper, name=f"{name}_pre")
        active = model.addVar(vtype=GRB.BINARY, name=f"{name}_binary")
        model.addConstr(preactivation == affine, name=f"{name}_definition")
        model.addConstr(output >= preactivation, name=f"{name}_above_pre")
        model.addConstr(output <= upper * active, name=f"{name}_active_upper")
        model.addConstr(
            output <= preactivation - lower * (1.0 - active),
            name=f"{name}_inactive_upper",
        )
        return output

    def _encode_network(self, model, state_norm, previous_action_norm, action_vars,
                        action_ref_norm, measured_agent_norm=None):
        radius = float(self.config.action_search_radius)
        action_lower = np.maximum(0.0, action_ref_norm - radius)
        action_upper = np.minimum(1.0, action_ref_norm + radius)
        fixed = [state_norm]
        if self.weights[0].shape[1] == 24:
            if measured_agent_norm is None:
                raise ValueError("24D NNDM needs the measured pusher position.")
            fixed.append(np.asarray(measured_agent_norm, dtype=np.float64).reshape(2))
        fixed.append(previous_action_norm)
        input_lower = np.concatenate((*fixed, action_lower))
        input_upper = np.concatenate((*fixed, action_upper))
        layer_bounds = self._preactivation_bounds(input_lower, input_upper)

        previous = [item for chunk in fixed for item in chunk] + list(action_vars)
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            lower, upper = layer_bounds[layer_index]
            current = []
            for neuron_index in range(weight.shape[0]):
                affine = gp.LinExpr(float(bias[neuron_index]))
                for input_index, coefficient in enumerate(weight[neuron_index]):
                    if coefficient != 0.0:
                        affine += float(coefficient) * previous[input_index]
                if layer_index == len(self.weights) - 1:
                    output = model.addVar(
                        lb=float(lower[neuron_index]),
                        ub=float(upper[neuron_index]),
                        name=f"z_{layer_index}_{neuron_index}",
                    )
                    model.addConstr(output == affine, name=f"linear_{layer_index}_{neuron_index}")
                else:
                    output = self._add_relu(
                        model, affine, float(lower[neuron_index]),
                        float(upper[neuron_index]), f"z_{layer_index}_{neuron_index}",
                    )
                current.append(output)
            previous = current

        if self.config.output_mode == "delta":
            delta_state = previous
            next_state = [state_norm[i] + delta_state[i] for i in range(18)]
        else:
            next_state = previous
            delta_state = [next_state[i] - state_norm[i] for i in range(18)]
        return delta_state, next_state

    def _encode_network_stage(self, model, state_values, state_lower,
                              state_upper, previous_action_values,
                              previous_action_lower, previous_action_upper,
                              action_vars, action_lower, action_upper, stage,
                              measured_agent_norm=None):
        """Encode one NNDM stage whose state may depend on earlier stages."""
        fixed_values = list(state_values)
        fixed_lower = [state_lower]
        fixed_upper = [state_upper]
        if self.weights[0].shape[1] == 24:
            if stage == 0:
                if measured_agent_norm is None:
                    raise ValueError("24D NNDM needs the measured pusher position.")
                agent_values = list(measured_agent_norm)
                agent_lower = np.asarray(measured_agent_norm, dtype=np.float64)
                agent_upper = agent_lower.copy()
            else:
                # Open-loop future pusher position is estimated by the last
                # projected target, in the separate pusher normalization.
                scale = self.action_range / self.agent_range
                offset = (self.action_min - self.agent_min) / self.agent_range
                agent_values = [previous_action_values[i] * float(scale[i]) +
                                float(offset[i]) for i in range(2)]
                agent_lower = previous_action_lower * scale + offset
                agent_upper = previous_action_upper * scale + offset
            fixed_values.extend(agent_values)
            fixed_lower.append(agent_lower)
            fixed_upper.append(agent_upper)
        fixed_values.extend(previous_action_values)
        fixed_lower.append(previous_action_lower)
        fixed_upper.append(previous_action_upper)
        input_lower = np.concatenate((*fixed_lower, action_lower))
        input_upper = np.concatenate((*fixed_upper, action_upper))
        layer_bounds = self._preactivation_bounds(input_lower, input_upper)

        previous = fixed_values + list(action_vars)
        for layer_index, (weight, bias) in enumerate(
                zip(self.weights, self.biases)):
            lower, upper = layer_bounds[layer_index]
            current = []
            for neuron_index in range(weight.shape[0]):
                affine = gp.LinExpr(float(bias[neuron_index]))
                for input_index, coefficient in enumerate(
                        weight[neuron_index]):
                    if coefficient != 0.0:
                        affine += float(coefficient) * previous[input_index]
                name = f"h{stage}_z_{layer_index}_{neuron_index}"
                if layer_index == len(self.weights) - 1:
                    output = model.addVar(
                        lb=float(lower[neuron_index]),
                        ub=float(upper[neuron_index]), name=name)
                    model.addConstr(
                        output == affine, name=f"{name}_definition")
                else:
                    output = self._add_relu(
                        model, affine, float(lower[neuron_index]),
                        float(upper[neuron_index]), name)
                current.append(output)
            previous = current

        output_lower, output_upper = layer_bounds[-1]
        if self.config.output_mode == "delta":
            next_state = [
                state_values[index] + previous[index]
                for index in range(18)]
            # Interval addition is conservative because it deliberately
            # ignores dependencies between x and the predicted delta.
            next_lower = state_lower + output_lower
            next_upper = state_upper + output_upper
        else:
            next_state = previous
            next_lower = output_lower
            next_upper = output_upper
        return next_state, next_lower, next_upper

    def _center_expression_and_bounds(self, state_values, state_lower,
                                      state_upper, indices=None):
        if indices is None:
            indices = self.config.center_keypoint_indices
        indices = np.asarray(indices, dtype=int)
        count = float(indices.size)
        expressions = []
        lower = np.zeros(2, dtype=np.float64)
        upper = np.zeros(2, dtype=np.float64)
        for coordinate in range(2):
            expression = gp.LinExpr()
            for keypoint_index in indices:
                state_index = 2 * keypoint_index + coordinate
                scale = float(self.keypoint_range[state_index] / count)
                offset = float(self.keypoint_min[state_index] / count)
                expression += offset + scale * state_values[state_index]
                lower[coordinate] += (
                    offset + scale * state_lower[state_index])
                upper[coordinate] += (
                    offset + scale * state_upper[state_index])
            expressions.append(expression)
        return expressions, lower, upper

    def _add_horizon_safety_constraints(
            self, model, current_values, current_lower, current_upper,
            next_values, next_lower, next_upper,
            nominal_current_norm, nominal_next_norm,
            obstacle_positions, obstacle_radii, initial_safe_pairs,
            stage, terminal):
        """MIND-style DCBF constraints linearized on a nominal rollout.

        Gradients and barrier values are constants from the reference
        actions' numerical NNDM rollout. The optimized current/next states
        remain exact MILP expressions through the encoded ReLU network, so no
        circle-direction selector binaries are needed.
        """
        slacks = []
        slack_scales = []
        slack_weights = []

        for proxy_index, (indices, proxy_radius) in enumerate(zip(
                self.config.safety_keypoint_groups,
                self.config.safety_proxy_radii)):
            current_proxy, _, _ = \
                self._center_expression_and_bounds(
                    current_values, current_lower, current_upper, indices)
            next_proxy, _, _ = self._center_expression_and_bounds(
                next_values, next_lower, next_upper, indices)
            nominal_current = self._center_raw(
                nominal_current_norm, indices)
            nominal_next = self._center_raw(nominal_next_norm, indices)
            for obstacle_index, (obstacle, obstacle_radius) in enumerate(
                    zip(obstacle_positions, obstacle_radii)):
                obstacle = np.asarray(obstacle, dtype=np.float64)
                safe_radius = (
                    float(obstacle_radius) * self.config.clearance_scale
                    + float(proxy_radius))
                relative = nominal_current - obstacle
                phi = float(safe_radius ** 2 - relative @ relative)
                gradient = -2.0 * relative
                # A proxy that is safe at the measured state must not spend
                # safety slack anywhere in this open-loop four-step chunk.
                # Also harden a previously unsafe proxy once the nominal
                # rollout predicts that it has recovered.
                hard_safe = bool(
                    initial_safe_pairs[proxy_index, obstacle_index]
                    or phi <= 0.0)

                nonworsening_slack = model.addVar(
                    lb=0.0, ub=0.0 if hard_safe else GRB.INFINITY,
                    name=(f"h{stage}_proxy{proxy_index}_obs"
                          f"{obstacle_index}_nonworsening_slack"))
                barrier_delta = (
                    float(gradient[0])
                    * (next_proxy[0] - current_proxy[0])
                    + float(gradient[1])
                    * (next_proxy[1] - current_proxy[1]))
                right_hand_side = (
                    -float(self.config.safe_alpha) * phi
                    if phi <= 0.0 else 0.0)
                model.addConstr(
                    barrier_delta
                    <= right_hand_side + nonworsening_slack,
                    name=(f"h{stage}_proxy{proxy_index}_obs"
                          f"{obstacle_index}_nonworsening"))
                slacks.append(nonworsening_slack)
                slack_scales.append(max(safe_radius ** 2, 1.0))
                slack_weights.append(float(
                    self.config.safe_slack_weight
                    if phi <= 0.0
                    else self.config.recovery_slack_weight))

                if not terminal:
                    continue
                terminal_relative = nominal_next - obstacle
                terminal_phi = float(
                    safe_radius ** 2
                    - terminal_relative @ terminal_relative)
                terminal_slack = model.addVar(
                    lb=0.0,
                    ub=(0.0 if hard_safe or terminal_phi <= 0.0
                        else GRB.INFINITY),
                    name=(f"h{stage}_proxy{proxy_index}_obs"
                          f"{obstacle_index}_terminal_slack"))
                terminal_gradient = -2.0 * terminal_relative
                linearized_terminal_phi = (
                    terminal_phi
                    + float(terminal_gradient[0])
                    * (next_proxy[0] - float(nominal_next[0]))
                    + float(terminal_gradient[1])
                    * (next_proxy[1] - float(nominal_next[1])))
                model.addConstr(
                    linearized_terminal_phi <= terminal_slack,
                    name=(f"h{stage}_proxy{proxy_index}_obs"
                          f"{obstacle_index}_terminal_safe"))
                slacks.append(terminal_slack)
                slack_scales.append(max(safe_radius ** 2, 1.0))
                slack_weights.append(float(self.config.terminal_slack_weight))
        return slacks, slack_scales, slack_weights

    def _center_raw(self, state_norm, indices=None):
        state_raw = state_norm * self.keypoint_range + self.keypoint_min
        points = state_raw.reshape(9, 2)
        if indices is None:
            indices = self.config.center_keypoint_indices
        return points[np.asarray(indices, dtype=int)].mean(axis=0)

    def _center_delta_expression(self, delta_state, indices=None):
        if indices is None:
            indices = self.config.center_keypoint_indices
        indices = np.asarray(indices, dtype=int)
        count = float(indices.size)
        delta_x = gp.LinExpr()
        delta_y = gp.LinExpr()
        for keypoint_index in indices:
            delta_x += float(self.keypoint_range[2 * keypoint_index] / count) * delta_state[2 * keypoint_index]
            delta_y += float(self.keypoint_range[2 * keypoint_index + 1] / count) * delta_state[2 * keypoint_index + 1]
        return delta_x, delta_y

    def _add_safety_constraints(self, model, delta_state, state_norm,
                                obstacle_positions, obstacle_radii,
                                stream_id=0, safety_mode="strict"):
        """Add a soft version of Eq. (10).

        The NNDM predicts a one-period increment, so multiplying Eq. (10) by
        dt gives grad(phi) Delta-x <= max(-phi, -gamma*dt).

        A nonnegative slack keeps the MILP feasible when the object is already
        too deep inside the safety region to recover in one control period.
        Separate safe/recovery alpha values avoid requiring an immediate exit
        while still demanding meaningful outward progress after penetration.
        When configured, an outer warning barrier enters this same recovery
        mode before the true clearance boundary. A larger release margin gives
        the mode hysteresis, preventing safe/recovery chatter at the boundary.
        """
        slacks = []
        slack_scales = []
        slack_weights = []
        for obstacle_index, (obstacle, obstacle_radius) in enumerate(
                zip(obstacle_positions, obstacle_radii)):
            for proxy_index, (indices, proxy_radius) in enumerate(zip(
                    self.config.safety_keypoint_groups,
                    self.config.safety_proxy_radii)):
                center = self._center_raw(state_norm, indices)
                delta_center_x, delta_center_y = \
                    self._center_delta_expression(delta_state, indices)
                relative = center - obstacle
                safe_radius = (
                    float(obstacle_radius) * self.config.clearance_scale
                    + float(proxy_radius))
                distance_squared = float(relative @ relative)
                warning_radius = (
                    safe_radius
                    + float(self.config.recovery_trigger_margin_px))
                release_radius = (
                    safe_radius
                    + float(self.config.recovery_release_margin_px))
                mode_key = (stream_id, obstacle_index, proxy_index)
                early_recovery = bool(
                    self._early_recovery_active.get(mode_key, False))
                # A recovery latch is set as soon as the current predicted
                # state enters the warning circle. It remains active until a
                # strictly larger clearance has been regained.
                if early_recovery:
                    if distance_squared >= release_radius ** 2:
                        early_recovery = False
                elif distance_squared < warning_radius ** 2:
                    early_recovery = True
                self._early_recovery_active[mode_key] = early_recovery

                if early_recovery:
                    barrier_radius = warning_radius
                    phi = barrier_radius ** 2 - distance_squared
                else:
                    barrier_radius = safe_radius
                    phi = barrier_radius ** 2 - distance_squared
                gradient = -2.0 * relative
                left_hand_side = (
                    gradient[0] * delta_center_x
                    + gradient[1] * delta_center_y)
                # ``safe_only`` is used by the recovery-specialized filter:
                # once its deliberately small proxy has already entered the
                # margin, leave the learned escape motion untouched instead
                # of adding a nonworsening/strict recovery CBF.
                if safety_mode == "safe_only" and phi > 0.0:
                    continue
                if phi <= 0.0:
                    right_hand_side = -self.config.safe_alpha * phi
                    slack_weight = self.config.safe_slack_weight
                elif safety_mode == "nonworsening":
                    # During an exterior pusher transfer the T should remain
                    # still rather than be forced to recover immediately. The
                    # first-order barrier may therefore not increase: the
                    # predicted center can stay put or move outward, but not
                    # move farther into the obstacle.
                    right_hand_side = 0.0
                    slack_weight = self.config.safe_slack_weight
                else:
                    recovery = max(
                        self.config.recovery_alpha * phi,
                        self.config.gamma * self.config.control_dt_sec)
                    right_hand_side = -recovery
                    slack_weight = self.config.recovery_slack_weight
                slack = model.addVar(
                    lb=0.0,
                    ub=(0.0 if self.config.hard_cbf_enabled
                        else (GRB.INFINITY if self.config.safety_slack_cap_px2 is None
                              else float(self.config.safety_slack_cap_px2))),
                    name=(f"mind_safety_slack_{obstacle_index}_"
                          f"proxy_{proxy_index}"))
                model.addConstr(
                    left_hand_side <= right_hand_side + slack,
                    name=(f"mind_safety_index_{obstacle_index}_"
                          f"proxy_{proxy_index}"),
                )
                slacks.append(slack)
                slack_scales.append(max(barrier_radius ** 2, 1.0))
                slack_weights.append(float(slack_weight))
        return slacks, slack_scales, slack_weights

    @staticmethod
    def _status_name(status: int) -> str:
        names = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.NODE_LIMIT: "NODE_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
        }
        return names.get(status, f"STATUS_{status}")

    def _solve(self, action_ref, state_norm, previous_action_norm,
               action_anchor,
               obstacle_positions, obstacle_radii, stream_id=0,
               safety_mode="strict"):
        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", int(self.config.verbose))
        environment.start()
        model = gp.Model("mind", env=environment)
        model.Params.OutputFlag = int(self.config.verbose)
        model.Params.MIPGap = float(self.config.mip_gap)
        model.Params.TimeLimit = float(self.config.time_limit_sec)
        model.Params.NodeLimit = int(self.config.node_limit)
        # Batch environments are parallelized by the policy. Keep each
        # independent Gurobi model single-threaded to avoid CPU oversubscription.
        model.Params.Threads = 2

        action_ref_norm = np.clip(
            (action_ref - self.action_min) / self.action_range, 0.0, 1.0)
        action_anchor_norm = np.clip(
            (action_anchor - self.action_min) / self.action_range, 0.0, 1.0)
        max_delta_norm = (
            float(self.config.max_action_delta_px) / self.action_range)
        radius = float(self.config.action_search_radius)
        action_vars = []
        for index, reference in enumerate(action_ref_norm):
            lower = max(
                0.0,
                float(reference) - radius,
                float(action_anchor_norm[index] - max_delta_norm[index]),
            )
            upper = min(
                1.0,
                float(reference) + radius,
                float(action_anchor_norm[index] + max_delta_norm[index]),
            )
            if lower > upper:
                raise RuntimeError(
                    "MIND action-search and max-action-delta bounds do not "
                    f"intersect for dimension {index}: {lower} > {upper}.")
            variable = model.addVar(
                lb=lower,
                ub=upper,
                name=f"u_{index}",
            )
            variable.Start = float(reference)
            action_vars.append(variable)

        delta_state, _ = self._encode_network(
            model, state_norm, previous_action_norm, action_vars,
            action_ref_norm, self._measured_agent_norm.get(stream_id))
        safety_slacks, safety_slack_scales, safety_slack_weights = \
            self._add_safety_constraints(
            model, delta_state, state_norm, obstacle_positions,
            obstacle_radii, stream_id=stream_id, safety_mode=safety_mode)

        absolute_errors = []
        for index, variable in enumerate(action_vars):
            error = model.addVar(lb=0.0, name=f"reference_error_{index}")
            model.addConstr(error >= variable - float(action_ref_norm[index]))
            model.addConstr(error >= float(action_ref_norm[index]) - variable)
            absolute_errors.append(error)
        # A normalized weighted objective deliberately permits a small DCBF
        # violation when exact lexicographic safety would stall the task. Raw
        # pixel-squared slack is divided by R^2 so one weight transfers across
        # obstacle sizes.
        normalized_slack = gp.quicksum(
            weight * slack / scale
            for slack, scale, weight in zip(
                safety_slacks, safety_slack_scales, safety_slack_weights))
        model.setObjective(
            float(self.config.tracking_cost_multiplier) * gp.quicksum(absolute_errors)
            + normalized_slack,
            GRB.MINIMIZE)
        model.optimize()

        diagnostics = {
            "status": self._status_name(model.Status),
            "solutions": int(model.SolCount),
            "runtime_sec": float(model.Runtime),
            "nodes": int(model.NodeCount),
            "binary_variables": int(model.NumBinVars),
            "safety_slack": None,
        }
        if model.SolCount > 0:
            diagnostics["safety_slack"] = float(sum(
                variable.X for variable in safety_slacks))
            optimized_norm = np.asarray([variable.X for variable in action_vars])
            optimized = optimized_norm * self.action_range + self.action_min
            return optimized, diagnostics
        return None, diagnostics

    def _solve_horizon(self, action_refs, state_norm, previous_action_norm,
                       action_anchor_norm, obstacle_positions, obstacle_radii,
                       measured_agent_norm=None):
        """Jointly optimize all controls and NNDM states in one MILP."""
        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", int(self.config.verbose))
        environment.start()
        model = gp.Model("mind_joint_mpc", env=environment)
        model.Params.OutputFlag = int(self.config.verbose)
        model.Params.MIPGap = float(self.config.mip_gap)
        model.Params.TimeLimit = float(self.config.time_limit_sec)
        model.Params.NodeLimit = int(self.config.node_limit)
        model.Params.Threads = 1

        references = np.clip(
            (action_refs - self.action_min) / self.action_range, 0.0, 1.0)
        initial_safe_pairs = np.asarray([
            [
                np.linalg.norm(
                    self._center_raw(state_norm, indices) - obstacle)
                >= (float(obstacle_radius) * self.config.clearance_scale
                    + float(proxy_radius))
                for obstacle, obstacle_radius in zip(
                    obstacle_positions, obstacle_radii)
            ]
            for indices, proxy_radius in zip(
                self.config.safety_keypoint_groups,
                self.config.safety_proxy_radii)
        ], dtype=np.bool_)
        nominal_states = [np.asarray(state_norm, dtype=np.float64).copy()]
        nominal_previous = np.asarray(
            previous_action_norm, dtype=np.float64).copy()
        nominal_agent = measured_agent_norm
        for reference in references:
            nominal_states.append(self._predict_next_state_norm(
                nominal_states[-1], nominal_previous, reference, nominal_agent))
            if self.weights[0].shape[1] == 24:
                nominal_agent = ((reference * self.action_range + self.action_min
                                  - self.agent_min) / self.agent_range)
            nominal_previous = reference
        radius = float(self.config.action_search_radius)
        state_values = list(state_norm)
        state_lower = state_norm.copy()
        state_upper = state_norm.copy()
        previous_values = list(previous_action_norm)
        previous_lower = previous_action_norm.copy()
        previous_upper = previous_action_norm.copy()
        all_action_vars = []
        absolute_errors = []
        safety_slacks = []
        safety_slack_scales = []
        safety_slack_weights = []

        for stage, reference in enumerate(references):
            action_lower = np.maximum(0.0, reference - radius)
            action_upper = np.minimum(1.0, reference + radius)
            delta_norm = (
                float(self.config.max_action_delta_px) / self.action_range)
            if stage == 0:
                action_lower = np.maximum(
                    action_lower, action_anchor_norm - delta_norm)
                action_upper = np.minimum(
                    action_upper, action_anchor_norm + delta_norm)
                movement_anchor = list(action_anchor_norm)
            else:
                action_lower = np.maximum(
                    action_lower, previous_lower - delta_norm)
                action_upper = np.minimum(
                    action_upper, previous_upper + delta_norm)
                movement_anchor = all_action_vars[-1]
            action_vars = []
            for action_index in range(2):
                variable = model.addVar(
                    lb=float(action_lower[action_index]),
                    ub=float(action_upper[action_index]),
                    name=f"h{stage}_u_{action_index}")
                variable.Start = float(reference[action_index])
                action_vars.append(variable)
                error = model.addVar(
                    lb=0.0, name=f"h{stage}_reference_error_{action_index}")
                model.addConstr(
                    error >= variable - float(reference[action_index]))
                model.addConstr(
                    error >= float(reference[action_index]) - variable)
                model.addConstr(
                    variable - movement_anchor[action_index]
                    <= float(delta_norm[action_index]),
                    name=f"h{stage}_rate_upper_{action_index}")
                model.addConstr(
                    movement_anchor[action_index] - variable
                    <= float(delta_norm[action_index]),
                    name=f"h{stage}_rate_lower_{action_index}")
                absolute_errors.append(error)
            all_action_vars.append(action_vars)

            next_values, next_lower, next_upper = \
                self._encode_network_stage(
                    model, state_values, state_lower, state_upper,
                    previous_values, previous_lower, previous_upper,
                    action_vars, action_lower, action_upper, stage,
                    measured_agent_norm)
            stage_slacks, stage_scales, stage_weights = \
                self._add_horizon_safety_constraints(
                model, state_values, state_lower, state_upper,
                next_values, next_lower, next_upper,
                nominal_states[stage], nominal_states[stage + 1],
                obstacle_positions, obstacle_radii,
                initial_safe_pairs,
                stage, terminal=(stage == len(references) - 1))
            safety_slacks.extend(stage_slacks)
            safety_slack_scales.extend(stage_scales)
            safety_slack_weights.extend(stage_weights)

            state_values = next_values
            state_lower = next_lower
            state_upper = next_upper
            previous_values = action_vars
            previous_lower = action_lower
            previous_upper = action_upper

        slack_objective = gp.quicksum(
            float(weight) * variable / float(scale)
            for variable, scale, weight in zip(
                safety_slacks, safety_slack_scales, safety_slack_weights))
        # Hierarchical optimization: safety violation is optimized first and
        # may never be traded for closer diffusion tracking. Only among
        # equally safe incumbents does the second objective minimize action
        # modification.
        model.ModelSense = GRB.MINIMIZE
        model.setObjectiveN(
            slack_objective, index=0, priority=2, weight=1.0,
            abstol=1.0e-6, reltol=0.0, name="safety_slack")
        model.setObjectiveN(
            gp.quicksum(absolute_errors), index=1, priority=1, weight=1.0,
            abstol=1.0e-6, reltol=0.0, name="diffusion_tracking")
        model.optimize()
        diagnostics = {
            "status": self._status_name(model.Status),
            "solutions": int(model.SolCount),
            "runtime_sec": float(model.Runtime),
            "nodes": int(model.NodeCount),
            "binary_variables": int(model.NumBinVars),
            "horizon": int(len(action_refs)),
            "safety_slack": None,
        }
        if model.SolCount > 0:
            diagnostics["safety_slack"] = float(sum(
                variable.X for variable in safety_slacks))
            optimized_norm = np.asarray([
                [variable.X for variable in stage_vars]
                for stage_vars in all_action_vars])
            optimized = optimized_norm * self.action_range + self.action_min
            return optimized, diagnostics
        return None, diagnostics

    def project_horizon(self, action_refs, state_raw, obstacle_positions,
                        obstacle_radii, stream_id=0, fallback_action=None):
        """Jointly solve and execute the complete MPC action chunk."""
        action_refs = np.asarray(action_refs, dtype=np.float64).reshape(-1, 2)
        if action_refs.shape[0] == 0:
            self._last_solver_status_sequence[stream_id] = ()
            return action_refs.copy(), ()
        if not self.config.enabled or obstacle_positions is None:
            return action_refs.copy(), ("DISABLED",) * len(action_refs)
        if gp is None:
            raise RuntimeError("MIND is enabled but gurobipy is not installed.")

        positions = np.asarray(
            obstacle_positions, dtype=np.float64).reshape(-1, 2)
        radii = np.asarray(obstacle_radii, dtype=np.float64).reshape(-1)
        if positions.shape[0] != radii.size:
            raise ValueError("Obstacle position/radius counts do not match.")
        state_raw = np.asarray(state_raw, dtype=np.float64).reshape(18)
        state_norm = (state_raw - self.keypoint_min) / self.keypoint_range
        references_norm = np.clip(
            (action_refs - self.action_min) / self.action_range, 0.0, 1.0)
        initial_previous_norm = references_norm[0]
        if self.weights[0].shape[1] == 24 and fallback_action is not None:
            initial_previous_norm = np.clip(
                (np.asarray(fallback_action, dtype=np.float64).reshape(2)
                 - self.action_min) / self.action_range, 0.0, 1.0)
        previous_norm = self._previous_action_norm.get(
            stream_id, initial_previous_norm).copy()
        if fallback_action is None:
            action_anchor = (
                previous_norm * self.action_range + self.action_min)
        else:
            action_anchor = np.asarray(
                fallback_action, dtype=np.float64).reshape(2)
        action_anchor_norm = np.clip(
            (action_anchor - self.action_min) / self.action_range, 0.0, 1.0)
        if self.weights[0].shape[1] == 24:
            if fallback_action is None:
                raise ValueError("24D MIND requires measured pusher position as fallback_action.")
            measured_agent_norm = ((action_anchor - self.agent_min) /
                                   self.agent_range)
        else:
            measured_agent_norm = None
        self._measured_agent_norm[stream_id] = measured_agent_norm
        result, diagnostics = self._solve_horizon(
            action_refs, state_norm, previous_norm, action_anchor_norm,
            positions, radii, measured_agent_norm)
        with self._diagnostics_lock:
            self._solve_count += 1
        status = diagnostics["status"]
        statuses = (status,) * len(action_refs)
        self._last_solver_status[stream_id] = status
        self._last_solver_status_sequence[stream_id] = statuses

        if result is not None:
            slack = float(diagnostics["safety_slack"])
            with self._diagnostics_lock:
                self._total_safety_slack += slack
                self._max_safety_slack = max(self._max_safety_slack, slack)
                if slack > 1.0e-7:
                    self._positive_slack_count += 1
            # Both controls are executed open-loop, so the following replan
            # must use the final control as the NNDM previous-action input.
            last_norm = np.clip(
                (result[-1] - self.action_min) / self.action_range, 0.0, 1.0)
            self._previous_action_norm[stream_id] = last_norm
            self._store_visualization_data(
                stream_id, state_norm, previous_norm,
                action_refs[0], result[0], action_anchor,
                positions, radii, diagnostics)
            return result, statuses

        if self.config.print_status:
            print(
                f"[MIND MPC] env={stream_id} {status} "
                f"horizon={len(action_refs)} "
                f"mip_time={diagnostics['runtime_sec']:.4f}s "
                "no_incumbent",
                flush=True)
        if self.config.fail_safe == "raise":
            raise RuntimeError(
                f"MIND joint MPC found no feasible incumbent: {diagnostics}")
        with self._diagnostics_lock:
            self._no_solution_count += 1
        statuses = (f"{status}_NO_SOLUTION",) * len(action_refs)
        self._last_solver_status_sequence[stream_id] = statuses
        # Match the established one-step fail-safe: an unsolved safety MIP
        # must not silently execute the unverified diffusion reference chunk.
        held = np.clip(action_anchor, self.action_min,
                       self.action_min + self.action_range)
        fallback = np.repeat(held[None, :], len(action_refs), axis=0)
        self._previous_action_norm[stream_id] = np.clip(
            (fallback[-1] - self.action_min) / self.action_range, 0.0, 1.0)
        return fallback, statuses

    def project(self, action_ref, state_raw, obstacle_positions, obstacle_radii,
                stream_id=0, fallback_action=None, action_anchor=None,
                safety_mode="strict"):
        action_ref = np.asarray(action_ref, dtype=np.float64).reshape(2)
        if not self.config.enabled or obstacle_positions is None:
            return action_ref.copy(), "disabled"
        if gp is None:
            raise RuntimeError("MIND is enabled but gurobipy is not installed.")
        positions = np.asarray(obstacle_positions, dtype=np.float64).reshape(-1, 2)
        radii = np.asarray(obstacle_radii, dtype=np.float64).reshape(-1)
        if positions.shape[0] != radii.size:
            raise ValueError("Obstacle position/radius counts do not match.")
        state_raw = np.asarray(state_raw, dtype=np.float64).reshape(18)
        state_norm = (state_raw - self.keypoint_min) / self.keypoint_range
        reference_norm = np.clip(
            (action_ref - self.action_min) / self.action_range, 0.0, 1.0)
        if self.weights[0].shape[1] == 24:
            if fallback_action is None:
                raise ValueError("24D MIND requires measured pusher position as fallback_action.")
            measured_agent_raw = np.asarray(fallback_action, dtype=np.float64).reshape(2)
            measured_agent_norm = ((measured_agent_raw - self.agent_min) /
                                   self.agent_range)
            initial_previous_norm = np.clip(
                (measured_agent_raw - self.action_min) / self.action_range,
                0.0, 1.0)
        else:
            measured_agent_norm = None
            initial_previous_norm = reference_norm
        self._measured_agent_norm[stream_id] = measured_agent_norm
        previous_norm = self._previous_action_norm.get(stream_id, initial_previous_norm)
        if action_anchor is None:
            if fallback_action is not None:
                action_anchor = fallback_action
            else:
                action_anchor = (
                    previous_norm * self.action_range + self.action_min)
        action_anchor = np.asarray(action_anchor, dtype=np.float64).reshape(2)
        result, diagnostics = self._solve(
            action_ref, state_norm, previous_norm, action_anchor,
            positions, radii, stream_id=stream_id, safety_mode=safety_mode)
        with self._diagnostics_lock:
            self._solve_count += 1
        self._last_solver_status[stream_id] = diagnostics["status"]
        if result is not None:
            slack = float(diagnostics["safety_slack"])
            with self._diagnostics_lock:
                self._total_safety_slack += slack
                self._max_safety_slack = max(self._max_safety_slack, slack)
                if slack > 1.0e-7:
                    self._positive_slack_count += 1
            self._previous_action_norm[stream_id] = np.clip(
                (result - self.action_min) / self.action_range, 0.0, 1.0)
            self._store_visualization_data(
                stream_id, state_norm, previous_norm, action_ref, result,
                action_anchor,
                positions, radii, diagnostics)
            # Successful-solve logging is temporarily disabled. Keep this
            # block for convenient re-enabling during solver diagnostics.
            # if self.config.print_status:
            #     print(
            #         f"[MIND] env={stream_id} {diagnostics['status']} "
            #         f"mip_time={diagnostics['runtime_sec']:.4f}s "
            #         f"u={np.round(action_ref, 2).tolist()}"
            #         f"->{np.round(result, 2).tolist()}",
            #         flush=True,
            #     )
            return result, diagnostics["status"].lower()

        with self._diagnostics_lock:
            self._no_solution_count += 1
        center = self._center_raw(state_norm)
        safety_details = []
        for obstacle, obstacle_radius in zip(positions, radii):
            safe_radius = (
                float(obstacle_radius) * self.config.clearance_scale
                + self.config.block_clearance_radius)
            distance = float(np.linalg.norm(center - obstacle))
            phi = safe_radius ** 2 - distance ** 2
            safety_details.append(
                f"safe_r={safe_radius:.1f} dist={distance:.1f} phi={phi:.1f}")
        print(
            f"[MIND] env={stream_id} NO_SOLUTION "
            f"status={diagnostics['status']} "
            f"solutions={diagnostics['solutions']} "
            f"mip_time={diagnostics['runtime_sec']:.4f}s "
            f"nodes={diagnostics['nodes']} "
            f"{' | '.join(safety_details)} "
            "fallback=agent_hold",
            flush=True,
        )
        message = f"MIND found no feasible incumbent: {diagnostics}"
        if self.config.fail_safe == "raise":
            raise RuntimeError(message)
        if fallback_action is None:
            raise RuntimeError(
                "mind.fail_safe=agent_hold requires the current agent position.")
        held = np.asarray(fallback_action, dtype=np.float64).reshape(2)
        held_norm = np.clip(
            (held - self.action_min) / self.action_range, 0.0, 1.0)
        self._previous_action_norm[stream_id] = held_norm
        self._store_visualization_data(
            stream_id, state_norm, previous_norm, action_ref, held,
            action_anchor,
            positions, radii, diagnostics)
        return held, f"{diagnostics['status'].lower()}_agent_hold"

    def project_sequence(self, action_refs, state_raw, obstacle_positions,
                         obstacle_radii, stream_id=0, fallback_action=None,
                         safety_modes=None):
        """Sequentially project an open-loop action chunk.

        Each action is solved by a separate exact one-step ReLU MIP. Between
        solves, the NNDM propagates the state using the projected action, so
        every executed action gets a safety constraint at its predicted state.
        In ordinary use, unsafe stages are ``nonworsening`` so the pusher can
        establish contact without predicting T motion farther inward. An
        optional terminal strict-recovery condition can be enabled in config.
        """
        action_refs = np.asarray(action_refs, dtype=np.float64).reshape(-1, 2)
        if action_refs.shape[0] == 0:
            self._last_solver_status_sequence[stream_id] = ()
            return action_refs.copy(), ()

        state_raw = np.asarray(state_raw, dtype=np.float64).reshape(18)
        if safety_modes is None:
            if (self.config.recovery_terminal_step_only
                    and len(action_refs) > 1):
                safety_modes = (("nonworsening",) * (len(action_refs) - 1)
                                + ("strict",))
            else:
                safety_modes = ("nonworsening",) * len(action_refs)
        else:
            safety_modes = tuple(safety_modes)
            if len(safety_modes) != len(action_refs):
                raise ValueError(
                    "MIND safety_modes must have one entry per action.")
            if any(mode not in {"strict", "nonworsening", "geometric_transfer",
                                "safe_only"}
                   for mode in safety_modes):
                raise ValueError(
                    "MIND safety_modes must be strict, nonworsening, safe_only, or geometric_transfer.")
        state_norm = (state_raw - self.keypoint_min) / self.keypoint_range
        projected_actions = []
        statuses = []
        first_visualization = None
        # The pusher state is not part of the 18-D NNDM output.  For stage 0,
        # holding means targeting the measured pusher position.  For later
        # stages, the best available hold target is the preceding projected
        # action, which is also the target expected to have just executed.
        stage_fallback = (
            None if fallback_action is None
            else np.asarray(fallback_action, dtype=np.float64).reshape(2))

        for stage, action_ref in enumerate(action_refs):
            reference_norm = np.clip(
                (action_ref - self.action_min) / self.action_range, 0.0, 1.0)
            previous_norm = self._previous_action_norm.get(
                stream_id, reference_norm).copy()
            if safety_modes[stage] == "geometric_transfer":
                projected = np.clip(np.asarray(action_ref, dtype=np.float64),
                                    self.action_min,
                                    self.action_min + self.action_range)
                if stage_fallback is not None:
                    delta = float(self.config.max_action_delta_px)
                    projected = np.clip(projected, stage_fallback - delta,
                                        stage_fallback + delta)
                status = "GEOMETRIC_TRANSFER"
            else:
                projected, status = self.project(
                    action_ref,
                    state_norm * self.keypoint_range + self.keypoint_min,
                    obstacle_positions,
                    obstacle_radii,
                    stream_id=stream_id,
                    fallback_action=stage_fallback,
                    action_anchor=stage_fallback,
                    safety_mode=safety_modes[stage],
                )
            if (self.config.hard_cbf_enabled
                    and status.endswith("_agent_hold")):
                # No certified action exists for one stage of this open-loop
                # chunk. Do not execute a successful prefix followed by a
                # fallback; stop the measured pusher for all four actions.
                if fallback_action is None:
                    raise RuntimeError(
                        "Hard-CBF hold requires the measured pusher position.")
                held = np.clip(
                    np.asarray(fallback_action, dtype=np.float64).reshape(2),
                    self.action_min, self.action_min + self.action_range)
                held_norm = np.clip(
                    (held - self.action_min) / self.action_range, 0.0, 1.0)
                self._previous_action_norm[stream_id] = held_norm
                statuses = (status,) * len(action_refs)
                self._last_solver_status_sequence[stream_id] = statuses
                self._last_solver_status[stream_id] = status
                self._last_visualization_data.pop(stream_id, None)
                if self.config.print_status:
                    print(
                        f"[MIND HARD CBF] env={stream_id} stage={stage} "
                        f"{status} fallback=hold_full_chunk",
                        flush=True,
                    )
                return np.repeat(held[None, :], len(action_refs), axis=0), statuses
            statuses.append(status)
            if stage == 0:
                first_visualization = self.last_visualization_data(stream_id)
            current_norm = np.clip(
                (projected - self.action_min) / self.action_range, 0.0, 1.0)
            if safety_modes[stage] != "geometric_transfer":
                state_norm = self._predict_next_state_norm(
                    state_norm, previous_norm, current_norm,
                    self._measured_agent_norm.get(stream_id))
            self._previous_action_norm[stream_id] = current_norm.copy()
            projected_actions.append(projected)
            stage_fallback = projected.copy()

        statuses = tuple(statuses)
        self._last_solver_status_sequence[stream_id] = statuses
        # The current video frame begins with action 0 of the four-action
        # chunk, so keep its NNDM prediction as the overlay diagnostic instead
        # of the final stage's prediction.
        if first_visualization is not None:
            self._last_visualization_data[stream_id] = first_visualization
        self._last_solver_status[stream_id] = (
            next((status for status in statuses
                  if status.endswith("_agent_hold")), statuses[-1]))
        return np.asarray(projected_actions), statuses


class ControlAffineMindSafetyFilter:
    """MIND safety filter backed by a fixed-state control-affine NNDM.

    This deliberately preserves the existing task-level behavior: each of the
    four diffusion actions is projected independently; proxy geometry,
    warning/recovery hysteresis, soft DCBF priority, action L1 tracking, and
    the 25-pixel rate bound are unchanged.  Only the transition relation is
    replaced.  At every stage, f(z) and G(z) are evaluated numerically at the
    current predicted augmented state, then the action projection is an LP.
    """

    def __init__(self, dynamics_model: nn.Module, normalization: Mapping,
                 config: MindConfig):
        if gp is None:
            raise RuntimeError("MIND is enabled but gurobipy is not installed.")
        if normalization.get("model_type") != "control_affine":
            raise ValueError("ControlAffineMindSafetyFilter needs affine normalization.")
        if config.objective_norm != "l1":
            raise ValueError("Only L1 action tracking is implemented.")
        if config.fail_safe not in {"raise", "agent_hold"}:
            raise ValueError("mind.fail_safe must be raise or agent_hold.")
        self.config = config
        self.state_mean = np.asarray(normalization["state_mean"], dtype=np.float64)
        self.state_std = np.asarray(normalization["state_std"], dtype=np.float64)
        self.keypoint_min = self.state_mean[:18]
        self.keypoint_range = self.state_std[:18]
        self.agent_mean = self.state_mean[18:]
        self.agent_std = self.state_std[18:]
        # PushT actions are absolute coordinates in the 512 x 512 canvas.
        self.action_min = np.zeros(2, dtype=np.float64)
        self.action_range = np.full(2, 512.0, dtype=np.float64)
        linears = [module for module in dynamics_model.backbone
                   if isinstance(module, nn.Linear)]
        if len(linears) != 2:
            raise ValueError("Control-affine backbone must have two Linear layers.")
        self.backbone_weights = [layer.weight.detach().cpu().numpy().astype(np.float64)
                                 for layer in linears]
        self.backbone_biases = [layer.bias.detach().cpu().numpy().astype(np.float64)
                                for layer in linears]
        self.drift_weight = dynamics_model.drift_head.weight.detach().cpu().numpy().astype(np.float64)
        self.drift_bias = dynamics_model.drift_head.bias.detach().cpu().numpy().astype(np.float64)
        self.control_weight = dynamics_model.control_head.weight.detach().cpu().numpy().astype(np.float64)
        self.control_bias = dynamics_model.control_head.bias.detach().cpu().numpy().astype(np.float64)
        if self.drift_weight.shape != (20, 96) or self.control_weight.shape != (40, 96):
            raise ValueError("Expected v4.0 affine 20->96->96->{20,40} model.")
        if len(config.safety_keypoint_groups) != len(config.safety_proxy_radii):
            raise ValueError("mind safety group/radius counts must match.")
        self.reset()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_diagnostics_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._diagnostics_lock = threading.Lock()

    def reset(self):
        self._early_recovery_active = {}
        self._last_solver_status = {}
        self._last_solver_status_sequence = {}
        self._last_visualization_data = {}
        self._solve_count = 0
        self._positive_slack_count = 0
        self._no_solution_count = 0
        self._total_safety_slack = 0.0
        self._max_safety_slack = 0.0
        self._diagnostics_lock = threading.Lock()

    def diagnostics_summary(self):
        with self._diagnostics_lock:
            return {
                "solve_count": int(self._solve_count),
                "positive_slack_count": int(self._positive_slack_count),
                "no_solution_count": int(self._no_solution_count),
                "mean_safety_slack": self._total_safety_slack / self._solve_count
                if self._solve_count else 0.0,
                "max_safety_slack": float(self._max_safety_slack),
            }

    def last_solver_status(self, stream_id=0):
        return self._last_solver_status.get(stream_id)

    def last_solver_status_sequence(self, stream_id=0):
        return self._last_solver_status_sequence.get(stream_id, ())

    def last_visualization_data(self, stream_id=0):
        value = self._last_visualization_data.get(stream_id)
        return None if value is None else dict(value)

    def _fg(self, z_norm):
        hidden = np.asarray(z_norm, dtype=np.float64)
        for weight, bias in zip(self.backbone_weights, self.backbone_biases):
            hidden = np.maximum(weight @ hidden + bias, 0.0)
        drift = self.drift_weight @ hidden + self.drift_bias
        gain = (self.control_weight @ hidden + self.control_bias).reshape(20, 2)
        return drift, gain

    def _predict_next_state_norm(self, z_norm, action_raw):
        drift, gain = self._fg(z_norm)
        p_raw = z_norm[18:] * self.agent_std + self.agent_mean
        v_norm = (np.asarray(action_raw, dtype=np.float64) - p_raw) / self.agent_std
        return z_norm + drift + gain @ v_norm

    def _proxy_raw(self, z_norm, indices):
        state_raw = z_norm[:18] * self.keypoint_range + self.keypoint_min
        return state_raw.reshape(9, 2)[np.asarray(indices, dtype=int)].mean(axis=0)

    def _next_proxy_expression(self, z_norm, drift, gain, action_vars, indices):
        p_raw = z_norm[18:] * self.agent_std + self.agent_mean
        # z_next = z + f - G p/std + G diag(1/std) u
        normalized_constant = z_norm + drift - gain @ (p_raw / self.agent_std)
        normalized_coefficients = gain / self.agent_std[None, :]
        indices = np.asarray(indices, dtype=int)
        expressions = []
        for coordinate in range(2):
            state_indices = 2 * indices + coordinate
            scale = self.keypoint_range[state_indices]
            constant = float(np.mean(
                self.keypoint_min[state_indices] + scale * normalized_constant[state_indices]))
            coeff = np.mean(scale[:, None] * normalized_coefficients[state_indices], axis=0)
            expression = gp.LinExpr(constant)
            expression += float(coeff[0]) * action_vars[0]
            expression += float(coeff[1]) * action_vars[1]
            expressions.append(expression)
        return expressions

    def _add_barrier_constraints(self, model, z_norm, drift, gain, action_vars,
                                 obstacle_positions, obstacle_radii, stream_id,
                                 safety_mode):
        slacks, scales, weights = [], [], []
        for obstacle_index, (obstacle, obstacle_radius) in enumerate(zip(
                obstacle_positions, obstacle_radii)):
            obstacle = np.asarray(obstacle, dtype=np.float64)
            for proxy_index, (indices, proxy_radius) in enumerate(zip(
                    self.config.safety_keypoint_groups, self.config.safety_proxy_radii)):
                current = self._proxy_raw(z_norm, indices)
                relative = current - obstacle
                distance_squared = float(relative @ relative)
                safe_radius = (float(obstacle_radius) * self.config.clearance_scale
                               + float(proxy_radius))
                warning_radius = safe_radius + float(self.config.recovery_trigger_margin_px)
                release_radius = safe_radius + float(self.config.recovery_release_margin_px)
                mode_key = (stream_id, obstacle_index, proxy_index)
                early = bool(self._early_recovery_active.get(mode_key, False))
                if early:
                    if distance_squared >= release_radius ** 2:
                        early = False
                elif distance_squared < warning_radius ** 2:
                    early = True
                self._early_recovery_active[mode_key] = early
                radius = warning_radius if early else safe_radius
                phi = float(radius ** 2 - distance_squared)
                if safety_mode == "safe_only" and phi > 0.0:
                    continue
                if phi <= 0.0:
                    rhs = -float(self.config.safe_alpha) * phi
                    weight = float(self.config.safe_slack_weight)
                elif safety_mode == "nonworsening":
                    rhs = 0.0
                    weight = float(self.config.safe_slack_weight)
                else:
                    rhs = -max(float(self.config.recovery_alpha) * phi,
                               float(self.config.gamma) * float(self.config.control_dt_sec))
                    weight = float(self.config.recovery_slack_weight)
                next_proxy = self._next_proxy_expression(
                    z_norm, drift, gain, action_vars, indices)
                gradient = -2.0 * relative
                lhs = (float(gradient[0]) * (next_proxy[0] - float(current[0]))
                       + float(gradient[1]) * (next_proxy[1] - float(current[1])))
                slack = model.addVar(lb=0.0, name=f"affine_slack_{obstacle_index}_{proxy_index}")
                model.addConstr(lhs <= rhs + slack,
                                name=f"affine_barrier_{obstacle_index}_{proxy_index}")
                slacks.append(slack)
                scales.append(max(radius ** 2, 1.0))
                weights.append(weight)
        return slacks, scales, weights

    def _solve(self, action_ref, z_norm, obstacle_positions, obstacle_radii,
               stream_id, action_anchor, safety_mode):
        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", int(self.config.verbose))
        environment.start()
        model = gp.Model("mind_control_affine", env=environment)
        model.Params.OutputFlag = int(self.config.verbose)
        model.Params.TimeLimit = float(self.config.time_limit_sec)
        model.Params.Threads = 2
        reference = np.clip(np.asarray(action_ref, dtype=np.float64), 0.0, 512.0)
        anchor = np.clip(np.asarray(action_anchor, dtype=np.float64), 0.0, 512.0)
        search = float(self.config.action_search_radius) * 512.0
        delta = float(self.config.max_action_delta_px)
        lower = np.maximum.reduce((np.zeros(2), reference - search, anchor - delta))
        upper = np.minimum.reduce((np.full(2, 512.0), reference + search, anchor + delta))
        action_vars = [model.addVar(lb=float(lower[i]), ub=float(upper[i]),
                                    name=f"u_{i}") for i in range(2)]
        errors = []
        for index, variable in enumerate(action_vars):
            error = model.addVar(lb=0.0, name=f"reference_error_{index}")
            model.addConstr(error >= variable - float(reference[index]))
            model.addConstr(error >= float(reference[index]) - variable)
            errors.append(error)
        drift, gain = self._fg(z_norm)
        slacks, scales, weights = self._add_barrier_constraints(
            model, z_norm, drift, gain, action_vars, obstacle_positions,
            obstacle_radii, stream_id, safety_mode)
        slack_objective = gp.quicksum(
            float(weight) * slack / float(scale)
            for slack, scale, weight in zip(slacks, scales, weights))
        model.ModelSense = GRB.MINIMIZE
        model.setObjectiveN(slack_objective, index=0, priority=2, weight=1.0,
                            abstol=1.0e-6, reltol=0.0, name="safety_slack")
        model.setObjectiveN(gp.quicksum(errors), index=1, priority=1, weight=1.0,
                            abstol=1.0e-6, reltol=0.0, name="diffusion_tracking")
        model.optimize()
        diagnostics = {
            "status": MindSafetyFilter._status_name(model.Status),
            "solutions": int(model.SolCount), "runtime_sec": float(model.Runtime),
            "nodes": int(model.NodeCount), "safety_slack": None,
        }
        if model.SolCount > 0:
            diagnostics["safety_slack"] = float(sum(slack.X for slack in slacks))
            return np.asarray([variable.X for variable in action_vars]), diagnostics
        return None, diagnostics

    def _store_visualization_data(self, stream_id, z_norm, action_ref, action_used,
                                  action_anchor, obstacle_positions, obstacle_radii,
                                  diagnostics):
        next_norm = self._predict_next_state_norm(z_norm, action_used)
        indices = self.config.safety_keypoint_groups[0]
        center_now = self._proxy_raw(z_norm, indices)
        center_next = self._proxy_raw(next_norm, indices)
        obstacle = np.asarray(obstacle_positions[0], dtype=np.float64)
        safe_radius = (float(obstacle_radii[0]) * self.config.clearance_scale
                       + float(self.config.safety_proxy_radii[0]))
        distance_now = float(np.linalg.norm(center_now - obstacle))
        distance_next = float(np.linalg.norm(center_next - obstacle))
        self._last_visualization_data[stream_id] = {
            "raw_action": np.asarray(action_ref).tolist(), "used_action": np.asarray(action_used).tolist(),
            "action_anchor": np.asarray(action_anchor).tolist(),
            "center_now": center_now.tolist(), "center_next": center_next.tolist(),
            "safe_radius": safe_radius, "distance_now": distance_now, "distance_next": distance_next,
            "phi_now": safe_radius ** 2 - distance_now ** 2,
            "phi_next": safe_radius ** 2 - distance_next ** 2,
            "action_l1_change": float(np.abs(np.asarray(action_used) - np.asarray(action_ref)).sum()),
            "action_delta_from_anchor_linf": float(np.abs(np.asarray(action_used) - np.asarray(action_anchor)).max()),
            "status": str(diagnostics["status"]), "solutions": int(diagnostics["solutions"]),
            "runtime_sec": float(diagnostics["runtime_sec"]), "nodes": int(diagnostics["nodes"]),
            "safety_slack": None if diagnostics["safety_slack"] is None else float(diagnostics["safety_slack"]),
        }

    def project_sequence(self, action_refs, state_raw, obstacle_positions,
                         obstacle_radii, stream_id=0, fallback_action=None,
                         safety_modes=None):
        action_refs = np.asarray(action_refs, dtype=np.float64).reshape(-1, 2)
        if action_refs.shape[0] == 0:
            self._last_solver_status_sequence[stream_id] = ()
            return action_refs.copy(), ()
        if obstacle_positions is None:
            return action_refs.copy(), ("DISABLED",) * len(action_refs)
        positions = np.asarray(obstacle_positions, dtype=np.float64).reshape(-1, 2)
        radii = np.asarray(obstacle_radii, dtype=np.float64).reshape(-1)
        if positions.shape[0] != radii.size:
            raise ValueError("Obstacle position/radius counts do not match.")
        if safety_modes is None:
            if self.config.recovery_terminal_step_only and len(action_refs) > 1:
                safety_modes = (("nonworsening",) * (len(action_refs) - 1) + ("strict",))
            else:
                safety_modes = ("nonworsening",) * len(action_refs)
        if len(safety_modes) != len(action_refs):
            raise ValueError("mind safety_modes must have one value per action.")
        state_raw = np.asarray(state_raw, dtype=np.float64).reshape(18)
        p_raw = (np.asarray(fallback_action, dtype=np.float64).reshape(2)
                 if fallback_action is not None else action_refs[0])
        z_norm = (np.concatenate((state_raw, p_raw)) - self.state_mean) / self.state_std
        projected_actions, statuses, first_visualization = [], [], None
        anchor = p_raw.copy()
        for stage, (reference, mode) in enumerate(zip(action_refs, safety_modes)):
            if mode == "geometric_transfer":
                projected = np.clip(reference, anchor - self.config.max_action_delta_px,
                                    anchor + self.config.max_action_delta_px)
                projected = np.clip(projected, 0.0, 512.0)
                diagnostics = {"status": "GEOMETRIC_TRANSFER", "solutions": 1,
                               "runtime_sec": 0.0, "nodes": 0, "safety_slack": 0.0}
                status = "GEOMETRIC_TRANSFER"
            else:
                projected, diagnostics = self._solve(reference, z_norm, positions, radii,
                                                      stream_id, anchor, mode)
                status = diagnostics["status"]
                with self._diagnostics_lock:
                    self._solve_count += 1
                if projected is None:
                    with self._diagnostics_lock:
                        self._no_solution_count += 1
                    projected = np.clip(reference, anchor - self.config.max_action_delta_px,
                                        anchor + self.config.max_action_delta_px)
                    projected = np.clip(projected, 0.0, 512.0)
                    status = f"{status}_AGENT_HOLD"
                else:
                    slack = float(diagnostics["safety_slack"])
                    with self._diagnostics_lock:
                        self._total_safety_slack += slack
                        self._max_safety_slack = max(self._max_safety_slack, slack)
                        if slack > 1.0e-7:
                            self._positive_slack_count += 1
            self._store_visualization_data(stream_id, z_norm, reference, projected,
                                           anchor, positions, radii, diagnostics)
            if stage == 0:
                first_visualization = self.last_visualization_data(stream_id)
            z_norm = self._predict_next_state_norm(z_norm, projected)
            anchor = projected.copy()
            projected_actions.append(projected)
            statuses.append(status)
        statuses = tuple(statuses)
        self._last_solver_status_sequence[stream_id] = statuses
        self._last_solver_status[stream_id] = next(
            (status for status in statuses if status.endswith("_AGENT_HOLD")), statuses[-1])
        if first_visualization is not None:
            self._last_visualization_data[stream_id] = first_visualization
        return np.asarray(projected_actions), statuses


def create_mind_safety_filter(dynamics_model: nn.Module, normalization: Mapping,
                              config: MindConfig):
    """Instantiate the matching legacy-ReLU or v4.0 affine MIND backend."""
    if normalization.get("model_type") == "control_affine":
        return ControlAffineMindSafetyFilter(dynamics_model, normalization, config)
    return MindSafetyFilter(dynamics_model, normalization, config)
