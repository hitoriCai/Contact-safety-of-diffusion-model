"""Search the full action domain, then certify the rate-limited command.

This module is installed only by this experiment's eval entry point. It does
not change the shared MIND source used by ongoing evaluations.
"""

import copy
from dataclasses import replace

import numpy as np

from diffusion_policy.policy.mind import MindSafetyFilter


_original_solve = MindSafetyFilter._solve
_original_summary = MindSafetyFilter.diagnostics_summary


def _record(filter_, field):
    with filter_._diagnostics_lock:
        counters = filter_.__dict__.setdefault("_global_rate_counters", {})
        counters[field] = counters.get(field, 0) + 1


def _clipped_action_is_cbf_safe(filter_, action, state_norm,
                                previous_action_norm, obstacle_positions,
                                obstacle_radii, stream_id, safety_mode):
    """Check the exact numerical NNDM and the same one-step CBF inequalities.

    The global MIP's safety certificate does not transfer through clipping.
    This check applies the model's actual ReLU forward pass to the command
    that would be executed, with zero safety slack.
    """
    action_norm = np.clip(
        (action - filter_.action_min) / filter_.action_range, 0.0, 1.0)
    next_state_norm = filter_._predict_next_state_norm(
        state_norm, previous_action_norm, action_norm)
    config = filter_.config
    tolerance = 1.0e-5
    for obstacle_index, (obstacle, obstacle_radius) in enumerate(
            zip(obstacle_positions, obstacle_radii)):
        for proxy_index, (indices, proxy_radius) in enumerate(zip(
                config.safety_keypoint_groups, config.safety_proxy_radii)):
            center = filter_._center_raw(state_norm, indices)
            next_center = filter_._center_raw(next_state_norm, indices)
            relative = center - obstacle
            safe_radius = (float(obstacle_radius) * config.clearance_scale
                           + float(proxy_radius))
            warning_radius = safe_radius + config.recovery_trigger_margin_px
            release_radius = safe_radius + config.recovery_release_margin_px
            distance_squared = float(relative @ relative)
            key = (stream_id, obstacle_index, proxy_index)
            early_recovery = bool(filter_._early_recovery_active.get(key, False))
            if early_recovery:
                if distance_squared >= release_radius ** 2:
                    early_recovery = False
            elif distance_squared < warning_radius ** 2:
                early_recovery = True
            barrier_radius = warning_radius if early_recovery else safe_radius
            phi = barrier_radius ** 2 - distance_squared
            if safety_mode == "safe_only" and phi > 0.0:
                continue
            if phi <= 0.0:
                rhs = -config.safe_alpha * phi
            elif safety_mode == "nonworsening":
                rhs = 0.0
            else:
                rhs = -max(config.recovery_alpha * phi,
                           config.gamma * config.control_dt_sec)
            lhs = float((-2.0 * relative) @ (next_center - center))
            if lhs > rhs + tolerance:
                return False
    return True


def _global_then_rate_solve(self, action_ref, state_norm,
                            previous_action_norm, action_anchor,
                            obstacle_positions, obstacle_radii, stream_id=0,
                            safety_mode="strict"):
    if not self.config.hard_cbf_enabled:
        raise RuntimeError("Global-then-clip requires hard CBF")
    if self.config.action_search_radius != 1.0:
        raise RuntimeError("Full-domain search requires action_search_radius=1")

    # A shallow filter copy has a private config, while its per-stream
    # recovery latch is shared with the live filter. No running source or
    # mutable config is changed, including during threaded model-1 calls.
    global_filter = copy.copy(self)
    global_filter.config = replace(
        self.config,
        max_action_delta_px=512.0)
    global_action, global_diag = _original_solve(
        global_filter, action_ref, state_norm, previous_action_norm,
        action_anchor, obstacle_positions, obstacle_radii,
        stream_id=stream_id, safety_mode=safety_mode)
    _record(self, "global_solves")
    if global_action is None:
        _record(self, "global_no_solution")
        return None, global_diag

    delta = float(self.config.max_action_delta_px)
    lower = np.maximum(self.action_min, action_anchor - delta)
    upper = np.minimum(self.action_min + self.action_range,
                       action_anchor + delta)
    command = np.clip(global_action, lower, upper)
    if np.allclose(command, global_action, rtol=0.0, atol=1.0e-8):
        _record(self, "global_solution_within_rate_limit")
        return command, {**global_diag, "status": "GLOBAL_WITHIN_RATE_LIMIT",
                         "global_action": global_action.tolist()}

    _record(self, "global_solution_clipped")
    if _clipped_action_is_cbf_safe(
            self, command, state_norm, previous_action_norm,
            obstacle_positions, obstacle_radii, stream_id, safety_mode):
        _record(self, "clipped_command_cbf_certified")
        return command, {**global_diag, "status": "GLOBAL_CLIPPED_CERTIFIED",
                         "global_action": global_action.tolist()}

    _record(self, "clipped_command_cbf_unsafe")
    # The clipped global optimum is unsafe. Search the executable 25-px box
    # with that optimum as guidance; execute only an independently certified
    # local result. A genuinely empty local safe set still causes agent hold.
    local_action, local_diag = _original_solve(
        self, command, state_norm, previous_action_norm,
        action_anchor, obstacle_positions, obstacle_radii,
        stream_id=stream_id, safety_mode=safety_mode)
    if local_action is not None:
        _record(self, "clipped_command_repaired_locally")
        return local_action, {
            **local_diag, "status": "GLOBAL_CLIP_UNSAFE_LOCAL_REPAIRED",
            "global_action": global_action.tolist(),
            "clipped_global_action": command.tolist(),
            "global_runtime_sec": global_diag["runtime_sec"],
        }
    _record(self, "global_feasible_but_no_rate_limited_safe_action")
    return None, {
        **local_diag,
        "status": "GLOBAL_FEASIBLE_POST_CLIP_UNSAFE_LOCAL_NO_SOLUTION",
        "global_action": global_action.tolist(),
        "clipped_global_action": command.tolist(),
        "global_runtime_sec": global_diag["runtime_sec"],
    }


def _summary_with_rate_diagnostics(self):
    result = _original_summary(self)
    with self._diagnostics_lock:
        result.update(self.__dict__.get("_global_rate_counters", {}).copy())
    return result


def install():
    """Patch only this evaluation process; child tasks keep their own code."""
    MindSafetyFilter._solve = _global_then_rate_solve
    MindSafetyFilter.diagnostics_summary = _summary_with_rate_diagnostics
