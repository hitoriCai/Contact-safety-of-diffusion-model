"""Expose physical action targets on [0,512]^2 without changing the NNDM.

The checkpoint's training normalization is folded algebraically into the
first-layer action columns. All existing MIND operations can then use page
coordinates consistently, including previous actions, ReLU bounds, safety
checks and visualization. The checkpoint and diffusion normalizers stay intact.
"""

import copy
from dataclasses import replace

import numpy as np

from diffusion_policy.policy.mind import GRB, MindSafetyFilter, gp
from experiments.model2_cbf_global_then_clip import filter as global_filter


PAGE_MIN = np.asarray([0.0, 0.0])
PAGE_MAX = np.asarray([512.0, 512.0])
_original_init = MindSafetyFilter.__init__
_installed = False


def _init_page_coordinates(self, dynamics_model, normalization, config):
    _original_init(self, dynamics_model, normalization, config)
    self.training_action_min = self.action_min.copy()
    self.training_action_range = self.action_range.copy()
    self.action_min = PAGE_MIN.copy()
    self.action_range = PAGE_MAX - PAGE_MIN

    # z_train = scale * z_page + offset, for both previous/current action.
    # For 24D the measured pusher occupies columns 18:20; it has its own
    # normalization and must not be folded into the action transform.
    scale = self.action_range / self.training_action_range
    offset = ((self.action_min - self.training_action_min)
              / self.training_action_range)
    first_weight = self.weights[0].copy()
    previous_start = self.weights[0].shape[1] - 4
    current_start = self.weights[0].shape[1] - 2
    self.biases[0] = (self.biases[0]
                      + first_weight[:, previous_start:previous_start + 2] @ offset
                      + first_weight[:, current_start:current_start + 2] @ offset)
    self.weights[0][:, previous_start:previous_start + 2] = (
        first_weight[:, previous_start:previous_start + 2] * scale)
    self.weights[0][:, current_start:current_start + 2] = (
        first_weight[:, current_start:current_start + 2] * scale)

    # Use the same physical tracking normalization as before, then apply the
    # configurable preference for following the diffusion target.
    self.action_tracking_weights = (
        scale * float(self.config.tracking_cost_multiplier))


def _solve_page_coordinates(self, action_ref, state_norm,
                             previous_action_norm, action_anchor,
                             obstacle_positions, obstacle_radii, stream_id=0,
                             safety_mode="strict"):
    """Existing one-step MIP in page coordinates, with the original cost."""
    environment = gp.Env(empty=True)
    environment.setParam("OutputFlag", int(self.config.verbose))
    environment.start()
    model = gp.Model("mind_full_page", env=environment)
    try:
        model.Params.OutputFlag = int(self.config.verbose)
        model.Params.MIPGap = float(self.config.mip_gap)
        model.Params.TimeLimit = float(self.config.time_limit_sec)
        model.Params.NodeLimit = int(self.config.node_limit)
        model.Params.Threads = 2

        reference = np.clip(
            (action_ref - self.action_min) / self.action_range, 0.0, 1.0)
        anchor = np.clip(
            (action_anchor - self.action_min) / self.action_range, 0.0, 1.0)
        delta = self.config.max_action_delta_px / self.action_range
        radius = float(self.config.action_search_radius)
        action_vars = []
        for index in range(2):
            lower = max(0.0, reference[index] - radius,
                        anchor[index] - delta[index])
            upper = min(1.0, reference[index] + radius,
                        anchor[index] + delta[index])
            if lower > upper:
                raise RuntimeError("Page action bounds and rate limits do not intersect")
            variable = model.addVar(lb=float(lower), ub=float(upper),
                                    name=f"u_{index}")
            variable.Start = float(np.clip(reference[index], lower, upper))
            action_vars.append(variable)

        # Use the executable box for ReLU interval bounds as well. Keeping
        # whole-page bounds here would add unnecessary binary variables.
        encoder = copy.copy(self)
        encoder.config = replace(
            self.config, action_search_radius=float(np.max(delta)))
        delta_state, _ = encoder._encode_network(
            model, state_norm, previous_action_norm, action_vars, anchor,
            self._measured_agent_norm.get(stream_id))
        slacks, scales, weights = self._add_safety_constraints(
            model, delta_state, state_norm, obstacle_positions, obstacle_radii,
            stream_id=stream_id, safety_mode=safety_mode)
        tracking_errors = []
        for index, variable in enumerate(action_vars):
            error = model.addVar(lb=0.0, name=f"reference_error_{index}")
            model.addConstr(error >= variable - float(reference[index]))
            model.addConstr(error >= float(reference[index]) - variable)
            tracking_errors.append(float(self.action_tracking_weights[index]) * error)
        normalized_slack = gp.quicksum(
            float(weight) * slack / float(scale)
            for slack, scale, weight in zip(slacks, scales, weights))
        model.setObjective(gp.quicksum(tracking_errors) + normalized_slack,
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
        if model.SolCount == 0:
            return None, diagnostics
        diagnostics["safety_slack"] = float(sum(slack.X for slack in slacks))
        optimized = np.asarray([variable.X for variable in action_vars])
        return optimized * self.action_range + self.action_min, diagnostics
    finally:
        model.dispose()
        environment.dispose()


def install(local_only=False):
    """Install only in the process started through this experiment's eval."""
    global _installed
    if _installed:
        return
    MindSafetyFilter.__init__ = _init_page_coordinates
    if local_only:
        # Keep the trained dynamics in page coordinates, but include the
        # executable movement box in the MIP itself. No global solve/clip.
        MindSafetyFilter._solve = _solve_page_coordinates
    else:
        global_filter._original_solve = _solve_page_coordinates
        global_filter.install()
    _installed = True
