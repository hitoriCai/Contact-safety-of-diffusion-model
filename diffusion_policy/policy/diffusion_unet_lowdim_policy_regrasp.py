"""Inference-only MIND policy with a local-stall regrasp recovery mode.

Normal task actions still come from the trained diffusion checkpoint. When the
T has already made progress but then remains nearly stationary beside an
obstacle, this policy temporarily replaces the leading action targets with a
collision-avoiding pusher transfer around the outside of the T. The ordinary
four one-step MIND projections are applied after this transform.
"""

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence
import math

import numpy as np
import torch

from diffusion_policy.policy.diffusion_unet_lowdim_policy import (
    DiffusionUnetLowdimPolicy,
)


@dataclass(frozen=True)
class RegraspConfig:
    enabled: bool = True
    # A replan executes four environment actions. Six consecutive replans
    # therefore give a 24-action (2.4 s at 10 Hz) local-stall window.
    history_replans: int = 6
    center_stall_radius_px: float = 4.0
    maximum_local_center_extent_px: float = 24.0
    maximum_net_center_motion_px: float = 10.0
    maximum_goal_progress_px: float = 3.0
    maximum_motion_efficiency: float = 0.45
    minimum_prior_progress_px: float = 8.0
    minimum_prior_elapsed_sec: float = 5.0
    # A large upcoming diffusion motion usually means the agent is executing
    # a purposeful side-transfer, not that T is locally stuck.
    # This was the 2026-09-15 v3.0/legacy-NNDM evaluation setting: a
    # regrasp can be considered only after each of the six preceding
    # MIND-projected four-action chunks has a path length below 30 px.
    maximum_trigger_action_path_length_px: float = 30.0
    # Regrasp is triggered from the legacy averaged T center, not the three
    # MIND shape proxies. Keep its warning radius explicit instead of silently
    # inheriting the current MIND clearance radius (35 px).
    trigger_center_clearance_radius_px: float = 75.0
    obstacle_proximity_margin_px: float = 10.0
    contact_radius_px: float = 48.0
    transfer_radius_px: float = 88.0
    radial_step_px: float = 16.0
    arc_step_degrees: float = 18.0
    escape_push_distance_px: float = 32.0
    target_tolerance_px: float = 12.0
    waypoint_capture_radius_px: float = 22.0
    maximum_agent_center_distance_px: float = 120.0
    workspace_margin_px: float = 8.0
    maximum_settle_replans: int = 12
    cooldown_replans: int = 8
    # A completed regrasp deliberately leaves the pusher on a new contact
    # side.  Diffusion may then spend several replans changing its approach
    # angle before it touches the T again.  Do not interpret that physically
    # correct no-motion interval as another local minimum: require one full
    # leave -> recontact -> real-push cycle before re-arming regrasp.
    rearm_release_distance_px: float = 45.0
    rearm_contact_distance_px: float = 35.0
    rearm_min_push_distance_px: float = 8.0
    goal_center: Sequence[float] = (213.573593, 298.426407)

    @classmethod
    def from_mapping(cls, values: Optional[Mapping]) -> "RegraspConfig":
        return cls(**({} if values is None else dict(values)))


class RegraspDiffusionUnetLowdimPolicy(DiffusionUnetLowdimPolicy):
    """Diffusion + MIND with an explicit pusher-side transfer state machine."""

    def __init__(self, *args, regrasp=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.regrasp_config = RegraspConfig.from_mapping(regrasp)
        if self.regrasp_config.history_replans < 3:
            raise ValueError("regrasp.history_replans must be at least 3.")
        if self.regrasp_config.center_stall_radius_px <= 0:
            raise ValueError("regrasp.center_stall_radius_px must be positive.")
        if self.regrasp_config.contact_radius_px <= 0:
            raise ValueError("regrasp.contact_radius_px must be positive.")
        if (self.regrasp_config.transfer_radius_px
                <= self.regrasp_config.contact_radius_px):
            raise ValueError(
                "regrasp.transfer_radius_px must exceed contact_radius_px.")
        if self.regrasp_config.radial_step_px <= 0:
            raise ValueError("regrasp.radial_step_px must be positive.")
        if not 0 < self.regrasp_config.arc_step_degrees <= 45:
            raise ValueError(
                "regrasp.arc_step_degrees must be in (0, 45].")
        self._regrasp_center_history = {}
        self._regrasp_agent_history = {}
        self._regrasp_initial_center = {}
        self._regrasp_modes = {}
        self._regrasp_action_modes = {}
        self._regrasp_cooldowns = {}
        self._regrasp_rearm_states = {}
        self._regrasp_active_flags = np.zeros(0, dtype=np.bool_)
        print(
            "[REGRASP CONFIG] "
            f"history={self.regrasp_config.history_replans} "
            f"stall_radius={self.regrasp_config.center_stall_radius_px:g} "
            f"trigger_center_r={self.regrasp_config.trigger_center_clearance_radius_px:g} "
            f"contact_radius={self.regrasp_config.contact_radius_px:g} "
            f"transfer_radius={self.regrasp_config.transfer_radius_px:g}",
            flush=True,
        )

    def reset(self):
        super().reset()
        if hasattr(self, "_regrasp_center_history"):
            self._regrasp_center_history.clear()
            self._regrasp_agent_history.clear()
            self._regrasp_initial_center.clear()
            self._regrasp_modes.clear()
            self._regrasp_action_modes.clear()
            self._regrasp_cooldowns.clear()
            self._regrasp_rearm_states.clear()
            self._regrasp_active_flags = np.zeros(0, dtype=np.bool_)

    @staticmethod
    def _first_position(value):
        return np.asarray(value, dtype=np.float64).reshape(-1, 2)[0]

    @staticmethod
    def _first_radius(value):
        return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])

    def _obstacle_for_stream(self, obstacles, stream_id, batch_size):
        positions = obstacles["pos"]
        radii = obstacles["r"]
        if isinstance(positions, (list, tuple)):
            return (
                self._first_position(positions[stream_id]),
                self._first_radius(radii[stream_id]),
            )
        positions_array = np.asarray(positions, dtype=np.float64)
        radii_array = np.asarray(radii, dtype=np.float64)
        if positions_array.ndim == 3 and positions_array.shape[0] == batch_size:
            position = positions_array[stream_id, 0]
            radius = radii_array.reshape(batch_size, -1)[stream_id, 0]
        else:
            position = positions_array.reshape(-1, 2)[0]
            radius = radii_array.reshape(-1)[0]
        return np.asarray(position, dtype=np.float64), float(radius)

    def _update_regrasp_arming(self, stream_id, center, agent, points):
        """Return whether a new local-stall trigger is currently permitted.

        After an intentional regrasp, stationary T observations are expected
        while diffusion moves the pusher around to its next contact pose.
        Require the pusher to leave the T, return to it, and genuinely move it
        before accepting another stall trigger.  The distances are measured
        against all nine observed T keypoints rather than the T centre, so the
        condition follows the actual asymmetric shape.
        """
        state = self._regrasp_rearm_states.get(stream_id)
        if state is None:
            return True
        config = self.regrasp_config
        shape_distance = float(np.min(np.linalg.norm(points - agent, axis=1)))
        if not state["departed"]:
            if shape_distance >= float(config.rearm_release_distance_px):
                state["departed"] = True
                print(
                    f"[REGRASP] env={stream_id} REARM_WAIT reapproach "
                    f"shape_dist={shape_distance:.1f}", flush=True)
            return False
        contact_center = state["contact_center"]
        if contact_center is None:
            if shape_distance <= float(config.rearm_contact_distance_px):
                state["contact_center"] = center.copy()
                print(
                    f"[REGRASP] env={stream_id} REARM_CONTACT "
                    f"shape_dist={shape_distance:.1f}", flush=True)
            return False
        if shape_distance > float(config.rearm_release_distance_px):
            # The pusher only brushed past the T; wait for a new contact.
            state["contact_center"] = None
            return False
        push_distance = float(np.linalg.norm(center - contact_center))
        if push_distance < float(config.rearm_min_push_distance_px):
            return False
        del self._regrasp_rearm_states[stream_id]
        # Build the subsequent stall window from post-push data;
        # pre-regrasp stationary observations must not leak into it.
        self._regrasp_center_history[stream_id].clear()
        self._regrasp_agent_history[stream_id].clear()
        print(
            f"[REGRASP] env={stream_id} REARMED push={push_distance:.1f}",
            flush=True)
        return True

    def _disarm_regrasp_until_new_push(self, stream_id):
        self._regrasp_rearm_states[stream_id] = {
            "departed": False,
            "contact_center": None,
        }

    @staticmethod
    def _linear_waypoints(start, end, maximum_step):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        distance = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(distance / maximum_step)))
        return [
            start + (end - start) * (index / count)
            for index in range(1, count + 1)
        ]

    @staticmethod
    def _in_workspace(point, margin):
        return bool(np.all(point >= margin) and np.all(point <= 512.0 - margin))

    def _build_regrasp_mode(self, center, agent, points, obstacle, stream_id,
                            trigger_metrics):
        config = self.regrasp_config
        radial = center - obstacle
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm < 1.0e-6:
            radial = np.asarray([1.0, 0.0])
        else:
            radial = radial / radial_norm
        # Move to the diametrically opposite pusher side without using a chord
        # through the T. The outside radius contains all nine keypoints plus a
        # pusher buffer, so transfer waypoints cannot contact the T.
        current_relative = agent - center
        current_radius = float(np.linalg.norm(current_relative))
        if current_radius < 1.0e-6:
            current_unit = -radial
        else:
            current_unit = current_relative / current_radius
        transfer_radius = max(
            float(config.transfer_radius_px),
            float(np.max(np.linalg.norm(points - center, axis=1))) + 20.0)
        retreat = center + transfer_radius * current_unit
        margin = float(config.workspace_margin_px)
        if not self._in_workspace(retreat, margin):
            return None
        transfer_waypoints = self._linear_waypoints(
            agent, retreat, float(config.radial_step_px))

        start_angle = math.atan2(current_unit[1], current_unit[0])
        arc_step = math.radians(float(config.arc_step_degrees))
        candidates = []
        for angle_delta, side in ((math.pi, "left"), (-math.pi, "right")):
            arc_count = max(1, int(math.ceil(abs(angle_delta) / arc_step)))
            arc = []
            feasible = True
            for index in range(1, arc_count + 1):
                angle = start_angle + angle_delta * index / arc_count
                point = center + transfer_radius * np.asarray(
                    [math.cos(angle), math.sin(angle)])
                if not self._in_workspace(point, margin):
                    feasible = False
                    break
                arc.append(point)
            if not feasible:
                continue
            target_unit = -current_unit
            directional_support = float(np.max((points - center) @ target_unit))
            contact_radius = max(float(config.contact_radius_px),
                                 directional_support + 18.0)
            contact_target = center + contact_radius * target_unit
            if not self._in_workspace(contact_target, margin):
                continue
            clearance = min(float(np.linalg.norm(point - obstacle)) for point in arc)
            candidates.append((clearance, side, arc, contact_target,
                               contact_radius))
        if not candidates:
            print(f"[REGRASP] env={stream_id} SKIP no_safe_arc", flush=True)
            return None
        _, side, arc, contact_target, contact_radius = max(
            candidates, key=lambda item: item[0])
        transfer_waypoints.extend(arc)
        approach_waypoints = self._linear_waypoints(
            transfer_waypoints[-1], contact_target,
            float(config.radial_step_px))
        # After reaching the mirror contact, push a short distance back toward
        # the old pusher side before returning control to diffusion. This
        # deliberately breaks the zero-motion equilibrium instead of merely
        # changing sides and immediately requesting the same blocked plan.
        escape_target = contact_target + (
            float(config.escape_push_distance_px) * current_unit)
        escape_waypoints = self._linear_waypoints(
            contact_target, escape_target, float(config.radial_step_px))
        # Free-space phases are geometrically outside the T. Only escape is a
        # true contact/push phase and therefore remains MIP-constrained.
        waypoints = [
            (point, "geometric_transfer", "transfer")
            for point in transfer_waypoints
        ] + [
            (point, "geometric_transfer", "approach")
            for point in approach_waypoints
        ] + [
            (point, "nonworsening", "escape")
            for point in escape_waypoints
        ]
        print(
            f"[REGRASP] env={stream_id} TRIGGER side={side} "
            f"center={np.round(center, 1).tolist()} "
            f"agent={np.round(agent, 1).tolist()} "
            f"mirror={np.round(contact_target, 1).tolist()} "
            f"escape={np.round(escape_target, 1).tolist()} "
            f"contact_r={contact_radius:.1f} "
            f"waypoints={len(waypoints)} "
            f"transfer={len(transfer_waypoints)} "
            f"approach={len(approach_waypoints)} "
            f"escape_steps={len(escape_waypoints)}",
            flush=True,
        )
        return {
            "queue": deque(waypoints),
            "target": np.asarray(escape_target, dtype=np.float64),
            "anchor_center": center.copy(),
            "side": side,
            "settle_replans": 0,
            "last_distance": float("inf"),
        }

    def _trigger_metrics(self, stream_id, center, agent, obstacle,
                         obstacle_radius, elapsed_sec=None,
                         action_path_length=None,
                         allow_oscillation_path=False):
        config = self.regrasp_config
        centers = self._regrasp_center_history[stream_id]
        agents = self._regrasp_agent_history[stream_id]
        if len(centers) < int(config.history_replans):
            return None
        progress = float(np.linalg.norm(
            center - self._regrasp_initial_center[stream_id]))
        moved_enough = progress >= float(config.minimum_prior_progress_px)
        waited_long_enough = bool(
            progress < float(config.minimum_prior_progress_px)
            and elapsed_sec is not None
            and float(elapsed_sec) > float(config.minimum_prior_elapsed_sec))
        if not (moved_enough or waited_long_enough):
            return None
        center_array = np.stack(tuple(centers))
        center_mean = center_array.mean(axis=0)
        center_extent = max(float(np.linalg.norm(item - center_mean))
                            for item in center_array)
        net_motion = float(np.linalg.norm(center_array[-1] - center_array[0]))
        path_length = float(sum(np.linalg.norm(
            center_array[index + 1] - center_array[index])
            for index in range(len(center_array) - 1)))
        goal = np.asarray(config.goal_center, dtype=np.float64)
        goal_progress = float(np.linalg.norm(center_array[0] - goal)
                              - np.linalg.norm(center_array[-1] - goal))
        # Curved, purposeful motion can have a low net/path ratio, so use
        # only actual low center displacement as the local-stall signal.
        if not (goal_progress <= float(config.maximum_goal_progress_px)
                and net_motion <= float(config.maximum_net_center_motion_px)):
            return None
        # Model-2 recovery uses the previously executed, MIND-projected main
        # action chunk. Do not guess from raw diffusion actions before one is
        # available.
        if action_path_length is None and not allow_oscillation_path:
            return None
        if (action_path_length is not None and float(action_path_length)
                >= float(config.maximum_trigger_action_path_length_px)):
            return None
        return {"center_extent": center_extent, "net_motion": net_motion,
                "path_length": path_length,
                "action_path_length": action_path_length,
                "goal_progress": goal_progress}

    def _apply_mode(self, plan, stream_id, center, agent, action_indices):
        config = self.regrasp_config
        mode = self._regrasp_modes[stream_id]
        if (mode["queue"] and mode["queue"][0][2] != "escape"
                and float(np.linalg.norm(center - mode["anchor_center"])) > 12.0):
            print(f"[REGRASP] env={stream_id} ABORT t_moved_during_transfer", flush=True)
            del self._regrasp_modes[stream_id]
            self._regrasp_cooldowns[stream_id] = int(config.cooldown_replans)
            return plan
        queue = mode["queue"]
        previous_count = len(queue)
        if queue:
            lookahead = list(queue)[:len(action_indices) + 2]
            distances = [float(np.linalg.norm(agent - item[0]))
                         for item in lookahead]
            nearest = int(np.argmin(distances))
            if distances[nearest] <= float(config.waypoint_capture_radius_px):
                for _ in range(nearest + 1):
                    queue.popleft()
                mode["settle_replans"] = 0
        if queue:
            distance = float(np.linalg.norm(agent - queue[0][0]))
        else:
            distance = float(np.linalg.norm(agent - mode["target"]))
        if len(queue) < previous_count or distance + 2.0 < mode["last_distance"]:
            mode["settle_replans"] = 0
        else:
            mode["settle_replans"] += 1
        mode["last_distance"] = distance
        if mode["settle_replans"] > int(config.maximum_settle_replans):
            print(f"[REGRASP] env={stream_id} ABORT path_stalled", flush=True)
            del self._regrasp_modes[stream_id]
            self._regrasp_cooldowns[stream_id] = int(config.cooldown_replans)
            return plan
        if not queue:
            if float(np.linalg.norm(agent - mode["target"])) \
                    <= float(config.target_tolerance_px):
                print(
                    f"[REGRASP] env={stream_id} COMPLETE "
                    f"side={mode['side']} agent={np.round(agent, 1).tolist()}",
                    flush=True,
                )
                del self._regrasp_modes[stream_id]
                self._regrasp_cooldowns[stream_id] = int(
                    config.cooldown_replans)
                self._disarm_regrasp_until_new_push(stream_id)
                self._regrasp_center_history[stream_id].clear()
                self._regrasp_agent_history[stream_id].clear()
                self._regrasp_action_modes.pop(stream_id, None)
                if self.mind_filter is not None:
                    self.mind_filter._previous_action_norm[stream_id] = np.clip(
                        (agent - self.mind_filter.action_min)
                        / self.mind_filter.action_range, 0.0, 1.0)
                return plan
        last_target = mode["target"]
        safety_modes = []
        catch_up = bool(queue) and distance > float(config.waypoint_capture_radius_px)
        for offset, action_index in enumerate(action_indices):
            if queue:
                waypoint_index = min(len(queue) - 1, 0 if catch_up else offset)
                last_target, safety_mode, _ = list(queue)[waypoint_index]
            else:
                safety_mode = "nonworsening"
            plan[action_index, :2] = torch.as_tensor(
                last_target, device=plan.device, dtype=plan.dtype)
            safety_modes.append(safety_mode)
        self._regrasp_action_modes[stream_id] = tuple(safety_modes)
        return plan

    def mind_safety_modes(self, batch_index, action_start, action_end):
        """Return phase modes under the configured recovery timing.

        Outer transfer is geometrically outside the T and bypasses the NNDM;
        the contact/escape phase remains an ordinary nonworsening MIP stage.
        """
        modes = self._regrasp_action_modes.get(batch_index)
        if modes is None or len(modes) != action_end - action_start:
            return None
        return modes

    def transform_diffusion_action_plan(
            self, action_pred, obs_dict, obstacles=None):
        if not self.regrasp_config.enabled or obstacles is None:
            self._regrasp_active_flags = np.zeros(
                action_pred.shape[0], dtype=np.bool_)
            return action_pred
        result = action_pred.clone()
        raw_obs = obs_dict["obs"][:, -1].detach().cpu().numpy()
        batch_size = raw_obs.shape[0]
        self._regrasp_active_flags = np.zeros(batch_size, dtype=np.bool_)
        action_start = self.n_obs_steps - 1 if self.oa_step_convention \
            else self.n_obs_steps
        action_end = min(
            action_start + int(self.mind_config.execution_action_steps),
            result.shape[1])
        action_indices = tuple(range(action_start, action_end))
        center_indices = np.asarray(
            self.mind_config.center_keypoint_indices, dtype=int)

        for stream_id in range(batch_size):
            self._regrasp_action_modes.pop(stream_id, None)
            points = raw_obs[stream_id, :18].reshape(9, 2)
            center = points[center_indices].mean(axis=0).astype(np.float64)
            agent = raw_obs[stream_id, 18:20].astype(np.float64)
            obstacle, obstacle_radius = self._obstacle_for_stream(
                obstacles, stream_id, batch_size)
            if stream_id not in self._regrasp_initial_center:
                self._regrasp_initial_center[stream_id] = center.copy()
            centers = self._regrasp_center_history.setdefault(
                stream_id, deque(maxlen=int(
                    self.regrasp_config.history_replans)))
            agents = self._regrasp_agent_history.setdefault(
                stream_id, deque(maxlen=int(
                    self.regrasp_config.history_replans)))
            centers.append(center.copy())
            agents.append(agent.copy())

            regrasp_armed = self._update_regrasp_arming(
                stream_id, center, agent, points)

            cooldown = int(self._regrasp_cooldowns.get(stream_id, 0))
            if cooldown > 0:
                self._regrasp_cooldowns[stream_id] = cooldown - 1
            trigger_metrics = None
            if (stream_id not in self._regrasp_modes and cooldown == 0
                    and regrasp_armed):
                trigger_metrics = self._trigger_metrics(
                    stream_id, center, agent, obstacle, obstacle_radius)
            if trigger_metrics is not None:
                mode = self._build_regrasp_mode(
                    center, agent, points, obstacle, stream_id,
                    trigger_metrics)
                if mode is not None:
                    self._regrasp_modes[stream_id] = mode
                else:
                    self._regrasp_cooldowns[stream_id] = int(
                        self.regrasp_config.cooldown_replans)
            if stream_id in self._regrasp_modes:
                result[stream_id] = self._apply_mode(
                    result[stream_id], stream_id, center, agent, action_indices)
            self._regrasp_active_flags[stream_id] = (
                stream_id in self._regrasp_modes)
        return result

    def predict_action(self, obs_dict, obstacles=None):
        result = super().predict_action(obs_dict, obstacles)
        flags = self._regrasp_active_flags
        if len(flags) != result['action'].shape[0]:
            flags = np.zeros(result['action'].shape[0], dtype=np.bool_)
        result['regrasp_active'] = torch.as_tensor(
            flags, device=result['action'].device, dtype=torch.bool)
        return result
