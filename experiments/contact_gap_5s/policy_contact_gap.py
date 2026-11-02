"""Local-stall recovery driven by the separately trained escape diffusion.

The ordinary task diffusion policy remains in control until the existing
regrasp stall detector fires.  Recovery then uses the obstacle-aware 23-D
diffusion model until the escape condition succeeds or the recovery reaches
its 160-action limit.  The transformed recovery plan still passes through
the existing MIND projection in ``DiffusionUnetLowdimPolicy.predict_action``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import dill
import hydra
import numpy as np
import torch

from diffusion_policy.policy.diffusion_unet_lowdim_policy_regrasp import (
    RegraspDiffusionUnetLowdimPolicy,
)
from diffusion_policy.policy.mind import MindSafetyFilter, load_dynamics_checkpoint


_LOCAL_KEYPOINTS = np.asarray([
    [9.269786713763871, 90.04097352393619],
    [-60.983472364471474, -1.1867297607997276],
    [60.91275557727772, -1.3933642242328201],
    [-0.2705034390160016, 25.54362499146542],
    [-14.064927576212233, 122.81585355412153],
    [39.002738500170146, 32.857404276587566],
    [-39.966414424694534, 32.729655446429945],
    [-16.82434437939744, 62.86317892234989],
    [23.54146122024909, -1.7002881094626152],
], dtype=np.float64)
_LOCAL_CENTER = np.asarray([0.0, 60.0], dtype=np.float64)
_PAGE_CORNERS = np.asarray([
    [0.0, 0.0], [0.0, 512.0], [512.0, 0.0], [512.0, 512.0],
], dtype=np.float64)
_LOCAL_T_POLYGONS = (
    np.asarray([[-60.0, 30.0], [60.0, 30.0],
                [60.0, 0.0], [-60.0, 0.0]], dtype=np.float64),
    np.asarray([[-15.0, 30.0], [-15.0, 120.0],
                [15.0, 120.0], [15.0, 30.0]], dtype=np.float64),
)


@dataclass(frozen=True)
class Model2RecoveryConfig:
    checkpoint_path: str
    min_steps: int = 0
    max_steps: int = 160
    escape_clearance_px: float = 100.0
    escape_displacement_px: float = 45.0
    escape_min_corner_distance_px: float = 150.0
    checkpoint_state: str = "ema_model"
    # Kept under the legacy name for checkpoint/config compatibility.  When
    # enabled, recovery uses three T-region MIND proxies rather than the old
    # three-corner surrogate.
    corner_mind_enabled: bool = True
    corner_mind_radius_px: float = 10.0
    region_mind_proxy_radii: Sequence[float] = (25.0, 25.0, 40.0)
    # ``None`` means every newly armed, qualifying stall may launch model 2.
    # A positive value is retained only as an optional evaluation cap.
    max_activations_per_episode: Optional[int] = None
    # By default, preserve the learned escape command for points already
    # inside the small corner margin. Enable this to add MIND's ordinary
    # nonworsening recovery constraint during model-2 regrasp.
    corner_mind_recovery_enabled: bool = True
    # Additional trigger for a pusher that travels back and forth while the
    # T remains stuck. These thresholds were screened on the frozen central-50
    # trajectories; they still require a new closed-loop evaluation.
    oscillation_enabled: bool = False
    oscillation_min_episode_steps: int = 100
    oscillation_min_target_path_px: float = 180.0
    oscillation_max_target_net_px: float = 40.0
    oscillation_min_reversals: int = 6
    oscillation_min_direction_step_px: float = 2.0
    oscillation_reversal_cosine: float = -0.5
    oscillation_persistence_replans: int = 6
    # A bounded escape hatch for a recovery that ended but never completed the
    # ordinary leave -> recontact -> real-push rearm sequence.
    timeout_rearm_enabled: bool = False
    timeout_rearm_after_steps: int = 120
    timeout_rearm_max_shape_distance_px: float = 40.0
    timeout_rearm_max_retries: int = 1
    no_contact_trigger_seconds: float = 5.0

    @classmethod
    def from_mapping(cls, values: Optional[Mapping]):
        if values is None or not values.get("checkpoint_path"):
            raise ValueError(
                "regrasp_model2.checkpoint_path is required")
        return cls(**dict(values))


class Model2RegraspDiffusionPolicy(RegraspDiffusionUnetLowdimPolicy):
    """Original task diffusion, with model 2 replacing regrasp actions."""

    def __init__(self, *args, regrasp_model2=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.regrasp_model2_config = Model2RecoveryConfig.from_mapping(
            regrasp_model2)
        cfg = self.regrasp_model2_config
        if not 0 <= cfg.min_steps <= cfg.max_steps:
            raise ValueError(
                "regrasp_model2 requires 0 <= min_steps <= max_steps")
        if cfg.escape_min_corner_distance_px < 0:
            raise ValueError(
                "escape_min_corner_distance_px must be nonnegative")
        if cfg.corner_mind_radius_px < 0:
            raise ValueError("corner_mind_radius_px must be nonnegative")
        region_radii = tuple(float(value)
                             for value in cfg.region_mind_proxy_radii)
        if len(region_radii) != 3 or any(value < 0.0 for value in region_radii):
            raise ValueError(
                "region_mind_proxy_radii must contain three nonnegative values")
        if (cfg.max_activations_per_episode is not None
                and cfg.max_activations_per_episode < 1):
            raise ValueError("max_activations_per_episode must be positive")
        if (cfg.oscillation_min_episode_steps < 0
                or cfg.oscillation_min_target_path_px <= 0
                or cfg.oscillation_max_target_net_px < 0
                or cfg.oscillation_min_reversals < 1
                or cfg.oscillation_min_direction_step_px <= 0
                or not -1.0 <= cfg.oscillation_reversal_cosine < 1.0
                or cfg.oscillation_persistence_replans < 1):
            raise ValueError("invalid model-2 oscillation trigger settings")
        if (cfg.timeout_rearm_after_steps < 0
                or cfg.timeout_rearm_max_shape_distance_px <= 0
                or cfg.timeout_rearm_max_retries < 0):
            raise ValueError("invalid model-2 timeout rearm settings")
        if cfg.no_contact_trigger_seconds <= 0:
            raise ValueError("no_contact_trigger_seconds must be positive")
        self._regrasp_region_proxy_radii = region_radii
        # Keep the independently loaded policy outside this module's registered
        # state_dict.  The main task checkpoint predates model 2 and must remain
        # strictly loadable.  It is loaded lazily only in the EMA policy that is
        # actually used for evaluation.
        object.__setattr__(self, "_model2_policy", None)
        self._model2_plans = {}
        self._model2_activation_counts = {}
        self._model2_replan_counts = {}
        self._main_mind_action_path_histories = {}
        self._main_mind_action_target_histories = {}
        self._oscillation_streaks = {}
        self._last_model2_exit_replans = {}
        self._timeout_rearm_retry_counts = {}
        self._regrasp_task_scores = {}
        self._no_contact_steps = {}
        self._regrasp_mind_guard_flags = np.zeros(0, dtype=np.bool_)
        self._regrasp_mind_filter = None
        if cfg.corner_mind_enabled and self.mind_filter is not None:
            # Model 2 changes only the nominal diffusion action.  Its MIND
            # uses the same three-region proxy construction as the main task,
            # but a smaller recovery-specific clearance margin.
            region_config = self._make_regrasp_mind_config()
            dynamics_model, normalization = load_dynamics_checkpoint(
                region_config.checkpoint_path)
            self._regrasp_mind_filter = MindSafetyFilter(
                dynamics_model, normalization, region_config)
        post_filter = (
            "THREE_REGION_MIND_"
            + "_".join(f"r{radius:g}" for radius in
                       self._regrasp_region_proxy_radii)
            if self._regrasp_mind_filter is not None else "REGRASP_MIND_OFF")
        print(
            "[REGRASP MODEL2 CONFIG] "
            f"checkpoint={cfg.checkpoint_path} "
            f"steps=[{cfg.min_steps},{cfg.max_steps}] "
            f"clearance={cfg.escape_clearance_px:g} "
            f"displacement={cfg.escape_displacement_px:g} "
            f"min_corner_distance>{cfg.escape_min_corner_distance_px:g} "
            f"action_delta_limit={self.mind_config.max_action_delta_px:g}px/axis "
            f"post_filter={post_filter}",
            flush=True,
        )

    def _make_regrasp_mind_config(self):
        """Build the recovery filter profile independently of its actions."""
        return replace(
            self.mind_config,
            block_clearance_radius=max(self._regrasp_region_proxy_radii),
            safety_proxy_radii=self._regrasp_region_proxy_radii,
        )

    def reset(self):
        super().reset()
        if hasattr(self, "_model2_plans"):
            self._model2_plans.clear()
        if hasattr(self, "_model2_activation_counts"):
            self._model2_activation_counts.clear()
        if hasattr(self, "_model2_replan_counts"):
            self._model2_replan_counts.clear()
        if hasattr(self, "_main_mind_action_path_histories"):
            self._main_mind_action_path_histories.clear()
            self._main_mind_action_target_histories.clear()
            self._oscillation_streaks.clear()
            self._last_model2_exit_replans.clear()
            self._timeout_rearm_retry_counts.clear()
        if hasattr(self, "_regrasp_task_scores"):
            self._regrasp_task_scores.clear()
            self._no_contact_steps.clear()
        self._regrasp_mind_guard_flags = np.zeros(0, dtype=np.bool_)
        if self._regrasp_mind_filter is not None:
            self._regrasp_mind_filter.reset()
        policy = getattr(self, "_model2_policy", None)
        if policy is not None:
            policy.reset()

    def _checkpoint_path(self):
        path = Path(self.regrasp_model2_config.checkpoint_path).expanduser()
        if not path.is_absolute() and not path.is_file():
            path = Path(__file__).resolve().parents[2] / path
        if not path.is_file():
            raise FileNotFoundError(
                f"diffusion model 2 checkpoint not found: {path}")
        return path

    def _get_model2_policy(self):
        policy = getattr(self, "_model2_policy", None)
        if policy is None:
            path = self._checkpoint_path()
            payload = torch.load(
                path.open("rb"), pickle_module=dill, map_location="cpu")
            policy = hydra.utils.instantiate(payload["cfg"].policy)
            state_name = self.regrasp_model2_config.checkpoint_state
            if state_name not in payload["state_dicts"]:
                if "model" not in payload["state_dicts"]:
                    raise KeyError(
                        f"model 2 checkpoint has no {state_name!r} or 'model' state")
                state_name = "model"
            policy.load_state_dict(payload["state_dicts"][state_name])
            policy.eval()
            for parameter in policy.parameters():
                parameter.requires_grad_(False)
            object.__setattr__(self, "_model2_policy", policy)
            print(
                f"[REGRASP MODEL2] loaded {state_name} from {path}",
                flush=True,
            )
        policy.to(device=self.device, dtype=self.dtype)
        policy.eval()
        return policy

    @staticmethod
    def _rigid_transform(points):
        source_mean = _LOCAL_KEYPOINTS.mean(axis=0)
        target_mean = points.mean(axis=0)
        covariance = (_LOCAL_KEYPOINTS - source_mean).T @ (
            points - target_mean)
        left, _, right_t = np.linalg.svd(covariance)
        rotation = right_t.T @ left.T
        if np.linalg.det(rotation) < 0:
            right_t[-1] *= -1
            rotation = right_t.T @ left.T
        translation = target_mean - source_mean @ rotation.T
        return rotation, translation

    @staticmethod
    def _point_polygon_distance(point, polygon):
        edges = np.roll(polygon, -1, axis=0) - polygon
        relative = point[None, :] - polygon
        cross = edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0]
        inside = bool(np.all(cross >= -1.0e-7) or np.all(cross <= 1.0e-7))
        if inside:
            return 0.0
        denominator = np.maximum(np.sum(edges * edges, axis=1), 1.0e-12)
        fraction = np.clip(np.sum(relative * edges, axis=1) / denominator,
                           0.0, 1.0)
        closest = polygon + fraction[:, None] * edges
        return float(np.min(np.linalg.norm(closest - point[None, :], axis=1)))

    @classmethod
    def _geometry(cls, points, obstacle, obstacle_radius):
        rotation, translation = cls._rigid_transform(points)
        center = _LOCAL_CENTER @ rotation.T + translation
        polygons = [local @ rotation.T + translation
                    for local in _LOCAL_T_POLYGONS]
        distance = min(cls._point_polygon_distance(obstacle, polygon)
                       for polygon in polygons)
        return center, float(distance - obstacle_radius)

    @staticmethod
    def _nearest_page_corner_distance(center):
        return float(np.min(np.linalg.norm(
            _PAGE_CORNERS - center[None, :], axis=1)))

    def _escape_succeeded(self, center, clearance, displacement):
        cfg = self.regrasp_model2_config
        return bool(
            clearance >= float(cfg.escape_clearance_px)
            and displacement >= float(cfg.escape_displacement_px)
            and self._nearest_page_corner_distance(center) > float(
                cfg.escape_min_corner_distance_px))

    def _finish_model2(self, stream_id, center, reason, clearance,
                       displacement):
        mode = self._regrasp_modes.pop(stream_id)
        print(
            f"[REGRASP MODEL2] env={stream_id} EXIT reason={reason} "
            f"steps={mode['executed_steps']} clearance={clearance:.1f} "
            f"displacement={displacement:.1f} "
            f"nearest_corner={self._nearest_page_corner_distance(center):.1f}",
            flush=True,
        )
        if (self._regrasp_mind_filter is not None
                and stream_id in self._regrasp_mind_filter._previous_action_norm):
            self.mind_filter._previous_action_norm[stream_id] = (
                self._regrasp_mind_filter._previous_action_norm[
                    stream_id].copy())
        self._regrasp_cooldowns[stream_id] = int(
            self.regrasp_config.cooldown_replans)
        # This replan is the first one after the last executed model-2 chunk.
        self._last_model2_exit_replans[stream_id] = (
            self._model2_replan_counts[stream_id] - 1)
        self._disarm_regrasp_until_new_push(stream_id)
        self._regrasp_center_history[stream_id].clear()
        self._regrasp_agent_history[stream_id].clear()
        self._regrasp_action_modes.pop(stream_id, None)
        # Require one fresh, actual main-task MIND chunk before a later
        # regrasp can use an action-length trigger again.
        self._main_mind_action_path_histories.pop(stream_id, None)
        self._main_mind_action_target_histories.pop(stream_id, None)
        self._oscillation_streaks.pop(stream_id, None)

    def set_task_scores(self, scores):
        """Receive each environment's current maximum task score from runner."""
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        self._regrasp_task_scores = {
            stream_id: float(score) for stream_id, score in enumerate(values)}

    def set_no_contact_steps(self, steps):
        """Receive consecutive physical pusher-to-T no-contact control steps."""
        values = np.asarray(steps, dtype=np.int64).reshape(-1)
        self._no_contact_steps = {
            stream_id: int(count) for stream_id, count in enumerate(values)}

    def _start_model2(self, stream_id, center, clearance, metrics):
        self._regrasp_modes[stream_id] = {
            "initial_center": center.copy(),
            "initial_clearance": float(clearance),
            "executed_steps": 0,
        }
        print(
            f"[REGRASP MODEL2] env={stream_id} ENTER "
            f"trigger={metrics['trigger_kind']} "
            f"clearance={clearance:.1f} "
            f"stall_net={metrics['net_motion']:.1f} "
            f"goal_progress={metrics['goal_progress']:.1f} "
            f"no_contact_steps={self._no_contact_steps.get(stream_id, 0)}",
            flush=True,
        )
        self._oscillation_streaks.pop(stream_id, None)

    def _oscillation_metrics(self, stream_id, replan_count, points, agent):
        """Screen the previous six executed MIND action chunks, not raw plans."""
        cfg = self.regrasp_model2_config
        if not cfg.oscillation_enabled:
            return None
        episode_steps = int(replan_count) * int(
            self.mind_config.execution_action_steps)
        if episode_steps < int(cfg.oscillation_min_episode_steps):
            return None
        shape_distance = float(np.min(np.linalg.norm(
            points - agent[None, :], axis=1)))
        history = self._main_mind_action_target_histories.get(stream_id)
        if history is None or len(history) < int(
                self.regrasp_config.history_replans):
            return None
        expected_chunk_steps = int(self.mind_config.execution_action_steps)
        if any(chunk.shape != (expected_chunk_steps, 2)
               for chunk in history):
            return None
        targets = np.concatenate(tuple(history), axis=0)
        vectors = np.diff(targets, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        path_length = float(lengths.sum())
        net_motion = float(np.linalg.norm(targets[-1] - targets[0]))
        pair_lengths = lengths[:-1] * lengths[1:]
        cosines = np.sum(vectors[:-1] * vectors[1:], axis=1) / np.maximum(
            pair_lengths, 1.0e-12)
        reversals = int(np.count_nonzero(
            (lengths[:-1] >= float(cfg.oscillation_min_direction_step_px))
            & (lengths[1:] >= float(cfg.oscillation_min_direction_step_px))
            & (cosines < float(cfg.oscillation_reversal_cosine))))
        if (path_length < float(cfg.oscillation_min_target_path_px)
                or net_motion > float(cfg.oscillation_max_target_net_px)
                or reversals < int(cfg.oscillation_min_reversals)):
            return None
        return {
            "target_path_px": path_length,
            "target_net_px": net_motion,
            "reversals": reversals,
            "shape_distance_px": shape_distance,
        }

    def _sample_model2(self, obs_dict, obstacles, stream_ids, batch_size):
        indices = torch.as_tensor(
            stream_ids, device=obs_dict["obs"].device, dtype=torch.long)
        task_obs = obs_dict["obs"].index_select(0, indices)[..., :20]
        obstacle_vectors = []
        for stream_id in stream_ids:
            position, radius = self._obstacle_for_stream(
                obstacles, stream_id, batch_size)
            obstacle_vectors.append(np.r_[position, radius])
        obstacle_tensor = torch.as_tensor(
            np.stack(obstacle_vectors), device=task_obs.device,
            dtype=task_obs.dtype)
        obstacle_tensor = obstacle_tensor[:, None, :].expand(
            -1, task_obs.shape[1], -1)
        escape_obs = torch.cat([task_obs, obstacle_tensor], dim=-1)
        with torch.no_grad():
            return self._get_model2_policy().predict_action(
                {"obs": escape_obs})["action_pred"]

    def mind_filter_for_stream(self, stream_id):
        """Use the 3-corner r=10 filter only while model 2 owns the stream."""
        active = getattr(self, '_regrasp_active_flags', None)
        if (self._regrasp_mind_filter is not None and active is not None
                and stream_id < len(active) and bool(active[stream_id])):
            return self._regrasp_mind_filter
        return super().mind_filter_for_stream(stream_id)

    def mind_projection_mask(self, batch_size):
        """The no-MIND ablation excludes only active recovery streams."""
        mask = super().mind_projection_mask(batch_size)
        active = getattr(self, '_regrasp_active_flags', None)
        if (self._regrasp_mind_filter is None and active is not None
                and len(active) == int(batch_size)):
            mask &= ~np.asarray(active, dtype=np.bool_)
        return mask

    def mind_safety_modes_for_stream(self, stream_id, action_count):
        """Select safe-only or nonworsening recovery for model-2 escape."""
        active = getattr(self, '_regrasp_active_flags', None)
        if (self._regrasp_mind_filter is not None and active is not None
                and stream_id < len(active) and bool(active[stream_id])):
            mode = (
                "nonworsening"
                if self.regrasp_model2_config.corner_mind_recovery_enabled
                else "safe_only")
            return (mode,) * int(action_count)
        return super().mind_safety_modes_for_stream(stream_id, action_count)

    def _limit_regrasp_action_deltas(self, selected, agent):
        """Apply MIND's action-rate limit without enabling its CBF filter.

        The first model-2 target is limited relative to the current measured
        pusher position; every later target is limited relative to the
        preceding limited target.  This keeps ``--no-regrasp_mind`` free of
        safety optimization while retaining the same per-axis 25 px command
        bound used by the main-task MIND controller.
        """
        maximum_delta = float(self.mind_config.max_action_delta_px)
        anchor = torch.as_tensor(
            agent, device=selected.device, dtype=selected.dtype)
        limited = selected.clone()
        for action_index in range(selected.shape[0]):
            current = torch.maximum(
                torch.minimum(selected[action_index], anchor + maximum_delta),
                anchor - maximum_delta)
            limited[action_index] = current
            anchor = current
        return limited

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
        self._regrasp_mind_guard_flags = np.zeros(batch_size, dtype=np.bool_)
        self._model2_plans = {}
        active_stream_ids = []
        cfg = self.regrasp_model2_config
        task_start = (self.n_obs_steps - 1 if self.oa_step_convention
                      else self.n_obs_steps)
        task_end = min(
            task_start + int(self.mind_config.execution_action_steps),
            action_pred.shape[1])

        for stream_id in range(batch_size):
            self._regrasp_action_modes.pop(stream_id, None)
            points = raw_obs[stream_id, :18].reshape(9, 2)
            agent = raw_obs[stream_id, 18:20].astype(np.float64)
            obstacle, obstacle_radius = self._obstacle_for_stream(
                obstacles, stream_id, batch_size)
            center, clearance = self._geometry(
                points, obstacle, obstacle_radius)

            # An optional cap can suppress subsequent recoveries; by default
            # the inherited leave -> recontact -> real-push arming state
            # permits every newly qualifying stall to launch model 2.
            activation_count = self._model2_activation_counts.get(stream_id, 0)
            if (cfg.max_activations_per_episode is not None
                    and activation_count >= int(cfg.max_activations_per_episode)
                    and stream_id not in self._regrasp_modes):
                continue

            replan_count = self._model2_replan_counts.get(stream_id, 0)
            elapsed_sec = (
                float(replan_count)
                * float(self.mind_config.execution_action_steps)
                * float(self.mind_config.control_dt_sec))
            self._model2_replan_counts[stream_id] = replan_count + 1
            action_history = self._main_mind_action_path_histories.get(
                stream_id)
            action_path_length = None
            if (action_history is not None
                    and len(action_history) >= int(self.regrasp_config.history_replans)
                    and all(float(length) < float(
                        self.regrasp_config.maximum_trigger_action_path_length_px)
                            for length in action_history)):
                action_path_length = float(max(action_history))

            if stream_id not in self._regrasp_initial_center:
                self._regrasp_initial_center[stream_id] = center.copy()
            centers = self._regrasp_center_history.setdefault(
                stream_id, __import__("collections").deque(
                    maxlen=int(self.regrasp_config.history_replans)))
            agents = self._regrasp_agent_history.setdefault(
                stream_id, __import__("collections").deque(
                    maxlen=int(self.regrasp_config.history_replans)))
            centers.append(center.copy())
            agents.append(agent.copy())
            armed = self._update_regrasp_arming(
                stream_id, center, agent, points)
            cooldown = int(self._regrasp_cooldowns.get(stream_id, 0))
            if cooldown > 0:
                self._regrasp_cooldowns[stream_id] = cooldown - 1
            score = float(self._regrasp_task_scores.get(stream_id, 0.0))
            metrics = None
            can_consider = bool(
                stream_id not in self._regrasp_modes
                and score < 0.85 and cooldown == 0)
            stall_metrics = None
            if can_consider:
                # First check the common progress and actual T-motion gates.
                # The two new branches intentionally do not require the old
                # six-chunk action-path <30 px condition.
                stall_metrics = self._trigger_metrics(
                    stream_id, center, agent, obstacle, obstacle_radius,
                    elapsed_sec=elapsed_sec,
                    allow_oscillation_path=True)

            oscillation = None
            if stall_metrics is not None and armed:
                oscillation = self._oscillation_metrics(
                    stream_id, replan_count, points, agent)
            self._oscillation_streaks[stream_id] = (
                self._oscillation_streaks.get(stream_id, 0) + 1
                if oscillation is not None else 0)

            if stall_metrics is not None:
                if armed and action_path_length is not None:
                    # Preserve the original small-action stall trigger.
                    metrics = dict(stall_metrics,
                                   action_path_length=action_path_length,
                                   trigger_kind="small_action")
                elif (armed and oscillation is not None
                      and self._oscillation_streaks[stream_id]
                      >= int(cfg.oscillation_persistence_replans)):
                    metrics = dict(stall_metrics, **oscillation,
                                   trigger_kind="oscillation")
                elif (not armed and cfg.timeout_rearm_enabled
                      and stream_id in self._regrasp_rearm_states
                      and self._timeout_rearm_retry_counts.get(stream_id, 0)
                      < int(cfg.timeout_rearm_max_retries)):
                    last_exit = self._last_model2_exit_replans.get(stream_id)
                    shape_distance = float(np.min(np.linalg.norm(
                        points - agent[None, :], axis=1)))
                    if (last_exit is not None
                            and (replan_count - last_exit)
                            * int(self.mind_config.execution_action_steps)
                            >= int(cfg.timeout_rearm_after_steps)
                            and shape_distance <= float(
                                cfg.timeout_rearm_max_shape_distance_px)):
                        metrics = dict(stall_metrics,
                                       trigger_kind="rearm_timeout",
                                       shape_distance_px=shape_distance)
            # Fourth trigger: five continuous seconds without actual
            # pusher-to-T physics contact. Keep the existing score, cooldown,
            # and post-regrasp arming gates, but do not require stall_metrics:
            # the pusher may miss the T while executing large actions.
            no_contact_steps = self._no_contact_steps.get(stream_id, 0)
            threshold_steps = int(np.ceil(
                cfg.no_contact_trigger_seconds /
                float(self.mind_config.control_dt_sec)))
            if (metrics is None and can_consider and armed
                    and no_contact_steps >= threshold_steps):
                recent_centers = list(centers)
                goal = np.asarray(self.regrasp_config.goal_center,
                                  dtype=np.float64)
                metrics = {
                    "trigger_kind": "no_contact_5s",
                    "net_motion": float(np.linalg.norm(
                        recent_centers[-1] - recent_centers[0])),
                    "goal_progress": float(
                        np.linalg.norm(recent_centers[0] - goal)
                        - np.linalg.norm(recent_centers[-1] - goal)),
                }
            if metrics is not None:
                if metrics["trigger_kind"] == "rearm_timeout":
                    self._timeout_rearm_retry_counts[stream_id] = (
                        self._timeout_rearm_retry_counts.get(stream_id, 0) + 1)
                self._model2_activation_counts[stream_id] = (
                    activation_count + 1)
                self._start_model2(
                    stream_id, center, clearance, metrics)

            mode = self._regrasp_modes.get(stream_id)
            if mode is None:
                continue
            displacement = float(np.linalg.norm(
                center - mode["initial_center"]))
            succeeded = self._escape_succeeded(
                center, clearance, displacement)
            if mode["executed_steps"] >= int(cfg.max_steps):
                self._finish_model2(
                    stream_id, center, "max_steps", clearance, displacement)
                continue
            if (mode["executed_steps"] >= int(cfg.min_steps) and succeeded):
                self._finish_model2(
                    stream_id, center, "escape_success", clearance,
                    displacement)
                continue
            active_stream_ids.append(stream_id)
            self._regrasp_active_flags[stream_id] = True

        if active_stream_ids:
            plans = self._sample_model2(
                obs_dict, obstacles, active_stream_ids, batch_size)
            task_start = (self.n_obs_steps - 1 if self.oa_step_convention
                          else self.n_obs_steps)
            model2_policy = self._get_model2_policy()
            model2_start = (
                model2_policy.n_obs_steps - 1
                if model2_policy.oa_step_convention
                else model2_policy.n_obs_steps)
            for model2_index, stream_id in enumerate(active_stream_ids):
                count = min(
                    result.shape[1] - task_start,
                    plans.shape[1] - model2_start)
                if count <= 0:
                    raise RuntimeError("model 2 produced no executable actions")
                selected = plans[
                    model2_index,
                    model2_start:model2_start + count,
                    :2]
                selected = self._limit_regrasp_action_deltas(
                    selected, raw_obs[stream_id, 18:20])
                result[
                    stream_id,
                    task_start:task_start + count,
                    :2] = selected
                self._model2_plans[stream_id] = (
                    task_start, selected.detach().clone())
                self._regrasp_modes[stream_id]["executed_steps"] += int(
                    self.mind_config.execution_action_steps)
        return result

    def predict_action(self, obs_dict, obstacles=None):
        result = super().predict_action(obs_dict, obstacles)
        # ``transform_diffusion_action_plan`` runs before MIND. Cache the
        # just-projected main-task chunk here, then use it at the next replan
        # as the trigger's action-length signal. This is the exact MIND output
        # that the environment is about to execute, without a second preview
        # solve or any mutation of MIND's internal state.
        projected = result.get("mind_projected_action_chunk")
        projected_mask = result.get("mind_projected_mask")
        active = result["regrasp_active"].detach().cpu().numpy()
        if projected is not None and projected_mask is not None:
            agents = obs_dict["obs"][:, -1, 18:20].detach().cpu().numpy()
            chunk = projected.detach().cpu().numpy()
            mask = projected_mask.detach().cpu().numpy()
            for stream_id, was_projected in enumerate(mask):
                if bool(was_projected) and not bool(active[stream_id]):
                    points = np.vstack((agents[stream_id][None, :],
                                        chunk[stream_id, :, :2]))
                    history = self._main_mind_action_path_histories.setdefault(
                        stream_id, __import__("collections").deque(
                            maxlen=int(self.regrasp_config.history_replans)))
                    history.append(float(np.linalg.norm(
                        np.diff(points, axis=0), axis=1).sum()))
                    target_history = (
                        self._main_mind_action_target_histories.setdefault(
                            stream_id, deque(maxlen=int(
                                self.regrasp_config.history_replans))))
                    target_history.append(
                        np.asarray(chunk[stream_id, :, :2],
                                   dtype=np.float64).copy())
        # Dedicated runner signal: unlike the generic visualization flag,
        # this suppresses task termination only for model-2 recovery.
        result["regrasp_force_continue"] = result[
            "regrasp_active"].detach().clone()
        # Keep the video annotation policy-specific: an ordinary mirror
        # regrasp should not be labelled as diffusion model 2.
        result["regrasp_model2_active"] = result[
            "regrasp_active"].detach().clone()
        if self._regrasp_mind_filter is None and self.mind_filter is not None:
            active = result["regrasp_active"].detach().cpu().numpy()
            for stream_id, is_active in enumerate(active):
                if bool(is_active):
                    action = result["action"][stream_id, -1, :2]
                    action = action.detach().cpu().numpy()
                    self.mind_filter._previous_action_norm[stream_id] = np.clip(
                        (action - self.mind_filter.action_min)
                        / self.mind_filter.action_range, 0.0, 1.0)
        raw = result.get("diffusion_action_pred")
        if raw is not None:
            for stream_id, (start, plan) in self._model2_plans.items():
                count = min(raw.shape[1] - start, plan.shape[0])
                raw[stream_id, start:start + count, :2] = plan[:count].to(
                    device=raw.device, dtype=raw.dtype)
        return result
