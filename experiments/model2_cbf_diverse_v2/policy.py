"""CBF-previewed model-2 candidates with a contact and stall-switch score."""

import copy
import numpy as np
import torch

from experiments.contact_gap_5s.policy_contact_gap import (
    Model2RegraspDiffusionPolicy, _LOCAL_T_POLYGONS,
)


class DiverseCBFModel2Policy(Model2RegraspDiffusionPolicy):
    candidate_count = 4
    direction_switch_enabled = True
    # Historical v2/hard runs use exactly one batch. An experimental child
    # policy may retry only when every candidate in that batch is infeasible.
    candidate_batch_limit = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._regrasp_mind_filter is None:
            raise ValueError("Model-2 CBF must be enabled with --regrasp_mind")
        self._candidate_replans = 0
        self._moving_selections = 0
        self._stuck_switches = 0
        self._last_centers = {}
        self._stuck_streaks = {}
        self._last_selected_directions = {}
        self._candidate_sampled_total = 0
        self._candidate_extra_batches_total = 0
        self._candidate_all_infeasible_total = 0
        direction_mode = (" + stalled-direction switch"
                          if self.direction_switch_enabled else "")
        print(f"[MODEL2 CBF DIVERSE V2] K={self.candidate_count}, "
              f"CBF preview + contact score{direction_mode}",
              flush=True)

    def reset(self):
        super().reset()
        self._candidate_replans = 0
        self._moving_selections = 0
        self._stuck_switches = 0
        self._last_centers = {}
        self._stuck_streaks = {}
        self._last_selected_directions = {}

    def _sample_model2(self, obs_dict, obstacles, stream_ids, batch_size):
        indices = torch.as_tensor(stream_ids, device=obs_dict["obs"].device,
                                  dtype=torch.long)
        task_obs = obs_dict["obs"].index_select(0, indices)[..., :20]
        obstacle_vectors = []
        for stream_id in stream_ids:
            position, radius = self._obstacle_for_stream(
                obstacles, stream_id, batch_size)
            obstacle_vectors.append(np.r_[position, radius])
        obstacle_tensor = torch.as_tensor(
            np.stack(obstacle_vectors), device=task_obs.device,
            dtype=task_obs.dtype)[:, None, :].expand(-1, task_obs.shape[1], -1)
        escape_obs = torch.cat([task_obs, obstacle_tensor], dim=-1)
        # Each repeated observation receives independent diffusion noise in
        # one batched network call. No hand-authored escape direction is imposed.
        repeated = escape_obs.repeat_interleave(self.candidate_count, dim=0)
        with torch.no_grad():
            all_plans = self._get_model2_policy().predict_action(
                {"obs": repeated})["action_pred"]
        plans = all_plans.reshape(len(stream_ids), self.candidate_count,
                                  *all_plans.shape[1:])
        model2_policy = self._get_model2_policy()
        start = (model2_policy.n_obs_steps - 1
                 if model2_policy.oa_step_convention else model2_policy.n_obs_steps)
        chosen = []
        filter_ = self._regrasp_mind_filter
        for row, stream_id in enumerate(stream_ids):
            state_raw = task_obs[row, -1, :18].detach().cpu().numpy()
            agent = task_obs[row, -1, 18:20].detach().cpu().numpy()
            obstacle, radius = self._obstacle_for_stream(
                obstacles, stream_id, batch_size)
            obstacle = np.asarray(obstacle).reshape(-1, 2)
            radius = np.asarray(radius).reshape(-1)
            initial_norm = (state_raw - filter_.keypoint_min) / filter_.keypoint_range
            initial_margins = self._proxy_margins(initial_norm, obstacle, radius)
            points = state_raw.reshape(9, 2)
            center, _ = self._geometry(points, obstacle[0], radius[0])
            stalled = False
            if self.direction_switch_enabled:
                last_center = self._last_centers.get(stream_id)
                streak = self._stuck_streaks.get(stream_id, 0)
                streak = (streak + 1 if last_center is not None
                          and np.linalg.norm(center - last_center) < 2.0 else 0)
                self._stuck_streaks[stream_id] = streak
                self._last_centers[stream_id] = center
                stalled = streak >= max(1, round(
                    3 * 4 / self.mind_config.execution_action_steps))
            rotation, translation = self._rigid_transform(points)
            polygons = [shape @ rotation.T + translation
                        for shape in _LOCAL_T_POLYGONS]
            best_score, best_movement = -np.inf, 0.0
            best_direction = None
            best_plan = None

            def score_candidate(plan):
                raw = plan[
                            start:start + self.mind_config.execution_action_steps,
                            :2]
                limited = self._limit_regrasp_action_deltas(raw, agent)
                actions = limited.detach().cpu().numpy()
                # Preview the complete execution chunk with an independent
                # copy of MIND. Candidate previews cannot alter the live CBF
                # controller's action history, hysteresis, or counters.
                preview = copy.deepcopy(filter_)
                before = preview.diagnostics_summary()
                projected, statuses = preview.project_sequence(
                    actions, state_raw, obstacle, radius,
                    stream_id=stream_id, fallback_action=agent,
                    safety_modes=("nonworsening",) * len(actions))
                after = preview.diagnostics_summary()
                if any(status.endswith("_agent_hold") for status in statuses):
                    return None
                slack = (after["mean_safety_slack"] * after["solve_count"]
                         - before["mean_safety_slack"] * before["solve_count"])
                path = np.vstack([agent, projected])
                segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
                movement = float(segment_lengths.sum())
                direction = projected[-1] - agent
                net = float(np.linalg.norm(direction))
                contact_distance = min(
                    self._point_polygon_distance(position, polygon)
                    for position in projected for polygon in polygons)
                state_norm = initial_norm.copy()
                previous_norm = filter_._previous_action_norm.get(
                    stream_id, np.clip((projected[0] - filter_.action_min)
                                       / filter_.action_range, 0.0, 1.0)).copy()
                worst_change = 0.0
                final_margins = initial_margins
                for action in projected:
                    current_norm = np.clip((action - filter_.action_min)
                                           / filter_.action_range, 0.0, 1.0)
                    state_norm = filter_._predict_next_state_norm(
                        state_norm, previous_norm, current_norm)
                    previous_norm = current_norm
                    margins = self._proxy_margins(state_norm, obstacle, radius)
                    worst_change = min(worst_change,
                                       float(np.min(margins - initial_margins)))
                    final_margins = margins
                margin_gain = float(np.min(final_margins - initial_margins))
                previous_direction = self._last_selected_directions.get(stream_id)
                novelty = 0.0
                if stalled and previous_direction is not None and net > 1.0:
                    cosine = float(np.dot(direction, previous_direction)
                                   / max(net * np.linalg.norm(previous_direction),
                                         1.0e-6))
                    novelty = 10.0 * (1.0 - np.clip(cosine, -1.0, 1.0))
                # For recovery there is no fixed goal: prioritize moving T
                # outward after CBF projection, approaching T, and changing
                # direction when several closed-loop replans made no progress.
                score = (4.0 * worst_change + 3.0 * margin_gain
                         + 0.10 * net + 0.025 * movement
                         + 0.2 * max(0.0, 40.0 - contact_distance)
                         + novelty - 0.03 * max(0.0, slack)
                         - 0.05 * float(np.linalg.norm(projected - actions)))
                return score, movement, direction

            candidate_plans = plans[row]
            for batch_index in range(int(self.candidate_batch_limit)):
                if batch_index > 0:
                    # A fresh network call samples fresh Gaussian diffusion
                    # noise. Retry this stream only; streams with a feasible
                    # candidate do not pay for extra inference or CBF solves.
                    repeated_one = escape_obs[row:row + 1].repeat_interleave(
                        self.candidate_count, dim=0)
                    with torch.no_grad():
                        candidate_plans = self._get_model2_policy().predict_action(
                            {"obs": repeated_one})["action_pred"]
                    self._candidate_extra_batches_total += 1
                self._candidate_sampled_total += self.candidate_count
                for index in range(self.candidate_count):
                    candidate = candidate_plans[index]
                    rating = score_candidate(candidate)
                    if rating is None:
                        continue
                    score, movement, direction = rating
                    if score > best_score:
                        best_score, best_movement = score, movement
                        best_direction = direction
                        best_plan = candidate
                if best_plan is not None:
                    break
            if best_plan is None:
                # Keep the live hard-CBF filter in control. It will hold the
                # measured pusher if this last proposal has no incumbent too.
                best_plan = candidate_plans[0]
                self._candidate_all_infeasible_total += 1
                print(
                    f"[MODEL2 CBF] env={stream_id} all "
                    f"{self.candidate_count * self.candidate_batch_limit} "
                    "sampled trajectories infeasible; live CBF decides hold",
                    flush=True)
            chosen.append(best_plan)
            self._candidate_replans += 1
            self._moving_selections += int(best_movement >= 12.0)
            if self.direction_switch_enabled and best_direction is not None:
                self._last_selected_directions[stream_id] = best_direction
            self._stuck_switches += int(stalled)
        return torch.stack(chosen, dim=0)

    def _proxy_margins(self, state_norm, obstacles, radii):
        filter_ = self._regrasp_mind_filter
        margins = []
        for obstacle, radius in zip(obstacles, radii):
            for indices, proxy_radius in zip(
                    filter_.config.safety_keypoint_groups,
                    filter_.config.safety_proxy_radii):
                center = filter_._center_raw(state_norm, indices)
                margins.append(float(np.linalg.norm(center - obstacle)
                                     - radius * filter_.config.clearance_scale
                                     - proxy_radius))
        return np.asarray(margins, dtype=np.float64)
