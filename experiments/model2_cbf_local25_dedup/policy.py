"""Sample12 once, merge near prefixes, try representatives in stable order."""
import copy
import json
import types

import numpy as np
import torch

from experiments.model2_cbf_full_page.filter import _solve_page_coordinates
from experiments.model2_cbf_full_page_1step_p047.policy import OneStepP047Policy


def merge_prefixes(plans, start, steps=4, mean_px=5.0, max_px=10.0):
    """Greedy groups against first representatives; never average actions."""
    prefixes = np.asarray(plans)[:, start:start + steps, :2]
    if prefixes.shape[1:] != (steps, 2) or not np.isfinite(prefixes).all():
        raise ValueError('Expected finite four-step future action prefixes')
    groups = []
    for index, prefix in enumerate(prefixes):
        for group in groups:
            distance = np.linalg.norm(prefix - prefixes[group[0]], axis=1)
            if distance.mean() <= mean_px and distance.max() <= max_px:
                group.append(index)
                break
        else:
            groups.append([index])
    return groups


def _solve_key(action_ref, state_norm, previous_norm, anchor, positions, radii,
               stream_id, safety_mode):
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (action_ref, state_norm, previous_norm, anchor, positions, radii)]
    return (int(stream_id), str(safety_mode),
            tuple((x.shape, x.tobytes()) for x in arrays))


def _solve_once_or_replay(self, action_ref, state_norm, previous_action_norm,
                          action_anchor, obstacle_positions, obstacle_radii,
                          stream_id=0, safety_mode='strict'):
    """Reuse a preview only for the exact same MIP inputs and safety mode.

    The ordinary live project()/project_sequence() still commits history,
    status, visualization and diagnostics exactly once. Preview copies never
    alter live state. A failed preview is replayed as failure, not re-solved.
    """
    key = _solve_key(action_ref, state_norm, previous_action_norm, action_anchor,
                     obstacle_positions, obstacle_radii, stream_id, safety_mode)
    pending = self._dedup_pending_solves.get(int(stream_id))
    if pending:
        expected, value = pending[0]
        if key != expected:
            raise RuntimeError('Candidate CBF certificate inputs changed before execution')
        pending.pop(0)
        if not pending:
            del self._dedup_pending_solves[int(stream_id)]
        self._dedup_reuse_count += 1
        return copy.deepcopy(value)
    if self._dedup_capture is None:
        raise RuntimeError('Live model2 CBF called without a candidate certificate')
    value = _solve_page_coordinates(
        self, action_ref, state_norm, previous_action_norm, action_anchor,
        obstacle_positions, obstacle_radii, stream_id=stream_id,
        safety_mode=safety_mode)
    self._dedup_capture.append((key, copy.deepcopy(value)))
    return value


class DeduplicatedLocalPolicy(OneStepP047Policy):
    candidate_count = 12
    candidate_batch_limit = 1
    direction_switch_enabled = False
    dedup_prefix_steps = 4
    dedup_mean_px = 5.0
    dedup_max_px = 10.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        filter_ = self._regrasp_mind_filter
        # Parent checks direct local25 before installing this cache adapter.
        filter_._dedup_pending_solves = {}
        filter_._dedup_capture = None
        filter_._dedup_reuse_count = 0
        filter_._solve = types.MethodType(_solve_once_or_replay, filter_)
        self._dedup_groups_total = 0
        self._dedup_merged_total = 0
        self._dedup_trials_total = 0
        self._dedup_solver_calls_total = 0
        self._dedup_solver_runtime_total = 0.0
        self._dedup_trace_file = None
        print('[MODEL2 DEDUP] K=12, one batch; future4 mean<=5px AND max<=10px; '
              'first representative, stable order, stop at first feasible; '
              'CBF preview/execute=1; reuse exact certificate', flush=True)

    def reset(self):
        super().reset()
        filter_ = getattr(self, '_regrasp_mind_filter', None)
        if filter_ is not None and hasattr(filter_, '_dedup_pending_solves'):
            filter_._dedup_pending_solves.clear()

    def predict_action(self, obs_dict, obstacles=None):
        result = super().predict_action(obs_dict, obstacles)
        if self._regrasp_mind_filter._dedup_pending_solves:
            raise RuntimeError('An unconsumed model2 CBF certificate remains')
        return result

    def _sample_model2(self, obs_dict, obstacles, stream_ids, batch_size):
        indices = torch.as_tensor(stream_ids, device=obs_dict['obs'].device,
                                  dtype=torch.long)
        task_obs = obs_dict['obs'].index_select(0, indices)[..., :20]
        vectors = []
        for stream_id in stream_ids:
            position, radius = self._obstacle_for_stream(obstacles, stream_id, batch_size)
            vectors.append(np.r_[position, radius])
        obstacle_tensor = torch.as_tensor(np.stack(vectors), device=task_obs.device,
                                           dtype=task_obs.dtype)[:, None, :]
        escape_obs = torch.cat([task_obs, obstacle_tensor.expand(
            -1, task_obs.shape[1], -1)], dim=-1)
        model2 = self._get_model2_policy()
        with torch.no_grad():
            all_plans = model2.predict_action({'obs': escape_obs.repeat_interleave(
                self.candidate_count, dim=0)})['action_pred']
        plans = all_plans.reshape(len(stream_ids), self.candidate_count,
                                  *all_plans.shape[1:])
        start = model2.n_obs_steps - 1 if model2.oa_step_convention else model2.n_obs_steps
        chosen = []
        filter_ = self._regrasp_mind_filter
        for row, stream_id in enumerate(stream_ids):
            if int(stream_id) in filter_._dedup_pending_solves:
                raise RuntimeError('Previous candidate certificate was not consumed')
            raw_plans = plans[row].detach().cpu().numpy()
            groups = merge_prefixes(raw_plans, start, self.dedup_prefix_steps,
                                    self.dedup_mean_px, self.dedup_max_px)
            state_raw = task_obs[row, -1, :18].detach().cpu().numpy()
            agent = task_obs[row, -1, 18:20].detach().cpu().numpy()
            position, radius = self._obstacle_for_stream(obstacles, stream_id, batch_size)
            positions = np.asarray(position).reshape(-1, 2)
            radii = np.asarray(radius).reshape(-1)
            modes = self.mind_safety_modes_for_stream(stream_id, 1)
            trials = []
            selected_index = None
            selected_capture = None
            first_failed_capture = None
            selected_movement = 0.0
            for group in groups:
                index = group[0]
                preview = copy.deepcopy(filter_)
                preview._dedup_pending_solves = {}
                preview._dedup_capture = []
                raw = plans[row, index, start:start + 1, :2]
                limited = self._limit_regrasp_action_deltas(raw, agent)
                projected, statuses = preview.project_sequence(
                    limited.detach().cpu().numpy(), state_raw, positions, radii,
                    stream_id=stream_id, fallback_action=agent, safety_modes=modes)
                capture = preview._dedup_capture
                if len(capture) != 1:
                    raise RuntimeError('One-step CBF must make exactly one local solve')
                diagnostics = capture[0][1][1]
                feasible = (capture[0][1][0] is not None and not any(
                    status.endswith('_agent_hold') for status in statuses))
                if feasible:
                    if np.max(np.abs(projected[0] - agent)) > 25.0001:
                        raise RuntimeError('Candidate exceeds the executable25px box')
                    if abs(float(diagnostics['safety_slack'])) > 1.e-7:
                        raise RuntimeError('Hard-CBF candidate has positive safety slack')
                self._dedup_trials_total += 1
                self._dedup_solver_calls_total += len(capture)
                self._dedup_solver_runtime_total += float(diagnostics['runtime_sec'])
                trials.append(dict(representative=index, group_size=len(group),
                                   feasible=bool(feasible), statuses=list(statuses),
                                   solver=dict(diagnostics)))
                if first_failed_capture is None:
                    first_failed_capture = capture
                if feasible:
                    selected_index = index
                    selected_capture = capture
                    selected_movement = float(np.linalg.norm(projected[0] - agent))
                    break
            all_failed = selected_index is None
            if all_failed:
                selected_index = groups[0][0]
                selected_capture = first_failed_capture
                self._candidate_all_infeasible_total += 1
                print(f'[MODEL2 DEDUP] env={stream_id} all {len(groups)} '
                      'representatives have no feasible incumbent; agent hold', flush=True)
            filter_._dedup_pending_solves[int(stream_id)] = selected_capture
            chosen.append(plans[row, selected_index])
            self._candidate_replans += 1
            self._candidate_sampled_total += self.candidate_count
            self._moving_selections += int(selected_movement >= 12.0)
            self._dedup_groups_total += len(groups)
            self._dedup_merged_total += self.candidate_count - len(groups)
            if self._dedup_trace_file is not None:
                event = dict(replan=self._candidate_replans, stream_id=int(stream_id),
                             model2_executed_steps=int(self._regrasp_modes.get(
                                 stream_id, {}).get('executed_steps', 0)),
                             groups=groups, selected_index=int(selected_index),
                             all_infeasible=bool(all_failed), trials=trials,
                             first4=np.round(raw_plans[:, start:start + 4, :2], 3).tolist())
                self._dedup_trace_file.write(json.dumps(event) + '\n')
        return torch.stack(chosen, dim=0)
