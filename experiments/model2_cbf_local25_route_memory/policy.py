"""Keep the active model-2 strategy; group alternatives only on failure/stall."""
import copy
import json
import types

import numpy as np
import torch

from experiments.model2_cbf_full_page_1step_p047.policy import OneStepP047Policy
from experiments.model2_cbf_local25_dedup.policy import _solve_once_or_replay
from experiments.model2_cbf_local25_route_memory.routes import RouteSelector, Trial


class RouteMemoryLocalPolicy(OneStepP047Policy):
    candidate_count = 12
    candidate_batch_limit = 1
    direction_switch_enabled = False

    def __init__(self, *args, route_similarity_px=50.,
                 route_progress_window_steps=None, route_progress_px=4., **kwargs):
        super().__init__(*args, **kwargs)
        self.route_similarity_px = float(route_similarity_px)
        self.route_progress_window_steps = int(
            self.regrasp_config.history_replans if route_progress_window_steps is None
            else route_progress_window_steps)
        self.route_progress_px = float(route_progress_px)
        self._route_selectors = {}
        self._last_executed_targets = {}
        self._route_trace_file = None
        self._route_counters = dict(replans=0, clustering_calls=0, groups=0,
                                    merged=0, trials=0, solver_runtime_sec=0.,
                                    switches=0, continued_without_clustering=0,
                                    proven_empty=0, stalled_safe_fallback=0)
        filter_ = self._regrasp_mind_filter
        filter_._dedup_pending_solves = {}
        filter_._dedup_capture = None
        filter_._dedup_reuse_count = 0
        filter_._solve = types.MethodType(_solve_once_or_replay, filter_)
        print('[MODEL2 ROUTE MEMORY] K=12; discrete Frechet on future15; '
              f'complete-link <= {self.route_similarity_px:g}px; medoid; '
              f'lazy grouping; progress window={self.route_progress_window_steps} '
              f'steps/{self.route_progress_px:g}px; CBF preview/execute=1; '
              'direct local25; no hand-designed directions', flush=True)

    def reset(self):
        super().reset()
        for name in ('_route_selectors', '_last_executed_targets'):
            getattr(self, name, {}).clear()
        filter_ = getattr(self, '_regrasp_mind_filter', None)
        if filter_ is not None and hasattr(filter_, '_dedup_pending_solves'):
            filter_._dedup_pending_solves.clear()

    def _start_model2(self, stream_id, center, clearance, metrics):
        super()._start_model2(stream_id, center, clearance, metrics)
        self._route_selectors.pop(int(stream_id), None)

    def _finish_model2(self, stream_id, *args, **kwargs):
        super()._finish_model2(stream_id, *args, **kwargs)
        self._route_selectors.pop(int(stream_id), None)

    def predict_action(self, obs_dict, obstacles=None):
        result = super().predict_action(obs_dict, obstacles)
        if self._regrasp_mind_filter._dedup_pending_solves:
            raise RuntimeError('Unconsumed model-2 candidate certificate')
        actions = result['action'].detach().cpu().numpy()
        for stream_id in range(len(actions)):
            self._last_executed_targets[stream_id] = actions[stream_id, -1, :2].copy()
        return result

    def _sample_model2(self, obs_dict, obstacles, stream_ids, batch_size):
        ids = torch.as_tensor(stream_ids, device=obs_dict['obs'].device, dtype=torch.long)
        task_obs = obs_dict['obs'].index_select(0, ids)[..., :20]
        vectors = [np.r_[self._obstacle_for_stream(obstacles, s, batch_size)[0],
                         self._obstacle_for_stream(obstacles, s, batch_size)[1]]
                   for s in stream_ids]
        obstacle_tensor = torch.as_tensor(np.stack(vectors), device=task_obs.device,
                                           dtype=task_obs.dtype)[:, None, :]
        escape_obs = torch.cat((task_obs, obstacle_tensor.expand(
            -1, task_obs.shape[1], -1)), dim=-1)
        model2 = self._get_model2_policy()
        with torch.no_grad():
            sampled = model2.predict_action({'obs': escape_obs.repeat_interleave(
                self.candidate_count, dim=0)})['action_pred']
        plans = sampled.reshape(len(stream_ids), self.candidate_count, *sampled.shape[1:])
        start = model2.n_obs_steps - 1 if model2.oa_step_convention else model2.n_obs_steps
        if plans.shape[2] - start != 15:
            raise RuntimeError('This controller expects exactly 15 future target points')
        selected_plans = []
        filter_ = self._regrasp_mind_filter
        for row, stream_id in enumerate(stream_ids):
            stream_id = int(stream_id)
            if stream_id in filter_._dedup_pending_solves:
                raise RuntimeError('Previous CBF certificate was not consumed')
            state = task_obs[row, -1, :18].detach().cpu().numpy()
            agent = task_obs[row, -1, 18:20].detach().cpu().numpy()
            position, radius = self._obstacle_for_stream(obstacles, stream_id, batch_size)
            positions = np.asarray(position).reshape(-1, 2)
            radii = np.asarray(radius).reshape(-1)
            center, clearance = self._geometry(state.reshape(9, 2), positions[0], radii[0])
            # All candidate probes use the same real previous command. In
            # particular, a stage transition must not use the candidate itself
            # as the dynamics model's missing previous-action input.
            previous = self._last_executed_targets.get(stream_id, agent)
            filter_._previous_action_norm[stream_id] = np.clip(
                (previous - filter_.action_min) / filter_.action_range, 0., 1.)
            modes = self.mind_safety_modes_for_stream(stream_id, 1)
            selector = self._route_selectors.setdefault(stream_id, RouteSelector(
                self.route_similarity_px, self.route_progress_window_steps,
                self.route_progress_px, self.regrasp_config.waypoint_capture_radius_px))

            def probe(path, candidate_index):
                preview = copy.deepcopy(filter_)
                preview._dedup_pending_solves = {}
                preview._dedup_capture = []
                projected, statuses = preview.project_sequence(
                    path[:1], state, positions, radii, stream_id=stream_id,
                    fallback_action=agent, safety_modes=modes)
                capture = preview._dedup_capture
                if len(capture) != 1:
                    raise RuntimeError('One-step preview must produce exactly one solve')
                action, diagnostics = capture[0][1]
                feasible = action is not None and not any(
                    status.endswith('_agent_hold') for status in statuses)
                if feasible:
                    # Gurobi's normalized feasibility tolerance can be about
                    # 0.000512 physical px. Check with a 0.001px numerical
                    # tolerance; the MIP bounds remain EXACTLY +/-25px and no
                    # post-solve clipping changes its certified solution.
                    if np.max(np.abs(projected[0] - agent)) > 25.001:
                        raise RuntimeError('CBF candidate violates local25 bounds')
                    used_slack = float(diagnostics['safety_slack'])
                    if getattr(self, 'depth_budget_enabled', False):
                        certificate = diagnostics.get('depth_budget', {})
                        if not certificate.get('verified', False):
                            raise RuntimeError('Candidate lacks a verified geometric depth certificate')
                    elif self.bounded_soft_cbf_enabled:
                        # One CBF slack variable exists for each proxy/obstacle
                        # pair. The aggregate diagnostic can exceed one cap.
                        pair_count = len(filter_.config.safety_proxy_radii) * len(radii)
                        if used_slack > pair_count * self.bounded_soft_cbf_cap_px2 + 1.e-5:
                            raise RuntimeError('Bounded-soft candidate exceeds total slack cap')
                    elif abs(used_slack) > 1.e-7:
                        raise RuntimeError('Hard-CBF candidate has safety slack')
                # Only INFEASIBLE is a proof. TIME_LIMIT, NODE_LIMIT etc.
                # merely mean no incumbent was found. With radius>=1 and the
                # fixed previous action above, the feasible set is reference
                # independent; switching references cannot fix an empty set.
                proven = (not feasible and diagnostics['status'] == 'INFEASIBLE'
                          and filter_.config.action_search_radius >= 1.)
                self._route_counters['trials'] += 1
                self._route_counters['solver_runtime_sec'] += float(diagnostics['runtime_sec'])
                return Trial(bool(feasible), bool(proven), dict(
                    capture=capture, projected=projected, statuses=list(statuses),
                    diagnostics=dict(diagnostics)))

            paths = plans[row, :, start:, :2].detach().cpu().numpy()
            decision = selector.select(paths, agent, center, clearance, probe)
            trial = decision['trial']
            filter_._dedup_pending_solves[stream_id] = trial.payload['capture']
            if decision['index'] is None:
                selected = plans[row, 0].clone()
                selected[start:, :2] = torch.as_tensor(
                    decision['path'], device=selected.device, dtype=selected.dtype)
            else:
                selected = plans[row, decision['index']]
            selected_plans.append(selected)
            self._candidate_replans += 1
            self._candidate_sampled_total += self.candidate_count
            self._candidate_all_infeasible_total += int(not trial.feasible)
            self._moving_selections += int(trial.feasible and np.linalg.norm(
                trial.payload['projected'][0] - agent) >= 12.)
            counts = self._route_counters
            counts['replans'] += 1
            counts['switches'] += int(decision['switched'])
            counts['proven_empty'] += int(trial.proven_empty)
            counts['stalled_safe_fallback'] += int(decision['reason'] == 'stalled_keep_safe_action')
            if decision['groups'] is None:
                counts['continued_without_clustering'] += int(trial.feasible)
            else:
                counts['clustering_calls'] += 1
                counts['groups'] += len(decision['groups'])
                counts['merged'] += 12 - len(decision['groups'])
            if self._route_trace_file is not None:
                event = {key: decision[key] for key in (
                    'index', 'groups', 'medoids', 'reason', 'stalled', 'switched', 'strategy_id')}
                event.update(stream_id=stream_id, replan=self._candidate_replans,
                             future15=np.round(paths, 3).tolist(),
                             selected_future15=np.round(decision['path'], 3).tolist(),
                             trials=[dict(index=entry['index'],
                                          feasible=entry['trial'].feasible,
                                          proven_empty=entry['trial'].proven_empty,
                                          **entry['trial'].payload['diagnostics'])
                                     for entry in decision['trials']])
                self._route_trace_file.write(json.dumps(event) + '\n')
        return torch.stack(selected_plans)
