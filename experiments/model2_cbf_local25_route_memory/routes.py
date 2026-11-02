"""Route geometry and selection, independent of Torch, the simulator and CBF."""
from collections import deque
from dataclasses import dataclass

import numpy as np


def _curve(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not len(value):
        raise ValueError('A path must have shape (N, 2), N >= 1')
    if not np.isfinite(value).all():
        raise ValueError('Path coordinates must be finite')
    return value


def frechet_distance(first, second):
    """Discrete Frechet on ordered vertices, in physical pixels; no densify."""
    first, second = _curve(first), _curve(second)
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    previous = np.full(len(second), np.inf)
    for i in range(len(first)):
        current = np.full(len(second), np.inf)
        for j in range(len(second)):
            if i == j == 0:
                predecessor = 0.0
            else:
                predecessor = min(previous[j],
                                  current[j - 1] if j else np.inf,
                                  previous[j - 1] if j else np.inf)
            current[j] = max(distances[i, j], predecessor)
        previous = current
    return float(previous[-1])


def group_routes(paths, threshold_px=50.0):
    """Deterministic complete-link cut and actual-member medoids.

    Merge the closest two clusters by their maximum pair distance, until
    every remaining merge exceeds the threshold. Ties use original indices.
    No chained matches, fixed group count, averaging or singleton removal.
    """
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[-1] != 2 or not len(paths):
        raise ValueError('Expected (K, N, 2) paths')
    if not np.isfinite(threshold_px) or threshold_px < 0:
        raise ValueError('Similarity threshold must be finite and nonnegative')
    for path in paths:
        _curve(path)
    distances = np.zeros((len(paths), len(paths)))
    for i in range(len(paths)):
        for j in range(i):
            distances[i, j] = distances[j, i] = frechet_distance(paths[i], paths[j])
    groups = [[i] for i in range(len(paths))]
    while len(groups) > 1:
        distance, _, _, i, j = min(
            (float(distances[np.ix_(a, b)].max()), tuple(a), tuple(b), i, j)
            for i, a in enumerate(groups) for j, b in enumerate(groups) if i < j)
        if distance > threshold_px:
            break
        groups[i] = sorted(groups[i] + groups[j])
        del groups[j]
        groups.sort(key=lambda group: group[0])
    medoids = [group[int(np.argmin(distances[np.ix_(group, group)].mean(axis=1)))]
               for group in groups]
    return groups, medoids, distances


def arc_position(path, point):
    """Arc coordinate of the nearest projection onto a polyline."""
    path, point = _curve(path), np.asarray(point, dtype=np.float64)
    if len(path) == 1:
        return 0.0
    edges = np.diff(path, axis=0)
    square = np.sum(edges * edges, axis=1)
    fraction = np.clip(np.sum((point - path[:-1]) * edges, axis=1)
                       / np.maximum(square, 1.e-12), 0., 1.)
    projections = path[:-1] + fraction[:, None] * edges
    index = int(np.argmin(np.linalg.norm(projections - point, axis=1)))
    lengths = np.sqrt(square)
    return float(lengths[:index].sum() + fraction[index] * lengths[index])


@dataclass
class Trial:
    feasible: bool
    proven_empty: bool = False
    payload: object = None


class RouteSelector:
    """One regrasp activation of one stream; never shared across scenes.

    The probe callback solves one step at the SAME measured state. A safe,
    progressing route always wins without clustering or comparing scores.
    """
    def __init__(self, threshold_px=50., window_steps=24, progress_px=4.,
                 waypoint_capture_px=22.):
        if window_steps < 1 or progress_px <= 0:
            raise ValueError('Progress window and tolerance must be positive')
        self.threshold_px = float(threshold_px)
        self.window_steps = int(window_steps)
        self.progress_px = float(progress_px)
        self.waypoint_capture_px = float(waypoint_capture_px)
        self.future = None
        self.reference = None
        self.history = deque(maxlen=self.window_steps + 1)
        self.failed_routes = []
        self.step = 0
        self.last_search = 0
        self.strategy_id = 0

    def _stalled(self):
        if len(self.history) < self.window_steps + 1:
            return False
        if self.step - self.last_search < self.window_steps:
            return False
        first, last = self.history[0], self.history[-1]
        t_motion = np.linalg.norm(last[1] - first[1])
        clearance_gain = last[2] - first[2]
        agent_motion = np.linalg.norm(last[0] - first[0])
        route_progress = (arc_position(self.reference, last[0])
                          - arc_position(self.reference, first[0]))
        return not (t_motion >= self.progress_px
                    or clearance_gain >= self.progress_px
                    or (agent_motion >= self.progress_px
                        and route_progress >= self.progress_px))

    def _remember_failure(self, agent, center):
        if self.future is not None:
            self.failed_routes.append((self.future.copy(), agent.copy(), center.copy()))
            self.failed_routes = self.failed_routes[-12:]

    def select(self, paths, agent, center, clearance, probe):
        paths = np.asarray(paths, dtype=np.float64)
        if paths.shape != (12, 15, 2) or not np.isfinite(paths).all():
            raise ValueError('Expected exactly 12 finite paths of 15 future targets')
        agent, center = np.asarray(agent).copy(), np.asarray(center).copy()
        self.step += 1
        self.history.append((agent.copy(), center.copy(), float(clearance)))
        # Failed routes become eligible again after a material geometry change.
        self.failed_routes = [record for record in self.failed_routes
                              if np.linalg.norm(agent - record[1]) < 50.
                              and np.linalg.norm(center - record[2]) < self.progress_px]
        initial = self.future is None
        index = 0
        if not initial:
            remaining = self.future
            if len(remaining) > 1 and np.linalg.norm(agent - remaining[0]) <= self.waypoint_capture_px:
                remaining = remaining[1:]
            matches = [frechet_distance(remaining, path) for path in paths]
            nearest = int(np.argmin(matches))
            if matches[nearest] <= self.threshold_px:
                index = nearest
                current = paths[index]
            else:
                # Keep a previously sampled route when this draw omits its mode.
                index = None
                current = np.concatenate((remaining, np.repeat(
                    remaining[-1:], 15 - len(remaining), axis=0)), axis=0)
        else:
            current = paths[0]
        stalled = not initial and self._stalled()
        trials = []
        cache = {}

        def test(path, candidate_index):
            key = path.tobytes()
            if key not in cache:
                cache[key] = probe(path, candidate_index)
                trials.append(dict(index=candidate_index, trial=cache[key]))
            return cache[key]

        first = test(current, index)
        groups = medoids = None
        selected, selected_index, trial = current, index, first
        switched = False
        if first.proven_empty:
            reason = 'local_domain_proven_infeasible'
        elif first.feasible and not stalled:
            reason = 'initial_first_feasible' if initial else 'continue_strategy'
        else:
            self._remember_failure(agent, center)
            self.last_search = self.step
            groups, medoids, distances = group_routes(paths, self.threshold_px)
            ordered = []
            for group, medoid in zip(groups, medoids):
                same = (index in group if index is not None
                        else frechet_distance(current, paths[medoid]) <= self.threshold_px)
                failed = any(frechet_distance(paths[medoid], old[0]) <= self.threshold_px
                             for old in self.failed_routes)
                ordered.append((same, failed, group[0], medoid))
            ordered.sort()
            reason = 'stalled_keep_safe_action' if first.feasible else 'no_incumbent_hold'
            for same, failed, _, medoid in ordered:
                # If the present strategy is safe, don't bounce back into an
                # already stalled strategy while geometry is unchanged.
                if first.feasible and (same or failed):
                    continue
                alternative = test(paths[medoid], medoid)
                if alternative.proven_empty:
                    # A feasible first trial would contradict an empty domain.
                    if not first.feasible:
                        selected, selected_index, trial = paths[medoid], medoid, alternative
                        reason = 'local_domain_proven_infeasible'
                        break
                    continue
                if alternative.feasible:
                    selected, selected_index, trial = paths[medoid], medoid, alternative
                    switched = not initial and not same
                    reason = 'switch_stall' if stalled else 'switch_infeasible'
                    break

        if trial.feasible and (initial or switched):
            self.strategy_id += 1
            self.reference = np.vstack((agent, selected))
            self.history.clear()
            self.history.append((agent.copy(), center.copy(), float(clearance)))
            self.last_search = self.step
        elif trial.feasible and self.reference is not None:
            # Extend the reference after reaching its end, so legitimate
            # repositioning beyond the first horizon still counts as progress.
            lengths = np.linalg.norm(np.diff(self.reference, axis=0), axis=1)
            if lengths.sum() - arc_position(self.reference, agent) < self.progress_px:
                self.reference = np.vstack((self.reference, selected))[-240:]
        if trial.feasible:
            self.future = selected.copy()
        return dict(path=selected, index=selected_index, trial=trial,
                    groups=groups, medoids=medoids, trials=trials,
                    reason=reason, stalled=stalled, switched=switched,
                    strategy_id=self.strategy_id)
