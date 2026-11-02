"""Switch to model 2 in the same replan when model-1 hard CBF is infeasible."""

from dataclasses import replace

import numpy as np
import torch

from experiments.model2_cbf_diverse_hard.policy import HardCBFModel2Policy


class InfeasibleRegraspPolicy(HardCBFModel2Policy):
    direction_switch_enabled = False
    # Four independent trajectories per batch, up to three batches only when
    # every trajectory in the preceding batch failed hard-CBF preview.
    candidate_batch_limit = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mind_config.execution_action_steps == 2:
            # These detectors were expressed in numbers of four-action
            # replans. Double the counts to preserve their time windows when
            # each replan now advances only two physical control steps.
            cfg = self.regrasp_config
            self.regrasp_config = replace(
                cfg,
                history_replans=cfg.history_replans * 2,
                maximum_settle_replans=cfg.maximum_settle_replans * 2,
                cooldown_replans=cfg.cooldown_replans * 2,
                maximum_trigger_action_path_length_px=(
                    cfg.maximum_trigger_action_path_length_px / 2),
            )
            recovery_cfg = self.regrasp_model2_config
            self.regrasp_model2_config = replace(
                recovery_cfg,
                oscillation_persistence_replans=(
                    recovery_cfg.oscillation_persistence_replans * 2),
            )
            print("[TWO-STEP REPLAN] execute=2, preview=2, "
                  "stall/cooldown windows preserved in physical steps",
                  flush=True)
        self._model1_infeasible_regrasp_count = 0
        self._model1_infeasible_regrasp_count_total = 0

    def reset(self):
        super().reset()
        self._model1_infeasible_regrasp_count = 0

    def predict_action(self, obs_dict, obstacles=None):
        # The parent computes model 1, then applies its hard CBF. The parent
        # result contains the exact per-stream no-incumbent signal, before any
        # of the returned actions have been sent to the environment.
        result = super().predict_action(obs_dict, obstacles)
        triggered = torch.zeros_like(result["regrasp_active"])
        result["model1_infeasible_trigger"] = triggered
        if obstacles is None:
            return result

        failed = result["mind_infeasible"].detach().cpu().numpy()
        active = result["regrasp_active"].detach().cpu().numpy()
        stream_ids = np.flatnonzero(failed & ~active).tolist()
        if not stream_ids:
            return result

        raw_obs = obs_dict["obs"][:, -1].detach().cpu().numpy()
        batch_size = raw_obs.shape[0]
        task_start = (self.n_obs_steps - 1 if self.oa_step_convention
                      else self.n_obs_steps)
        model2_policy = self._get_model2_policy()
        model2_start = (model2_policy.n_obs_steps - 1
                        if model2_policy.oa_step_convention
                        else model2_policy.n_obs_steps)
        action_count = int(self.mind_config.execution_action_steps)

        for stream_id in stream_ids:
            points = raw_obs[stream_id, :18].reshape(9, 2)
            obstacle, radius = self._obstacle_for_stream(
                obstacles, stream_id, batch_size)
            obstacle = np.asarray(obstacle).reshape(-1, 2)
            radius = np.asarray(radius).reshape(-1)
            center, clearance = self._geometry(
                points, obstacle[0], radius[0])
            self._model2_activation_counts[stream_id] = (
                self._model2_activation_counts.get(stream_id, 0) + 1)
            self._start_model2(stream_id, center, clearance, {
                "trigger_kind": "model1_infeasible",
                "net_motion": 0.0,
                "goal_progress": 0.0,
            })
            self._regrasp_active_flags[stream_id] = True
            triggered[stream_id] = True
            self._model1_infeasible_regrasp_count += 1
            self._model1_infeasible_regrasp_count_total += 1
            # The parent cached the model-1 hold as a main-task action chunk.
            # It was never executed, so exclude it from later stall triggers.
            for history in (self._main_mind_action_path_histories,
                            self._main_mind_action_target_histories):
                if stream_id in history and history[stream_id]:
                    history[stream_id].pop()

        # One batched model-2 call draws four independent diffusion candidates
        # per affected stream. Candidate previews do not mutate the live CBF.
        plans = self._sample_model2(
            obs_dict, obstacles, stream_ids, batch_size)
        for row, stream_id in enumerate(stream_ids):
            agent = raw_obs[stream_id, 18:20]
            obstacle, radius = self._obstacle_for_stream(
                obstacles, stream_id, batch_size)
            obstacle = np.asarray(obstacle).reshape(-1, 2)
            radius = np.asarray(radius).reshape(-1)
            raw = plans[row, model2_start:model2_start + action_count, :2]
            if raw.shape[0] != action_count:
                raise RuntimeError(
                    "Model 2 produced fewer actions than the execution chunk")
            limited = self._limit_regrasp_action_deltas(raw, agent)
            projected, statuses = self._regrasp_mind_filter.project_sequence(
                limited.detach().cpu().numpy(),
                raw_obs[stream_id, :18], obstacle, radius,
                stream_id=stream_id, fallback_action=agent,
                safety_modes=self.mind_safety_modes_for_stream(
                    stream_id, action_count),
            )
            projected = torch.as_tensor(
                projected, device=result["action"].device,
                dtype=result["action"].dtype)
            end = task_start + action_count
            result["action"][stream_id, :action_count, :2] = projected
            result["action_pred"][stream_id, task_start:end, :2] = projected
            result["diffusion_action_pred"][stream_id, task_start:end, :2] = (
                limited.to(device=projected.device, dtype=projected.dtype))
            result["mind_projected_action_chunk"][stream_id, :action_count,
                                                   :2] = projected
            result["mind_infeasible"][stream_id] = any(
                status.endswith("_agent_hold") or "no_solution" in status.lower()
                for status in statuses)
            for key in ("regrasp_active", "regrasp_force_continue",
                        "regrasp_model2_active"):
                result[key][stream_id] = True
            visualization = self._regrasp_mind_filter.last_visualization_data(
                stream_id)
            if visualization is not None and "mind_predicted_center" in result:
                result["mind_predicted_center"][stream_id] = torch.as_tensor(
                    visualization["center_next"], device=projected.device,
                    dtype=projected.dtype)
            self._model2_plans[stream_id] = (task_start, limited.detach().clone())
            self._regrasp_modes[stream_id]["executed_steps"] += action_count
            print(
                f"[MODEL1 INFEASIBLE -> REGRASP] env={stream_id} "
                f"model2_statuses={statuses}", flush=True)
        return result
