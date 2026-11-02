"""One-step local 25-px hard CBF and lower proxy mean(P0,P4,P7)."""

from dataclasses import replace

from experiments.model2_cbf_full_page.policy import FullPagePolicy
from experiments.model2_cbf_full_page.filter import _solve_page_coordinates


GROUPS = ((1, 3), (2, 3), (0, 4, 7))
class OneStepP047Policy(FullPagePolicy):
    require_matching_stage_radii = True

    def __init__(self, *args, mind=None, regrasp_model2=None, **kwargs):
        mind_config = dict(mind or {})
        radii = tuple(float(x) for x in mind_config["safety_proxy_radii"])
        mind_config.update(
            safety_keypoint_groups=[list(group) for group in GROUPS],
            safety_proxy_radii=list(radii),
            block_clearance_radius=max(radii),
        )
        recovery_config = dict(regrasp_model2 or {})
        recovery_radii = tuple(float(x) for x in recovery_config["region_mind_proxy_radii"])
        if self.require_matching_stage_radii and radii != recovery_radii:
            raise ValueError("This experiment requires matching model-1/model-2 radii")
        super().__init__(*args, mind=mind_config,
                         regrasp_model2=recovery_config, **kwargs)
        if self.mind_config.execution_action_steps != 1:
            raise ValueError("This experiment requires execution_action_steps=1")
        # Match the completed one-step experiment: original trigger windows
        # were measured in four-action replans, so preserve physical time.
        cfg = self.regrasp_config
        self.regrasp_config = replace(
            cfg,
            history_replans=cfg.history_replans * 4,
            maximum_settle_replans=cfg.maximum_settle_replans * 4,
            cooldown_replans=cfg.cooldown_replans * 4,
            maximum_trigger_action_path_length_px=(
                cfg.maximum_trigger_action_path_length_px / 4),
        )
        cfg = self.regrasp_model2_config
        self.regrasp_model2_config = replace(
            cfg,
            oscillation_persistence_replans=(
                cfg.oscillation_persistence_replans * 4),
        )
        if self.direction_switch_enabled:
            raise RuntimeError("Direction-switch scoring must remain disabled")
        for label, filter_, expected_radii in (
                ("model1", self.mind_filter, radii),
                ("model2", self._regrasp_mind_filter, recovery_radii)):
            assert tuple(tuple(x) for x in filter_.config.safety_keypoint_groups) == GROUPS
            assert tuple(filter_.config.safety_proxy_radii) == expected_radii
            if filter_.config.hard_cbf_enabled:
                assert filter_.config.safety_slack_cap_px2 is None
            elif getattr(self, 'depth_budget_enabled', False):
                assert filter_.config.safety_slack_cap_px2 is None
                assert hasattr(filter_, 'depth_max_total_px')
            elif self.bounded_soft_cbf_enabled:
                assert filter_.config.safety_slack_cap_px2 == self.bounded_soft_cbf_cap_px2
            else:
                raise RuntimeError("Unexpected soft CBF profile")
            if (filter_.config.max_action_delta_px != 25.0
                    or filter_._solve.__func__ is not _solve_page_coordinates):
                raise RuntimeError("Both models require direct local 25px CBF solves")
            if getattr(self, "depth_budget_enabled", False):
                safety = (f"soft nominal CBF + hard depth guards: "
                          f"+{filter_.depth_per_step_px:g}px/step, "
                          f"total<={filter_.depth_max_total_px:g}px")
            else:
                safety = ("hard CBF" if filter_.config.hard_cbf_enabled else
                          f"bounded soft CBF <= {filter_.config.safety_slack_cap_px2:g} px^2")
            print(f"[P047 ONE STEP] {label}: groups={GROUPS}, radii={expected_radii}, "
                  f"execute=1, direct local 25px/axis {safety}; no global solve/clip",
                  flush=True)
