"""Check that both model stages use the opt-in physical page domain."""

import numpy as np
from experiments.model2_cbf_full_page.filter import _solve_page_coordinates

from experiments.model2_cbf_global_then_clip.policy import GlobalThenClipPolicy


class FullPagePolicy(GlobalThenClipPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mind_config.joint_mpc_enabled:
            raise ValueError("This variant uses sequential one-step CBF solves")
        for stage, filter_ in (("model 1", self.mind_filter),
                               ("model 2", self._regrasp_mind_filter)):
            if (filter_ is None
                    or not hasattr(filter_, "training_action_min")
                    or not np.array_equal(filter_.action_min, [0.0, 0.0])
                    or not np.array_equal(filter_.action_range, [512.0, 512.0])):
                raise RuntimeError(f"{stage} did not enable full-page CBF coordinates")
        local_only = self.mind_filter._solve.__func__ is _solve_page_coordinates
        mode = ("direct local 25px/axis CBF; no global solve/clip"
                if local_only else
                "25px/axis execution limit with post-clip CBF check")
        print("[PAGE COORDINATE CBF] model 1 and model 2: page x=0..512, "
              f"y=0..512; equivalent trained dynamics; {mode}", flush=True)
