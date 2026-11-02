"""Origin-style PushT diffusion policy used for baseline training rollouts.

This adapter deliberately exposes the same no-obstacle ``predict_action`` API
as the original Diffusion Policy code.  It reuses the unchanged diffusion
sampling and loss implementation, but forces MIND off so no NNDM, MIP safety
filter, infeasibility fallback, or recovery state is constructed.
"""

from typing import Dict

import torch

from diffusion_policy.policy.diffusion_unet_lowdim_policy import (
    DiffusionUnetLowdimPolicy,
)


class OriginDiffusionUnetLowdimPolicy(DiffusionUnetLowdimPolicy):
    """Pure task-completion policy with the original one-argument API."""

    def __init__(self, *args, **kwargs):
        # A checkpoint/config may carry a MIND stanza.  Ignore it here so this
        # policy remains a strict raw-diffusion rollout path.
        kwargs.pop('mind', None)
        super().__init__(*args, mind={'enabled': False}, **kwargs)

    def reset(self):
        """Origin policy has no safety-controller state to reset."""

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return super().predict_action(obs_dict, obstacles=None)
