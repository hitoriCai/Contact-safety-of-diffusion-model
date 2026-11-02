from typing import Dict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.policy.mind import (
    MindConfig,
    MindSafetyFilter,
    create_mind_safety_filter,
    load_dynamics_checkpoint,
)

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            mind=None,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        self.mind_config = MindConfig.from_mapping(mind)
        if self.mind_config.enabled:
            mind_config_source = "config" if mind is not None else "fallback_defaults"
            print(
                "[MIND CONFIG] "
                f"source={mind_config_source} "
                f"clearance_scale={self.mind_config.clearance_scale:g} "
                f"block_clearance_radius={self.mind_config.block_clearance_radius:g}",
                flush=True,
            )
        self.mind_filter = None
        if self.mind_config.enabled:
            dynamics_model, dynamics_normalization = load_dynamics_checkpoint(
                self.mind_config.checkpoint_path)
            # The standalone NNDM is immediately converted to NumPy weights by
            # the filter.  It is intentionally not registered as part of the
            # diffusion policy, so old diffusion checkpoints remain loadable
            # and the NNDM is never updated by the policy optimizer.
            self.mind_filter = create_mind_safety_filter(
                dynamics_model, dynamics_normalization, self.mind_config)

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def reset(self):
        """Reset MIND's previous-action state at an episode boundary."""
        if self.mind_filter is not None:
            self.mind_filter.reset()

    def transform_diffusion_action_plan(
            self, action_pred, obs_dict, obstacles=None):
        """Optional inference-only plan hook used by specialized policies.

        The standard policy is intentionally an identity transform. This
        keeps train.py and eval.py bit-for-bit equivalent while allowing an
        independent recovery policy to replace a short execution chunk before
        the existing MIND projection is applied.
        """
        return action_pred

    def mind_safety_modes(self, batch_index, action_start, action_end):
        """Return optional per-action MIND modes for a transformed chunk.

        ``None`` preserves strict recovery for every action. Specialized
        inference policies can request a temporary nonworsening barrier for
        selected pusher-only transfer actions.
        """
        return None

    def mind_projection_enabled(self):
        """Allow inference-only subclasses to retain the NNDM but skip MIND."""
        return (
            self.mind_filter is not None
            and not bool(getattr(self, "skip_mind_projection", False)))

    def mind_projection_mask(self, batch_size):
        """Return which vectorized environments should receive MIND.

        The default preserves the original all-environment projection.  A
        specialized inference policy may opt one environment out temporarily
        (for example, while a separately trained recovery controller owns it)
        without disabling MIND for the rest of the batch.
        """
        return np.ones(int(batch_size), dtype=np.bool_)

    def mind_filter_for_stream(self, stream_id):
        """Return the safety filter that owns one vectorized environment."""
        return self.mind_filter

    def mind_visualization_data(self, stream_id):
        """Expose the matching filter's diagnostics to the video runner."""
        return self.mind_filter_for_stream(stream_id).last_visualization_data(
            stream_id)

    def mind_safety_modes_for_stream(self, stream_id, action_count):
        """Optional per-stream MIND mode sequence; ``None`` uses defaults."""
        return None
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], obstacles=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        # Preserve the raw diffusion plan for diagnostic video overlays. MIND
        # may replace the first action in ``action_pred`` below.
        diffusion_action_pred = action_pred.detach().clone()
        action_pred = self.transform_diffusion_action_plan(
            action_pred, obs_dict, obstacles)

        # MIND is an online safety controller, not part of training. The
        # standard path projects each leading action with an independent
        # one-step MIP. The opt-in joint-MPC path optimizes the whole chunk
        # and its predicted NNDM trajectory in one MIP.
        mind_infeasible = np.zeros(B, dtype=np.bool_)
        mind_visualization = [None] * B
        mind_action_slice = None
        mind_projected_mask = np.zeros(B, dtype=np.bool_)
        if obstacles is not None and self.mind_projection_enabled():
            action_start = To - 1 if self.oa_step_convention else To
            action_end = min(
                             action_start + self.mind_config.execution_action_steps,
                             action_pred.shape[1])
            if action_start < action_end:
                mind_action_slice = (action_start, action_end)
                action_np = action_pred.detach().cpu().numpy().copy()
                current_obs_np = obs_dict['obs'][:, -1].detach().cpu().numpy()
                state_np = current_obs_np[:, :18]
                agent_position_np = current_obs_np[:, 18:20]
                batch_size = action_np.shape[0]
                projection_mask = np.asarray(
                    self.mind_projection_mask(batch_size), dtype=np.bool_)
                if projection_mask.shape != (batch_size,):
                    raise ValueError(
                        'mind_projection_mask must return one flag per '
                        'vectorized environment.')
                projection_indices = np.flatnonzero(projection_mask)

                positions = obstacles['pos']
                radii = obstacles['r']
                # The runner supplies one obstacle set per vectorized env.
                # A single array is also accepted for direct one-env calls.
                per_batch = isinstance(positions, (list, tuple))
                if not per_batch:
                    positions_array = np.asarray(positions)
                    per_batch = positions_array.ndim == 3 and positions_array.shape[0] == batch_size
                def project_one(batch_index):
                    filter_ = self.mind_filter_for_stream(batch_index)
                    batch_positions = positions[batch_index] if per_batch else positions
                    batch_radii = radii[batch_index] if per_batch else radii
                    action_chunk = action_np[
                        batch_index, action_start:action_end, :2]
                    if self.mind_config.joint_mpc_enabled:
                        if not isinstance(filter_, MindSafetyFilter):
                            raise TypeError(
                                'Joint MPC requires the legacy ReLU NNDM '
                                'MindSafetyFilter.')
                        projected, statuses = filter_.project_horizon(
                            action_chunk, state_np[batch_index],
                            batch_positions, batch_radii,
                            stream_id=batch_index,
                            fallback_action=agent_position_np[batch_index],
                        )
                    else:
                        projected, statuses = filter_.project_sequence(
                            action_chunk, state_np[batch_index],
                            batch_positions, batch_radii,
                            stream_id=batch_index,
                            fallback_action=agent_position_np[batch_index],
                            safety_modes=self.mind_safety_modes_for_stream(
                                batch_index, action_end - action_start),
                        )
                    return (batch_index, projected, statuses,
                            filter_.last_visualization_data(
                                batch_index))

                if len(projection_indices) > 0:
                    worker_count = min(
                        int(self.mind_config.parallel_workers),
                        len(projection_indices))
                    if worker_count == 1:
                        projection_results = list(map(
                            project_one, projection_indices))
                    else:
                        with ThreadPoolExecutor(
                                max_workers=worker_count) as executor:
                            projection_results = list(executor.map(
                                project_one, projection_indices))
                    for batch_index, projected, statuses, visualization in projection_results:
                        action_np[batch_index, action_start:action_end, :2] = projected
                        mind_infeasible[batch_index] = any(
                            (status.endswith("_agent_hold")
                             or "NO_SOLUTION" in status)
                            for status in statuses)
                        mind_visualization[batch_index] = visualization
                    mind_projected_mask = projection_mask
                    # Test the original diffusion policy without MIND:
                    # 1. Comment out the self.mind_filter.project(...) block above.
                    # 2. Uncomment all three lines below. They override any
                    #    solver status left over from the previous rollout step.
                    # projected = action_np[batch_index, action_index, :2].copy()
                    # status = "disabled"
                    # mind_infeasible[batch_index] = False
                action_pred = torch.as_tensor(
                    action_np, device=naction_pred.device, dtype=naction_pred.dtype)
                naction_pred = self.normalizer['action'].normalize(action_pred)
                nsample[..., :Da] = naction_pred
        # Skipped-MIND logging is temporarily disabled. Keep this block for
        # convenient re-enabling when checking policy/runner integration.
        # elif self.mind_config.print_status:
        #     reason = "no_obstacle_data" if obstacles is None else "mind_disabled"
        #     print(f"[MIND] skipped ({reason})", flush=True)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            execution_steps = self.n_action_steps
            if self.mind_projection_enabled():
                execution_steps = self.mind_config.execution_action_steps
            end = start + execution_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'diffusion_action_pred': diffusion_action_pred,
            # Per-environment flag used only by evaluation-video overlays.
            'mind_infeasible': torch.as_tensor(
                mind_infeasible, device=action.device, dtype=torch.bool),
            'mind_projected_mask': torch.as_tensor(
                mind_projected_mask, device=action.device, dtype=torch.bool),
        }
        # Sequentially projected execution chunk, retained separately from
        # the full action plan for evaluation-video overlays.
        if mind_action_slice is not None:
            mind_start, mind_end = mind_action_slice
            result['mind_projected_action_chunk'] = action_pred[
                :, mind_start:mind_end, :2].detach().clone()
        if any(item is not None for item in mind_visualization):
            centers = np.full((B, 2), np.nan, dtype=np.float32)
            for index, item in enumerate(mind_visualization):
                if item is not None:
                    centers[index] = item['center_next']
            result['mind_predicted_center'] = torch.as_tensor(
                centers, device=action.device, dtype=action.dtype)
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
