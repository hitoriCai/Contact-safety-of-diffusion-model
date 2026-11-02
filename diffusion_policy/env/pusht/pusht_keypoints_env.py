from typing import Dict, Sequence, Union, Optional
import pathlib
from gym import spaces
from diffusion_policy.env.pusht.pusht_env import PushTEnv
from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
import numpy as np
import cv2

class PushTKeypointsEnv(PushTEnv):
    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            keypoint_visible_rate=1.0, 
            agent_keypoints=False,
            draw_keypoints=True,  # False
            reset_to_state=None,
            safety_stress_obstacles=False,
            render_action=True,
            local_keypoint_map: Dict[str, np.ndarray]=None, 
            color_map: Optional[Dict[str, np.ndarray]]=None):
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            reset_to_state=reset_to_state,
            safety_stress_obstacles=safety_stress_obstacles,
            render_action=render_action)
        ws = self.window_size

        if local_keypoint_map is None:
            # create default keypoint definition
            kp_kwargs = self.genenerate_keypoint_manager_params()
            local_keypoint_map = kp_kwargs['local_keypoint_map']
            color_map = kp_kwargs['color_map']

        # create observation spaces
        Dblockkps = np.prod(local_keypoint_map['block'].shape)
        Dagentkps = np.prod(local_keypoint_map['agent'].shape)
        Dagentpos = 2

        Do = Dblockkps
        if agent_keypoints:
            # blockkp + agnet_pos
            Do += Dagentkps
        else:
            # blockkp + agnet_kp
            Do += Dagentpos
        # obs + obs_mask
        Dobs = Do * 2

        low = np.zeros((Dobs,), dtype=np.float64)
        high = np.full_like(low, ws)
        # mask range 0-1
        high[Do:] = 1.

        # (block_kps+agent_kps, xy+confidence)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float64
        )

        self.keypoint_visible_rate = keypoint_visible_rate
        self.agent_keypoints = agent_keypoints
        self.draw_keypoints = draw_keypoints
        self.kp_manager = PymunkKeypointManager(
            local_keypoint_map=local_keypoint_map,
            color_map=color_map)
        self.draw_kp_map = None
        self.mind_clearance_radius = None
        self.mind_clearance_scale = 1.0
        self.mind_center_keypoint_indices = (0, 3)
        self.mind_safety_keypoint_groups = None
        self.mind_safety_proxy_radii = None
        self.diffusion_trajectory = None
        self.mind_optimized_trajectory = None
        self.mind_optimized_trajectory_start = None
        self.mind_used_action = None
        self.mind_predicted_center = None
        self.mind_diagnostic_data = None
        self.mind_infeasible = False
        self.regrasp_active = False
        self.regrasp_target = None
        self.regrasp_label = None

    def set_mind_visualization(self, clearance_radius, clearance_scale,
            center_keypoint_indices, infeasible=False,
            safety_keypoint_groups=None, safety_proxy_radii=None):
        """Set overlays for the next recorded frame without changing dynamics."""
        self.mind_clearance_radius = (
            None if clearance_radius is None else float(clearance_radius))
        self.mind_clearance_scale = float(clearance_scale)
        self.mind_center_keypoint_indices = tuple(
            int(index) for index in center_keypoint_indices)
        self.mind_safety_keypoint_groups = (
            None if safety_keypoint_groups is None else tuple(
                tuple(int(index) for index in group)
                for group in safety_keypoint_groups))
        self.mind_safety_proxy_radii = (
            None if safety_proxy_radii is None else tuple(
                float(radius) for radius in safety_proxy_radii))
        self.mind_infeasible = bool(infeasible)

    def set_diffusion_trajectory(self, trajectory):
        """Set the raw diffusion action plan to render on the next frame."""
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.ndim != 2 or trajectory.shape[1] < 2:
            raise ValueError('diffusion trajectory must have shape (T, >=2).')
        self.diffusion_trajectory = trajectory[:, :2].copy()

    def set_mind_optimized_trajectory(self, trajectory, start_position):
        """Set the sequential MIND-projected chunk for the video overlay."""
        if trajectory is None:
            self.mind_optimized_trajectory = None
            self.mind_optimized_trajectory_start = None
            return
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.ndim != 2 or trajectory.shape[1] < 2:
            raise ValueError('MIND trajectory must have shape (T, >=2).')
        self.mind_optimized_trajectory = trajectory[:, :2].copy()
        self.mind_optimized_trajectory_start = np.asarray(
            start_position, dtype=np.float64).reshape(2).copy()

    def set_mind_step_overlay(self, used_action, predicted_center):
        """Set MIND's executed action and NNDM T-center prediction."""
        self.mind_used_action = np.asarray(
            used_action, dtype=np.float64).reshape(2).copy()
        if predicted_center is None:
            self.mind_predicted_center = None
        else:
            predicted_center = np.asarray(predicted_center, dtype=np.float64)
            self.mind_predicted_center = (
                None if not np.all(np.isfinite(predicted_center))
                else predicted_center.reshape(2).copy())

    def set_mind_diagnostic_data(self, diagnostic_data):
        self.mind_diagnostic_data = (
            None if diagnostic_data is None else dict(diagnostic_data))

    def set_regrasp_active(self, active):
        """Show whether mirror regrasp is controlling the current chunk."""
        self.regrasp_active = bool(active)

    def set_regrasp_label(self, label):
        """Set a policy-specific recovery annotation for the video overlay."""
        self.regrasp_label = None if label is None else str(label)

    def set_regrasp_target(self, target):
        """Show the active temporary T-center target without changing physics."""
        if target is None:
            self.regrasp_target = None
            return
        target = np.asarray(target, dtype=np.float64).reshape(2)
        self.regrasp_target = (
            target.copy() if np.all(np.isfinite(target)) else None)

    def save_mind_diagnostic_image(self, path, scale=2):
        """Write a high-resolution scene plus a separate MIND info panel."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The recorded video is intentionally 96x96.  Render stills from the
        # native 512px PyGame canvas instead, so diagnostic text is readable.
        previous_render_size = self.render_size
        try:
            self.render_size = max(512, int(previous_render_size))
            frame = self._render_frame('rgb_array')
        finally:
            self.render_size = previous_render_size
        if scale != 1:
            frame = cv2.resize(
                frame, None, fx=float(scale), fy=float(scale),
                interpolation=cv2.INTER_NEAREST)

        panel_width = max(310, frame.shape[1] // 2)
        panel = np.full((frame.shape[0], panel_width, 3), 246, dtype=np.uint8)
        cv2.putText(panel, 'MIND step-1 diagnostics', (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, color=(30, 30, 30),
                    thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(panel, 'cyan: raw diffusion plan', (16, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color=(0, 150, 150),
                    thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(panel, 'green X: executed MIND action', (16, 87),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color=(0, 140, 0),
                    thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(panel, 'red X: NNDM next T center', (16, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color=(190, 20, 20),
                    thickness=1, lineType=cv2.LINE_AA)
        if self.mind_diagnostic_data is not None:
            data = self.mind_diagnostic_data
            raw_action = data.get('raw_action', (np.nan, np.nan))
            used_action = data.get('used_action', (np.nan, np.nan))
            predicted_center = data.get('center_next', (np.nan, np.nan))
            lines = [
                'status: %s   solutions: %s' % (
                    data.get('status', '?'), data.get('solutions', '?')),
                'u_raw:  (%.1f, %.1f)' % (raw_action[0], raw_action[1]),
                'u_MIND: (%.1f, %.1f)' % (used_action[0], used_action[1]),
                '|u_MIND-u_raw|_1: %.3f' % data.get('action_l1_change', np.nan),
                'T_NNDM_next: (%.1f, %.1f)' % (
                    predicted_center[0], predicted_center[1]),
                'distance: %.1f -> %.1f  (R=%.1f)' % (
                    data.get('distance_now', np.nan),
                    data.get('distance_next', np.nan),
                    data.get('safe_radius', np.nan)),
                'phi: %.1f -> %.1f' % (
                    data.get('phi_now', np.nan), data.get('phi_next', np.nan)),
                'slack: %.4g' % (
                    np.nan if data.get('safety_slack') is None
                    else data.get('safety_slack')),
                'solve: %.3fs, nodes: %s' % (
                    data.get('runtime_sec', np.nan), data.get('nodes', '?')),
            ]
            for index, line in enumerate(lines):
                cv2.putText(panel, line, (16, 152 + index * 31),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            color=(30, 30, 30), thickness=1,
                            lineType=cv2.LINE_AA)
        composite = np.concatenate((frame, panel), axis=1)
        if not cv2.imwrite(str(path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)):
            raise IOError(f'Could not save MIND diagnostic image: {path}')

    @classmethod
    def genenerate_keypoint_manager_params(cls):
        env = PushTEnv()
        kp_manager = PymunkKeypointManager.create_from_pusht_env(env)
        kp_kwargs = kp_manager.kwargs
        return kp_kwargs

    def _get_obs(self):
        # get keypoints
        obj_map = {
            'block': self.block
        }
        if self.agent_keypoints:
            obj_map['agent'] = self.agent

        kp_map = self.kp_manager.get_keypoints_global(
            pose_map=obj_map, is_obj=True)
        # python dict guerentee order of keys and values
        kps = np.concatenate(list(kp_map.values()), axis=0)

        # select keypoints to drop
        n_kps = kps.shape[0]
        visible_kps = self.np_random.random(size=(n_kps,)) < self.keypoint_visible_rate
        kps_mask = np.repeat(visible_kps[:,None], 2, axis=1)

        # save keypoints for rendering
        vis_kps = kps.copy()
        vis_kps[~visible_kps] = 0
        draw_kp_map = {
            'block': vis_kps[:len(kp_map['block'])]
        }
        if self.agent_keypoints:
            draw_kp_map['agent'] = vis_kps[len(kp_map['block']):]
        self.draw_kp_map = draw_kp_map
        
        # construct obs
        obs = kps.flatten()
        obs_mask = kps_mask.flatten()
        if not self.agent_keypoints:
            # passing agent position when keypoints are not available
            agent_pos = np.array(self.agent.position)
            obs = np.concatenate([
                obs, agent_pos
            ])
            obs_mask = np.concatenate([
                obs_mask, np.ones((2,), dtype=bool)
            ])

        # obs, obs_mask
        obs = np.concatenate([
            obs, obs_mask.astype(obs.dtype)
        ], axis=0)
        return obs
    
    
    def _render_frame(self, mode):
        img = super()._render_frame(mode)
        if self.draw_keypoints:
            self.kp_manager.draw_keypoints(
                img, self.draw_kp_map, radius=int(img.shape[0]/96))

        if self.mind_clearance_radius is not None:
            # Draw the same proxy-center definitions used by the active MIND MIP.
            block_keypoints = self.kp_manager.get_keypoints_global(
                pose_map={'block': self.block}, is_obj=True)['block']
            scale_xy = np.array([img.shape[1], img.shape[0]], dtype=np.float64) \
                / float(self.window_size)
            groups = self.mind_safety_keypoint_groups
            radii = self.mind_safety_proxy_radii
            if groups is None or radii is None:
                groups = (self.mind_center_keypoint_indices,)
                radii = (self.mind_clearance_radius,)
            for group, proxy_radius in zip(groups, radii):
                center = block_keypoints[list(group)].mean(axis=0)
                center_px = np.rint(center * scale_xy).astype(np.int32)
                radius_px = max(1, int(round(
                    proxy_radius * float(scale_xy.mean()))))
                cv2.circle(
                    img, tuple(center_px), radius_px,
                    color=(255, 165, 0),
                    thickness=max(1, img.shape[0] // 64))
                cv2.circle(
                    img, tuple(center_px), max(1, img.shape[0] // 128),
                    color=(255, 165, 0), thickness=-1)

            # Forbidden-region radii remain active in MIND but are no longer
            # drawn as purple circles in any recorded evaluation video.

        # Cyan path and dots are the raw diffusion action sequence before the
        # MIND projection. The larger first dot is the action MIND may alter.
        if self.diffusion_trajectory is not None:
            scale_xy = np.array([img.shape[1], img.shape[0]], dtype=np.float64) \
                / float(self.window_size)
            plan_px = np.rint(self.diffusion_trajectory * scale_xy).astype(np.int32)
            if len(plan_px) >= 2:
                cv2.polylines(img, [plan_px.reshape(-1, 1, 2)], False,
                              color=(0, 255, 255),
                              thickness=max(1, img.shape[0] // 96),
                              lineType=cv2.LINE_AA)
            for index, point in enumerate(plan_px):
                radius = max(1, img.shape[0] // (72 if index == 0 else 112))
                cv2.circle(img, tuple(point), radius, color=(0, 255, 255),
                           thickness=-1, lineType=cv2.LINE_AA)

        # Orange path and dots are the 4 actions after sequential
        # MIND projection. It begins at the measured pusher position, unlike
        # the cyan raw diffusion plan which shows action targets only.
        if (self.mind_optimized_trajectory is not None
                and self.mind_optimized_trajectory_start is not None):
            scale_xy = np.array([img.shape[1], img.shape[0]], dtype=np.float64) \
                / float(self.window_size)
            trajectory = np.vstack([
                self.mind_optimized_trajectory_start.reshape(1, 2),
                self.mind_optimized_trajectory])
            trajectory_px = np.rint(trajectory * scale_xy).astype(np.int32)
            cv2.polylines(img, [trajectory_px.reshape(-1, 1, 2)], False,
                          color=(0, 165, 255),
                          thickness=max(1, img.shape[0] // 88),
                          lineType=cv2.LINE_AA)
            for point in trajectory_px[1:]:
                cv2.circle(img, tuple(point), max(2, img.shape[0] // 96),
                           color=(0, 165, 255), thickness=-1,
                           lineType=cv2.LINE_AA)

        scale_xy = np.array([img.shape[1], img.shape[0]], dtype=np.float64) \
            / float(self.window_size)
        # Green X: MIND-corrected action that is actually sent to the env.
        if self.mind_used_action is not None:
            point = np.rint(self.mind_used_action * scale_xy).astype(np.int32)
            cv2.drawMarker(img, tuple(point), color=(0, 220, 0),
                           markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=max(7, img.shape[0] // 18),
                           thickness=max(1, img.shape[0] // 96),
                           line_type=cv2.LINE_AA)
        # Red X: NNDM prediction of the next T proxy center after that action.
        if self.mind_predicted_center is not None:
            point = np.rint(
                self.mind_predicted_center * scale_xy).astype(np.int32)
            cv2.drawMarker(img, tuple(point), color=(230, 30, 30),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=max(7, img.shape[0] // 18),
                           thickness=max(1, img.shape[0] // 96),
                           line_type=cv2.LINE_AA)

        if self.mind_infeasible:
            banner_height = max(18, img.shape[0] // 5)
            cv2.rectangle(
                img, (0, 0), (img.shape[1] - 1, banner_height),
                color=(220, 0, 0), thickness=-1)
            font_scale = max(0.36, img.shape[0] / 240.0)
            cv2.putText(
                img, 'INFEASIBLE', (4, banner_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color=(255, 255, 255), thickness=max(1, img.shape[0] // 96),
                lineType=cv2.LINE_AA)
        if self.regrasp_active:
            if self.regrasp_target is not None:
                point = np.rint(
                    self.regrasp_target * scale_xy).astype(np.int32)
                radius = max(5, img.shape[0] // 28)
                thickness = max(1, img.shape[0] // 96)
                # Yellow bullseye: current recovery target for the T center.
                cv2.circle(
                    img, tuple(point), radius, color=(255, 215, 0),
                    thickness=thickness, lineType=cv2.LINE_AA)
                cv2.drawMarker(
                    img, tuple(point), color=(255, 215, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=radius * 2 + 3, thickness=thickness,
                    line_type=cv2.LINE_AA)
                label_position = (
                    min(point[0] + radius + 2, img.shape[1] - 35),
                    max(point[1] - radius - 2, 10))
                cv2.putText(
                    img, 'target', label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.30, img.shape[0] / 300.0),
                    color=(255, 215, 0), thickness=thickness,
                    lineType=cv2.LINE_AA)
            label = self.regrasp_label or 'on'
            font_scale = (max(0.30, img.shape[0] / 360.0)
                          if self.regrasp_label else
                          max(0.42, img.shape[0] / 220.0))
            cv2.putText(
                img, label, (4, img.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color=(0, 210, 0), thickness=max(1, img.shape[0] // 80),
                lineType=cv2.LINE_AA)
        return img
