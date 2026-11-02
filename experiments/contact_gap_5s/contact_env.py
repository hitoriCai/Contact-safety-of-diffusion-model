"""Report physical pusher-to-T contact separately from other collisions."""

from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv


class ContactPushTKeypointsEnv(PushTKeypointsEnv):
    def step(self, action):
        self.pusher_t_contact = False
        observation, reward, done, info = super().step(action)
        info["pusher_t_contact"] = bool(self.pusher_t_contact)
        return observation, reward, done, info

    def _handle_collision(self, arbiter, space, data):
        super()._handle_collision(arbiter, space, data)
        shapes = arbiter.shapes
        if (any(shape in shapes for shape in self.agent.shapes)
                and any(shape in shapes for shape in self.block.shapes)):
            self.pusher_t_contact = True
