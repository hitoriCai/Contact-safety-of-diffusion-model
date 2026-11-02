"""Track an exact control-step streak without pusher-to-T contact."""

from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper


class ContactMultiStepWrapper(MultiStepWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_contact_steps = 0

    def reset(self):
        self.no_contact_steps = 0
        return super().reset()

    def _add_info(self, info):
        super()._add_info(info)
        if info["pusher_t_contact"]:
            self.no_contact_steps = 0
        else:
            self.no_contact_steps += 1

    def get_no_contact_steps(self):
        return int(self.no_contact_steps)
