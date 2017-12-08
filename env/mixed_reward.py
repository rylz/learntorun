from osim.env import *
from env.augmented_run import AugRunEnv

class MixedRewardEnv(AugRunEnv):
    STATE_PELVIS_X = 1
    STATE_PELVIS_Y = 2
    STATE_VELOCITY = 4
    HEAD_X = 22
    TORSO_Y = 27

    def compute_reward(self):
        # Compute ligaments penalty
        lig_pen = 0
        # Get ligaments
        for j in range(20, 26):
            lig = opensim.CoordinateLimitForce.safeDownCast(self.osim_model.forceSet.get(j))
            lig_pen += lig.calcLimitForce(self.osim_model.state) ** 2

        velocity = self.current_state[self.STATE_VELOCITY]
        torso = self.current_state[self.TORSO_Y]
        delta_x = self.current_state[self.STATE_PELVIS_X] - self.last_state[self.STATE_PELVIS_X]
        lean_backwards = min(0,self.current_state[self.HEAD_X] - self.current_state[self.STATE_PELVIS_X])
        constant_reward = 10e-6

        return 7 * velocity + 3 * delta_x - math.sqrt(lig_pen) * 10e-7 + 0.5 * torso + lean_backwards + constant_reward
