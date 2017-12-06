from osim.env import *
from env.augmented_run import AugRunEnv

class VelocityTorsoRewardEnv(AugRunEnv):
    STATE_VELOCITY = 4 # center of mass velocity index in the state vector
    TORSO_Y = 27

    CONSTANT_REWARD = 10e-5

    def compute_reward(self):
        # Compute ligaments penalty
        lig_pen = 0
        # Get ligaments
        for j in range(20, 26):
            lig = opensim.CoordinateLimitForce.safeDownCast(self.osim_model.forceSet.get(j))
            lig_pen += lig.calcLimitForce(self.osim_model.state) ** 2

        velocity = self.current_state[self.STATE_VELOCITY]
        torso = self.current_state[self.TORSO_Y]
        return 10 * velocity - math.sqrt(lig_pen) * 10e-7 + 0.5 * torso
