from osim.env import *
from env.augmented_run import AugRunEnv
from env.model_metadata import *

class VelocityRewardEnv(AugRunEnv):
    def compute_reward(self):
        # Compute ligaments penalty
        lig_pen = 0
        # Get ligaments
        for j in range(20, 26):
            lig = opensim.CoordinateLimitForce.safeDownCast(self.osim_model.forceSet.get(j))
            lig_pen += lig.calcLimitForce(self.osim_model.state) ** 2

        velocity = self.current_state[PELVIS_X_VEL]
        return velocity - math.sqrt(lig_pen) * 10e-8
