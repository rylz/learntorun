from osim.env import *

class VelocityRewardEnv(RunEnv):
    STATE_VELOCITY = 4 # center of mass velocity index in the state vector

    def compute_reward(self):
        # Compute ligaments penalty
        lig_pen = 0
        # Get ligaments
        for j in range(20, 26):
            lig = opensim.CoordinateLimitForce.safeDownCast(self.osim_model.forceSet.get(j))
            lig_pen += lig.calcLimitForce(self.osim_model.state) ** 2

        velocity = self.current_state[STATE_VELOCITY]
        return velocity - math.sqrt(lig_pen) * 10e-8
