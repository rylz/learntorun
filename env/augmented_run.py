from osim.env import *

class AugRunEnv(RunEnv):
    def compute_metric(self):
        # Compute ligaments penalty
        lig_pen = 0
        # Get ligaments
        for j in range(20, 26):
            lig = opensim.CoordinateLimitForce.safeDownCast(self.osim_model.forceSet.get(j))
            lig_pen += lig.calcLimitForce(self.osim_model.state) ** 2

        # Get the pelvis X delta
        delta_x = self.current_state[self.STATE_PELVIS_X] - self.last_state[self.STATE_PELVIS_X]

        return delta_x - math.sqrt(lig_pen) * 10e-8

    def step(self, action):
        res = super(RunEnv, self)._step(action)
        res[3]['metric'] = self.compute_metric()
        return res;
