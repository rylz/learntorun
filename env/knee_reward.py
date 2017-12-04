from osim.env import *
from env.augmented_run import AugRunEnv

class KneeRewardEnv(AugRunEnv):
  STATE_VELOCITY = 4 # center of mass velocity index in the state vector
  R_KNEE_ANGLE = 7
  L_KNEE_ANGLE = 10
  HEAD = 22
  PELVIS = 1
  def compute_reward(self):
    velocity = self.current_state[self.STATE_VELOCITY]
    knee_r_angle = self.current_state[self.R_KNEE_ANGLE]
    knee_l_angle = self.current_state[self.L_KNEE_ANGLE]
    pelvis_x = self.current_state[self.PELVIS]
    head_x = self.current_state[self.HEAD]
    lig_pen = 0
    for j in range(20, 26):
      lig = opensim.CoordinateLimitForce.safeDownCast(self.osim_model.forceSet.get(j))
      lig_pen += lig.calcLimitForce(self.osim_model.state) ** 2

    return velocity - .2*knee_r_angle - .2*knee_l_angle + .75*(head_x - pelvis_x) - (math.sqrt(lig_pen) * 10e-8)

