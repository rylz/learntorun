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
    return .5*velocity - .1*knee_r_angle - .1*knee_l_angle + .5*(head_x - pelvis_x)

