from osim.env import *

class EnvWithShapedReward(RunEnv):
	@override
	def compute_reward(...):
    knee_r_angle = self.current_state[7]
    knee_l_angle = self.current_state[10]
	  pelvis_velocity = self.current_state[4]
		return pelvis_velocity -.005*knee_r_angle -.005*knee_l_angle
