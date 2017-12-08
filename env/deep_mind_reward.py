from osim.env import *
from env.augmented_run import AugRunEnv
from . import model_metadata

class DeepMindRewardEnv(AugRunEnv):
    """Deep Mind Reward Environment.

    Reward inspired by that used in "Emergence of Locomotion Behaviors in Rich Environments" from
    Heess et. al. at DeepMind. This paper uses a slightly different derivative model of PPO than
    the PPO2 implementation that we intend to use this reward function with, but the problem and
    their environment is nonetheless very similar to ours, so this is worth trying.

    """
    STATE_VELOCITY = 4
    TORSO_Y = 27
    TOES_L_Y = 29
    TOES_R_Y = 31


    def compute_reward(self):
        # Compute ligaments penalty
        lig_pen = 0
        # Get ligaments
        for j in range(20, 26):
            lig = opensim.CoordinateLimitForce.safeDownCast(self.osim_model.forceSet.get(j))
            lig_pen += lig.calcLimitForce(self.osim_model.state) ** 2

        velocity = self.current_state[model_metadata.PELVIS_X_VEL]
        torso = self.current_state[model_metadata.TORSO_Y]

        higher_foot = max(
                self.current_state[model_metadata.TOES_L_Y],
                self.current_state[model_metadata.TOES_R_Y])
        delta_h = torso - higher_foot

        # penalize the foot for deviating from "ideal" 1.2 meters from torso in vertical dimension
        threshold = abs(delta_h - 1.2)
        # heavier penalty for being very close to the torso
        indicator = 1 if delta_h < 0.3 else 0

        return 10 * velocity + 0.5 * torso - threshold - 10 * indicator - math.sqrt(lig_pen) * 10e-7
