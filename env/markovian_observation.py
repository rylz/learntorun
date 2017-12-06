from osim.env import *
from env.augmented_run import AugRunEnv
from env.model_metadata import *

class MarkovianObsEnv(AugRunEnv):
  def __init__(self, visualize = True, max_obstacles = 3):
    super(MarkovianObsEnv, self).__init__(visualize, max_obstacles)
    ninput = self.ninput + PSOAS_L - HEAD_X
    self.observation_space = ( [-math.pi] * ninput, [math.pi] * ninput )
    self.observation_space = convert_to_gym(self.observation_space)

  def get_observation(self):
    bodies = ['head', 'pelvis', 'torso', 'toes_l', 'toes_r', 'talus_l', 'talus_r']

    pelvis_pos = [self.pelvis.getCoordinate(i).getValue(self.osim_model.state) for i in range(3)]
    pelvis_vel = [self.pelvis.getCoordinate(i).getSpeedValue(self.osim_model.state) for i in range(3)]

    jnts = ['hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l']
    joint_angles = [self.osim_model.get_joint(jnts[i]).getCoordinate().getValue(self.osim_model.state) for i in range(6)]
    joint_vel = [self.osim_model.get_joint(jnts[i]).getCoordinate().getSpeedValue(self.osim_model.state) for i in range(6)]

    mass_pos = [self.osim_model.model.calcMassCenterPosition(self.osim_model.state)[i] for i in range(2)]  
    mass_vel = [self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)[i] for i in range(2)]

    body_transforms = [[self.osim_model.get_body(body).getTransformInGround(self.osim_model.state).p()[i] for i in range(2)] for body in bodies]

    muscles = [ self.env_desc['muscles'][self.MUSCLES_PSOAS_L], self.env_desc['muscles'][self.MUSCLES_PSOAS_R] ]

    # see the next obstacle
    obstacle = self.next_obstacle()

    current_state = pelvis_pos + pelvis_vel + joint_angles + joint_vel + mass_pos + mass_vel + list(flatten(body_transforms)) + muscles + obstacle
    if hasattr(self, 'last_state'):
      body_vel = [(current_state[i] - self.last_state[i]) / 0.01 for i in range(HEAD_X, PSOAS_L)]
    else:
      body_vel = [0. for i in range(HEAD_X, PSOAS_L)]
    self.current_state = current_state + body_vel
    return self.current_state