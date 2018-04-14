import numpy as np
from physics_sim import PhysicsSim
from tasks.task import Task

class Landing(Task):
    """Landing (environment) that defines the goal of the agent to achieve some height and give feedback on performance"""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None, reward_func=None):
        """Initialize a Landing object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        super(Landing, self).__init__(init_pose, init_velocities, init_angle_velocities, runtime, target_pos)
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.start_pos = self.sim.pose[:3]
        self.action_repeat = 3

        # state made of current position, velocity and angular velocity
        self.state_size = self.action_repeat * (6 + 3 + 3)
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.runtime = runtime
        self.base_distance = np.sqrt((init_pose[:3] - target_pos[:3])**2).sum()
        self.penalties_obj = {}
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 0.])

        self.penalties = 0
        self.reward = 0


    def get_reward(self):
        """Uses current pose of sim to return reward."""

        reward = 0
        penalties = 0
        # base_reward
        reward = self.base_distance

        residual_distances = self.sim.pose[:3]-self.target_pos[:3]
        distance = np.sqrt( (residual_distances**2).sum())

        # penalty for distance from target
        # max: 10000 ; min: 0
        pos_xy_penalty = abs(residual_distances[:2]).sum()
        pos_z_penalty = 2*abs(residual_distances[2])
        pos_penalties = pos_xy_penalty + pos_z_penalty
        penalties += pos_penalties

        self.penalties_obj['pos_xy'] = round(pos_xy_penalty,2)
        self.penalties_obj['pos_z'] = round(pos_z_penalty,2)
        self.penalties_obj['pos'] = round(pos_penalties,2)

        #penalty for euler angles, we want the takeoff to be stable
        euler_penalty = 10 * abs(self.sim.pose[3:6]).sum()
        penalties += euler_penalty
        self.penalties_obj['euler'] = round(euler_penalty,2)

        if(distance < self.base_distance * 0.5):
            # extra reward for flying near the target
            reward += self.base_distance * 0.2
            reward -= abs(self.sim.v[2])**2
            self.penalties_obj['extra_r1'] = reward
            if distance < self.base_distance * 0.1:
                reward += self.base_distance * 0.1
                reward -= abs(self.sim.v[2])**2
                self.penalties_obj['extra_r2'] = reward
