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
        # initialize
        reward = 0
        penalties = 0
        base_reward = self.base_distance # base_reward

        # penalty for distance from target
        residual_distances = self.sim.pose[:3]-self.target_pos[:3]
        pos_xy_penalty = abs(residual_distances[:2]**2).sum() # maybe again squaring?
        pos_z_penalty = abs(residual_distances[2]**2)
        self.penalties_obj['pos_xy'] = round(pos_xy_penalty,2)
        self.penalties_obj['pos_z'] = round(pos_z_penalty,2)

        #penalty for euler angles, we want the takeoff to be stable
        euler_penalty = abs(self.sim.pose[3:6]).sum()
        self.penalties_obj['euler'] = round(euler_penalty,2)

        # penalty for x and y Velocity
        # distance to velocity relationship
        velo_x_penalty = abs(self.sim.v[0])
        velo_y_penalty = abs(self.sim.v[1])
        velo_z_penalty = self.sim.v[2]
        self.penalties_obj['v_x'] = round(velo_x_penalty, 2)
        self.penalties_obj['v_y'] = round(velo_y_penalty, 2)
        self.penalties_obj['v_z'] = round(velo_z_penalty, 2)

        # factoring individual penalties
        velo_penalties = velo_x_penalty + velo_y_penalty + velo_z_penalty
        pos_penalties = pos_xy_penalty + 10 * pos_z_penalty
        # angular_v_penalties = av_phi_penalties + av_theta_penalties + av_psi_penalties
        self.penalties_obj['pos'] = round(pos_penalties,2)
        self.penalties_obj['velo'] = round(velo_penalties,2)
        # self.penalties_obj['av_all'] = round(angular_v_penalties,2)

        # adding penalties
        penalties += pos_penalties
        penalties += 10 * euler_penalty**2
        penalties += velo_penalties
        # penalties += angular_v_penalties

        # behaviour near target
        distance = np.sqrt( (residual_distances**2).sum())
        if(distance < self.base_distance * 0.2):
            # extra reward for flying near the target
            # reward += base_reward**2
            penalties += round(abs(velo_z_penalty) * 2, 2)**2
            self.penalties_obj['extra_v_penalty_1'] = round(abs(velo_z_penalty) * 2, 2)**2
            # self.penalties_obj['extra_reward_1'] = reward
            if distance < self.base_distance * 0.1:
                # reward += base_reward**2
                penalties += round(abs(velo_z_penalty) * 2, 2)**2
                # self.penalties_obj['extra_reward_2'] = reward
                self.penalties_obj['extra_v_penalty_2'] = round(abs(velo_z_penalty) * 2, 2)**2

        factored_penalties = penalties * 0.01 # factoring penalties
        reward = base_reward + reward - factored_penalties
        self.penalties = factored_penalties
        self.reward = reward
        return reward
