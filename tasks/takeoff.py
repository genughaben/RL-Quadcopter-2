import numpy as np
from physics_sim import PhysicsSim
from tasks.task import Task

class Takeoff(Task):
    """Takeoff (environment) that defines the goal of the agent to achieve some height and give feedback on performance"""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None, reward_func=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        super(Takeoff, self).__init__(init_pose, init_velocities, init_angle_velocities, runtime, target_pos)
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

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        self.penalties = 0
        self.reward = 0


    def get_reward(self):
        """Uses current pose of sim to return reward."""

        remain_distance = np.sqrt( ((self.sim.pose[:3]-self.target_pos)**2).sum() )
        remain_x_distance = abs(self.sim.pose[0] - self.target_pos[0])
        remain_y_distance = abs(self.sim.pose[1] - self.target_pos[1])
        remain_z_distance = abs(self.sim.pose[2] - self.target_pos[2]) #[1,0] ; 1 if distance is maximal; 0 if target is arrived at

        reward = 0
        penalties = 0
        penalties += remain_x_distance**2
        penalties += remain_y_distance**2
        penalties += 5 * remain_distance**2 # possibly remove or change to 2
        penalties += 2 * remain_z_distance**2 # possibly change to 5
        # penalty for euler angles
        penalties += abs(self.sim.pose[3:6]).sum()

        # penalty for velocity
        penalties += max(remain_x_distance + self.sim.v[0],0)**2
        penalties += max(remain_y_distance + self.sim.v[1],0)**2
        penalties += 2 * max(remain_z_distance + self.sim.v[2],0)**2 # change to 5

        # # angular velocity
        penalties += max(self.sim.pose[3] + self.sim.angular_v[0],0)**2
        penalties += max(self.sim.pose[4] + self.sim.angular_v[1],0)**2
        penalties += 2 * max(self.sim.pose[5] + self.sim.angular_v[2],0)**2 # change to 5

        penalties = penalties*0.0005
        self.penalties = penalties
        reward += 100
        # if remain_distance < 10:
        #     reward += 10
        # if self.sim.time >= self.runtime and remain_distance < 10:
        #     reward +=100

        reward = reward - penalties
        self.reward = reward
        return reward

    def original_reward():
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def reward_emphasize_z(sim_pose, target_pose):
        reward = 0
        x_diviation = abs(sim_pose[0] - target_pose[0])
        y_diviation = abs(sim_pose[1] - target_pose[1])
        z_diviation = abs(sim_pose[2] - target_pose[2])
        basic_reward = 1.
        reward = basic_reward - 0.9*z_diviation
        reward = reward - 0.05*x_diviation
        reward = reward - 0.05*y_diviation
        return reward

    def get_reward_with_orientation():
        #add small bias to allow for small euler angles
       euler_bias = 10
       Eulers_angle_penalty = abs(self.sim.pose[3:] - self.target_pos[3:]).sum() - euler_bias

       # Reward based on how close we are to our designated coordinate z. This should be our main objective
       z_reward = abs(self.sim.pose[2] - self.target_pos[2])

       # Reward agent for minimaly straying or not moving from x,y axis (more stable take off)
       other_reward = abs(self.sim.pose[:2] - self.target_pos[:2]).sum()

       penalties = (-.0003*(other_reward) - .0009*(z_reward) - .0003*(Eulers_angle_penalty))/3

       reward =   1 + penalties # penalties should be a negative number  # add 1 for every second flying
       return reward
