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
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 100.])
        self.reward_func = reward_func
        self.z_distance = abs(self.target_pos[2] - self.sim.lower_bounds[2])
        self.runtime = runtime

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # max reward = 1
        # min reward = - inf
        rel_current_runtime = self.sim.time / self.runtime # [0,1]
        rel_remaining_runtime = 1. - rel_current_runtime # [1,0]
        rel_remaining_distance = abs(self.sim.pose[2] - self.target_pos[2]) / self.z_distance #[0,1]

        # EXAMPLES
        # begin: cur_time 0, remain_time: 1. distance: 0.9 reward: 0.1 * 100 - 0*100 = -
        #
        #

        reward = 0
        # distance component
        reward += 20 * (1. - rel_remaining_distance)
        # time component
        reward += 10 * rel_current_runtime
        # bonus for reaching the target
        if(self.sim.pose[2] >= self.target_pos[2]):
             reward += 5 * (1 - rel_remaining_distance)
             done = True
        # malus for not reaching the target
        elif(self.sim.time >= self.sim.runtime):
             reward -= 5 * (1 + rel_remaining_distance)
             done = True
        if(self.reward_func):
            reward = self.reward_func(self.sim.pose[:3], self.target_pos)
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

        def euler_bias():
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
