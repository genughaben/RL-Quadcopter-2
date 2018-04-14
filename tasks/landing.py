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

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 0.])

        self.penalties = 0
        self.reward = 0


    def get_reward(self):
        """Uses current pose of sim to return reward."""



        remain_distance = np.sqrt( ((self.sim.pose[:3]-self.target_pos)**2).sum() )
        remain_x_distance = self.target_pos[0] - self.sim.pose[0]
        remain_y_distance = self.target_pos[1] - self.sim.pose[1]
        remain_z_distance = self.target_pos[2] - self.sim.pose[2]

        reward = 0
        penalties = 0
        penalties += remain_x_distance**2
        penalties += remain_y_distance**2
        penalties += 5 * remain_z_distance**2 # possibly change to 5
        penalties += remain_distance**2
        # penalties += remain_distance**2 # possibly remove or change to 2
        # penalties += self.sim.time * abs(remain_x_distance)
        # penalties += self.sim.time * abs(remain_y_distance)
        penalties += self.sim.time * abs(remain_z_distance)
        # penalty for euler angles
        penalties += abs(self.sim.pose[3:6]).sum()

        # penalty for velocity
        penalties += abs(remain_x_distance + self.sim.v[0])**2
        penalties += abs(remain_y_distance + self.sim.v[1])**2
        penalties += abs(remain_z_distance + self.sim.v[2])**2 # change to 5
        # penalties += self.sim.time * abs(self.sim.v[0])**2
        # penalties += self.sim.time * abs(self.sim.v[1])**2
        # penalties += self.sim.time * abs(self.sim.v[2])**2

        # # angular velocity
        # penalties += max(self.sim.pose[3] + self.sim.angular_v[0],0)**2
        # penalties += max(self.sim.pose[4] + self.sim.angular_v[1],0)**2
        # penalties += 2 * max(self.sim.pose[5] + self.sim.angular_v[2],0)**2 # change to 5

        penalties = penalties*0.001
        self.penalties = penalties
        reward += 100
        if remain_distance < 2:
            reward += 10
        if self.sim.time >= self.runtime:
            reward +=100

        reward = reward - penalties
        self.reward = reward
        return reward

        # reward = 0
        # penalties = 0
        # current_position = self.sim.pose[:3]
        # # penalty for euler angles, we want the takeoff to be stable
        # penalties += abs(self.sim.pose[3:6]).sum()
        # # penalty for distance from target
        # penalties += abs(current_position[0]-self.target_pos[0])**2
        # penalties += abs(current_position[1]-self.target_pos[1])**2
        # penalties += 10*abs(current_position[2]-self.target_pos[2])**2
        #
        # # link velocity to residual distance
        # penalties += abs(abs(current_position-self.target_pos).sum() - abs(self.sim.v).sum())
        #
        # distance = np.sqrt((current_position[0]-self.target_pos[0])**2 + (current_position[1]-self.target_pos[1])**2 + (current_position[2]-self.target_pos[2])**2)
        # # extra reward for flying near the target
        # if distance < 10:
        #     reward += 1000
        # # constant reward for flying
        # reward += 100
        # self.penalties = penalties
        # reward = reward - penalties*0.0002
        #
        # self.reward = reward
        # return reward
