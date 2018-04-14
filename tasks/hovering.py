import numpy as np
from physics_sim import PhysicsSim
from tasks.task import Task

class Hovering(Task):
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
        super(Hovering, self).__init__(init_pose, init_velocities, init_angle_velocities, runtime, target_pos)
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
        self.penalties_obj = {}


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
        # penalties += remain_distance # # possibly remove or change to 2
        penalties += 2*remain_z_distance**2 # possibly change to 5
        self.penalties_obj['pos_penalty'] = penalties
        # penalty for euler angles
        # penalties += abs(self.sim.pose[3:6]).sum()
        # penalties += abs(self.sim.v[0])**2
        # penalties += abs(self.sim.v[1])**2
        # penalties += 2*abs(self.sim.v[2])**2 # change to 5
        penalties += abs(remain_distance - abs(self.sim.v[:3]).sum())**2

        self.penalties_obj['velo_penalty'] = penalties
        # penalties += abs(self.sim.v[:2]).sum()/2
        # penalties += abs(self.sim.v[2])
        # penalty for velocity
        # penalties += abs(remain_x_distance + self.sim.v[0])
        # penalties += abs(remain_y_distance + self.sim.v[1])
        # penalties += abs(remain_y_distance + self.sim.v[2])

        # angular velocity
        # penalties += max(self.sim.pose[3] + self.sim.angular_v[0],0)
        # penalties += max(self.sim.pose[4] + self.sim.angular_v[1],0)
        # penalties += max(self.sim.pose[5] + self.sim.angular_v[2],0) # change to 5

        self.penalties_obj['all_penalty'] = penalties
        penalties = penalties*0.01
        self.penalties_obj['rel_penalty'] = penalties
        self.penalties = penalties

        reward += 100
        if remain_distance < 2:
            reward += 100
        if self.sim.time >= self.runtime and remain_distance < 2:
            reward +=1000

        reward -= self.penalties_obj['pos_penalty']
        reward -= self.penalties_obj['velo_penalty']
        self.reward = reward
        return reward
