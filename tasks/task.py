import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
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
        penalties += max(remain_x_distance + self.sim.v[0],0)
        penalties += max(remain_y_distance + self.sim.v[1],0)
        penalties += 2 * max(remain_z_distance + self.sim.v[2],0) # change to 5

        # # angular velocity
        penalties += max(self.sim.pose[3] + self.sim.angular_v[0],0)
        penalties += max(self.sim.pose[4] + self.sim.angular_v[1],0)
        penalties += 2 * max(self.sim.pose[5] + self.sim.angular_v[2],0) # change to 5

        penalties = penalties*0.0005
        self.penalties = penalties
        reward += 100
        if remain_distance < 10:
            reward += 10
        if self.sim.time >= self.runtime and remain_distance < 10:
            reward +=100

        reward = reward - penalties
        self.reward = reward
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            state = self.current_state()
            pose_all.append(self.current_state())
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def current_state(self):
        """The state contains information about current position, velocity and angular velocity"""
        state = np.concatenate([np.array(self.sim.pose), np.array(self.sim.v), np.array(self.sim.angular_v)])
        return state

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.current_state()] * self.action_repeat)
        return state
