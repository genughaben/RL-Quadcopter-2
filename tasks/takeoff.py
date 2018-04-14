import numpy as np
from physics_sim import PhysicsSim
from tasks.task import Task

class Takeoff(Task):
    """Takeoff (environment) that defines the goal of the agent to achieve some height and give feedback on performance"""
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

        self.penalties_obj = {}
        self.penalties = 0
        self.reward = 0


    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 0
        penalties = 0

        # penalty for euler angles, we want the takeoff to be stable
        # euler_penalty = 10 * abs(self.sim.pose[3:6]).sum()
        # penalties += euler_penalty
        #
        # self.penalties_obj['euler'] = round(euler_penalty,2)

        # penalty for distance from target
        residual_distances = self.sim.pose[:3]-self.target_pos[:3]
        pos_x_penalty = abs(residual_distances[0])**2
        pos_y_penalty = abs(residual_distances[1])**2
        pos_z_penalty = 10*abs(residual_distances[2])**2
        pos_penalties = pos_x_penalty + pos_y_penalty + pos_z_penalty
        penalties += pos_penalties

        self.penalties_obj['pos_x'] = round(pos_x_penalty,2)
        self.penalties_obj['pos_y'] = round(pos_y_penalty,2)
        self.penalties_obj['pos_z'] = round(pos_z_penalty,2)
        self.penalties_obj['pos'] = round(pos_penalties,2)

        # distance to velocity relationship
        velo_x_penalty = abs(abs(residual_distances[0]) - abs(self.sim.v[0]))
        velo_y_penalty = abs(abs(residual_distances[1]) - abs(self.sim.v[1]))
        velo_z_penalty = 10*abs(abs(residual_distances[2]) - abs(self.sim.v[2]))
        velo_penalties = velo_x_penalty + velo_y_penalty + velo_z_penalty
        penalties += velo_penalties

        self.penalties_obj['v_x'] = round(velo_x_penalty, 2)
        self.penalties_obj['v_y'] = round(velo_y_penalty, 2)
        self.penalties_obj['v_z'] = round(velo_z_penalty, 2)
        self.penalties_obj['velo'] = round(velo_penalties,2)

        #anguar velocity
        av_phi_penalties = abs(self.sim.pose[3] + self.sim.angular_v[0])**2
        av_theta_penalties = abs(self.sim.pose[4] + self.sim.angular_v[1])**2
        av_psi_penalties = 10 * abs(self.sim.pose[5] + self.sim.angular_v[2])**2 # change to 5
        angular_v_penalties = av_phi_penalties + av_theta_penalties + av_psi_penalties
        penalties += angular_v_penalties

        self.penalties_obj['av_p'] = av_phi_penalties
        self.penalties_obj['av_t'] = av_theta_penalties
        self.penalties_obj['av_p'] = av_psi_penalties
        self.penalties_obj['av_all'] = angular_v_penalties

        distance = np.sqrt( (residual_distances**2).sum())

        # behaviour near target
        if(self.sim.pose[2] >= self.target_pos[2]):
            # extra reward for flying near the target
            reward += 100
            if distance < 5.:
                reward += 100

        self.penalties_obj['all'] = round(penalties,2)

        #summary calculation
        factored_penalties = penalties*0.05
        # base reward
        reward += 100
        reward = reward - factored_penalties

        self.penalties = factored_penalties
        self.reward = reward
        return reward
