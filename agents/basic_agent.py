import random

class Basic_Agent():
    def __init__(self, task):
        self.task = task

    def act(self, state):
        new_thrust = random.gauss(450., 25.)
        return [new_thrust + random.gauss(0., 1.) for x in range(4)]

    def reset_episode(self):
        state = self.task.reset()
        return state

    def step(self,  action, reward, next_state, done):
        pass

class Trivial_Takeoff_Agent():
    def __init__(self, task, speed = 425, noise = random.gauss(0., 1.)):
        self.task = task
        self.speed = speed
        self.noise = noise

    def act(self, state):
        new_thrust = self.speed + self.noise
        return [new_thrust for x in range(4)]

    def reset_episode(self):
        self.count = 0
        self.total_reward = 0
        state = self.task.reset()
        return state

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

    def print_final(self):
        print("total_reward: ",self.total_reward)
        print("final count: ", self.count)
