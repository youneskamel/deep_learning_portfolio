import random
from collections import deque

class Memory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, obs, obs_, done, reward, action ):
        observation = {"obs": obs, "obs_": obs_, "done": done, "reward": reward, "action": action}
        self.memory.append(observation)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)