import numpy as np
import torch as T
from q_network import Qnetwork
from memory import Memory

class Agent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_shape,
                 mem_size, batch_size, epsilon_min=0.01, epsilon_dec=5e-7,
                 replace_target_cnt=1000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.replace_target_cnt = replace_target_cnt
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = Memory(mem_size)

        self.q_eval = Qnetwork(self.lr, self.n_actions,
                                    input_shape=self.input_shape)

        self.q_next = Qnetwork(self.lr, self.n_actions,
                                    input_shape=self.input_shape)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation]),dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)   
        return action     
        
    def store_transition(self, state, action, reward, state_, done):
        self.memory.push(state, state_, done, reward, action)

    def uniform_sample_memory(self):
        obs, obs_, done, reward, action = [], [], [], [], []
        batch = self.memory.sample(self.batch_size)
        for example in batch :
            obs.append(example["obs"])
            obs_.append(example["obs_"])
            done.append(example["done"])
            reward.append(example["reward"])
            action.append(example["action"])
        reward = T.tensor(np.array(reward), dtype=T.float32).to(self.q_eval.device)
        obs = T.tensor(np.array(obs), dtype=T.float32).to(self.q_eval.device)
        obs_ = T.tensor(np.array(obs_), dtype=T.float32).to(self.q_eval.device)
        action = T.tensor(np.array(action), dtype=T.int64).to(self.q_eval.device)
        done = T.tensor(np.array(done), dtype=T.bool).to(self.q_eval.device)

        return obs, obs_, done, reward, action

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_min :
            self.epsilon = self.epsilon - self.epsilon_dec
        else : self.epsilon_min

    def learn(self):
        if self.memory.__len__() < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, states_, dones, rewards, actions = self.uniform_sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_eval_next_action = self.q_eval.forward(states_).argmax(dim=1)[0]
        q_next = self.q_next.forward(states_)[indices, q_eval_next_action]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)

        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
