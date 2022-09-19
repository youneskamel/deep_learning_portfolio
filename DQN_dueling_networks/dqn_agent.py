import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from memory import Memory

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = Memory(mem_size)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

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
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

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
