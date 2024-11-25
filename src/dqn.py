# src/dqn.py

import torch
import torch.nn.functional as F
import numpy as np
from src.models import DNN_3

class DQN:
    def __init__(self, state_dim, action_dim, learning_rate, gamma,
                 epsilon, epsilon_min, epsilon_decay, target_update, device):
        self.action_dim = action_dim
        self.q_net = DNN_3(state_dim, action_dim).to(device)  
        self.target_q_net = DNN_3(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.epsilon_min = epsilon_min  
        self.epsilon_decay = epsilon_decay  
        self.target_update = target_update  
        self.count = 0  
        self.device = device
        self.losses = []  

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            with torch.no_grad():
                action = self.q_net(state).argmax().item()
            return action

    def update(self, replay_buffer, replay_buffer_2, batch_size):
        
        half_batch_size = batch_size // 2
        
        b_s1, b_a1, b_r1, b_ns1, b_d1 = replay_buffer.sample(batch_size)
        b_s2, b_a2, b_r2, b_ns2, b_d2 = replay_buffer_2.sample(half_batch_size)

        b_s = np.concatenate((b_s1, b_s2), axis=0)
        b_r = np.concatenate((b_r1, b_r2), axis=0)
        b_ns = np.concatenate((b_ns1, b_ns2), axis=0)
        b_a = np.concatenate((b_a1, b_a2), axis=0)
        b_d = np.concatenate((b_d1, b_d2), axis=0)

        states = torch.tensor(b_s, dtype=torch.float).to(self.device)
        actions = torch.tensor(b_a).view(-1, 1).to(self.device)
        rewards = torch.tensor(b_r, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(b_ns, dtype=torch.float).to(self.device)
        dones = torch.tensor(b_d, dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = F.mse_loss(q_values, q_targets).mean()

        self.losses.append(dqn_loss.item())

        self.optimizer.zero_grad()

        dqn_loss.backward()

        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

        return dqn_loss

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)