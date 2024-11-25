# src/replay_buffer.py

import os
import csv
import random
import numpy as np
from collections import deque
import pandas as pd

class ReplayBuffer:
    def __init__(self, capacity, save_path):
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        self.transition_count = 0  
        self.buffer = deque(maxlen=capacity)  
        self.capacity = capacity
        self.save_path = save_path

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.transition_count += 1
        self.save_to_csv(state, action, reward, next_state, done)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

    def save_to_csv(self, state, action, reward, next_state, done):
        with open(self.save_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([state, action, reward, next_state, done])

    def load_from_csv(self, file_path, num_rows=None):
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if num_rows is not None:
                df = df.head(num_rows)
            for _, row in df.iterrows():
                state = np.array(eval(row['state']))
                action = row['action']
                reward = row['reward']
                next_state = np.array(eval(row['next_state']))
                done = row['done']
                self.buffer.append((state, action, reward, next_state, done))
        else:
            print(f"[ERROR] No historical data found at {file_path}")