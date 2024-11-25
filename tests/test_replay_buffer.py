from src.replay_buffer import ReplayBuffer
import numpy as np

def test_replay_buffer():
    capacity = 5
    buffer = ReplayBuffer(capacity=capacity)
    for i in range(7):
        state = [i] * 3
        action = i % 2
        reward = float(i)
        next_state = [i + 1] * 3
        done = i % 2 == 0
        buffer.add(state, action, reward, next_state, done)
    assert buffer.size() == capacity, f"Expected buffer size {capacity}, got {buffer.size()}"
    
    batch_size = 3
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    assert len(states) == batch_size, "Sampled states length mismatch"
    assert len(actions) == batch_size, "Sampled actions length mismatch"
    assert len(rewards) == batch_size, "Sampled rewards length mismatch"
    assert len(next_states) == batch_size, "Sampled next_states length mismatch"
    assert len(dones) == batch_size, "Sampled dones length mismatch"
    print("test_replay_buffer passed")
