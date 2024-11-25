import torch
from src.dqn import DQN
from src.replay_buffer import ReplayBuffer

def test_dqn():
    state_dim = 10
    action_dim = 4
    device = torch.device("cpu")
    agent = DQN(
        state_dim, action_dim, learning_rate=0.001, gamma=0.99,
        epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99,
        target_update=10, device=device
    )
    state = [0.1] * state_dim
    action = agent.take_action(state)
    assert 0 <= action < action_dim, f"Action {action} out of bounds [0, {action_dim})"
    
    # Add dummy data to replay buffer
    replay_buffer = ReplayBuffer(capacity=100)
    for i in range(20):
        replay_buffer.add(state, action, 1.0, state, False)
    
    # Update the agent
    loss = agent.update(replay_buffer, replay_buffer, batch_size=5)
    assert loss is not None, "Agent update did not return loss"
    print("test_dqn passed")
