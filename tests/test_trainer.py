import torch
from src.dqn import DQN
from src.replay_buffer import ReplayBuffer
from src.trainer import DQNTrainer
from src.utils import read_parameters_from_txt

def test_trainer():
    # Dummy parameters
    parameters = {
        'state_dim': 10,
        'action_dim': 4,
        'lr': 0.001,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.1,
        'epsilon_decay': 0.99,
        'target_update': 10,
        'epochs': 1,
        'buffer_size': 100,
        'minimal_size': 10,
        'batch_size': 5,
        'signal_factor': 0.0,
        'signal_loss': False,
        'RL_flag': True,
        'FPS': 1,
        'E_factor_day': 1.0,
        'T_factor_day': 1.0,
        'E_factor_night': 1.0,
        'T_factor_night': 1.0,
        'HVAC_action_list': [[0, 1], [1, 0]],
        'save_idf': 'dummy_file.idf'
    }
    
    agent = DQN(
        state_dim=parameters['state_dim'], 
        action_dim=parameters['action_dim'],
        learning_rate=parameters['lr'], 
        gamma=parameters['gamma'],
        epsilon=parameters['epsilon'], 
        epsilon_min=parameters['epsilon_min'],
        epsilon_decay=parameters['epsilon_decay'], 
        target_update=parameters['target_update'],
        device=torch.device("cpu")
    )
    
    replay_buffer = ReplayBuffer(capacity=parameters['buffer_size'])
    replay_buffer_2 = ReplayBuffer(capacity=parameters['buffer_size'])
    weather_data = './data/weather_data/ARIZONA.epw'
    filename_to_run = 'dummy_file.idf'
    
    trainer = DQNTrainer(
        agent, 
        replay_buffer, 
        replay_buffer_2, 
        weather_data, 
        filename_to_run, 
        parameters
    )
    
    # Since EnergyPlus API interactions are complex, we'll mock them or skip for this test
    # Here we'll just test initialization
    assert trainer.agent == agent, "Trainer agent mismatch"
    assert trainer.replay_buffer == replay_buffer, "Trainer replay_buffer mismatch"
    print("test_trainer passed")
