import os
import pandas as pd
import torch
from src.utils import read_parameters_from_txt
from src.dqn import DQN
from src.replay_buffer import ReplayBuffer
from src.trainer import DQNTrainer

def main():
    # Read parameters
    parameters = read_parameters_from_txt('parameters.txt')
    
    # Ensure all necessary parameters are present
    required_keys = [
        'state_dim', 'action_dim', 'lr', 'gamma', 'epsilon',
        'epsilon_min', 'epsilon_decay', 'target_update', 'epochs',
        'buffer_size', 'minimal_size', 'batch_size',
        'signal_factor', 'signal_loss', 'RL_flag', 'FPS',
        'E_factor_day', 'T_factor_day', 'E_factor_night', 'T_factor_night',
        'HVAC_action_list', 'save_idf'
    ]
    
    for key in required_keys:
        if key not in parameters:
            raise ValueError(f"Missing parameter: {key}")
    
    # Convert parameters appropriately
    parameters = {
        key: eval(value) if value.replace('.', '', 1).isdigit() else value
        for key, value in parameters.items()
    }
    
    # Specify the weather file and epoch to evaluate
    weather_data = 'data/weather_data/ARIZONA.epw'  # Example
    epoch = 19  # Example epoch
    
    agent = DQN(
        state_dim=parameters['state_dim'],
        action_dim=parameters['action_dim'],
        learning_rate=parameters['lr'],
        gamma=parameters['gamma'],
        epsilon=parameters['epsilon'],
        epsilon_min=parameters['epsilon_min'],
        epsilon_decay=parameters['epsilon_decay'],
        target_update=parameters['target_update'],
        device=torch.device("mps") if torch.has_mps else torch.device("cpu")
    )
    
    replay_buffer = ReplayBuffer(capacity=parameters['buffer_size'])
    replay_buffer_2 = ReplayBuffer(capacity=100000)  # Adjust as needed
    
    trainer = DQNTrainer(agent, replay_buffer, replay_buffer_2, weather_data, parameters['save_idf'], parameters)
    trainer.load_model_weights()
    EPLUS = trainer.evaluate_model(epoch)
    
    # Save evaluation results
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f'data/final_results/evaluation_{timestamp}.csv'
    EPLUS.energy_data_to_csv(filename)
    print(f"Evaluation data saved to {filename}")

if __name__ == '__main__':
    main()
