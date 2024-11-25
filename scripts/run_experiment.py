import os
import pandas as pd
import torch
import random
import numpy as np

import sympy
import getpass
import cProfile
import pstats
import sys
import os
sys.path.clear()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the pyenergyplus directory to the Python path
pyenergyplus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../OpenStudio-1.4.0/EnergyPlus'))
sys.path.append(pyenergyplus_path)
# Optional: Verify the paths
print("Updated sys.path:", sys.path)

# Optional: Verify the paths
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
        'E_factor_day', 'T_factor_day', 'E_factor_night', 'T_factor_night', 'save_idf'
    ]
    
    for key in required_keys:
        if key not in parameters:
            raise ValueError(f"Missing parameter: {key}")
    
    # Convert parameters appropriately
    parameters = {
        key: eval(value) if value.replace('.', '', 1).isdigit() else value
        for key, value in parameters.items()
    }
    
    climate_files = [
        './data/weather_data/ARIZONA.epw',
        './data/weather_data/CALIFORNIA.epw',
        './data/weather_data/DUBAI.epw',
        './data/weather_data/LDN.epw',
        './data/weather_data/CAPETOWN.epw',
        './data/weather_data/TEXAS.epw',
        './data/weather_data/SINGAPORE.epw',
        './data/weather_data/SOUTHCAROLINA.epw',
        './data/weather_data/TOKYO.epw',
        './data/weather_data/VANCOUVER.epw',
        './data/weather_data/ARIZONA_modified.epw',
        './data/weather_data/CALIFORNIA_modified.epw',
        './data/weather_data/DUBAI_modified.epw',
        './data/weather_data/LDN_modified.epw',
        './data/weather_data/CAPETOWN_modified.epw',
        './data/weather_data/SOUTHCAROLINA_modified.epw',
        './data/weather_data/SINGAPORE_modified.epw',
        './data/weather_data/TEXAS_modified.epw',
        './data/weather_data/TOKYO_modified.epw',
        './data/weather_data/VANCOUVER_modified.epw',
    ]
    
    seed_values = [52]
    num_rows_list = [100000]
    rae_files = [
        './data/weather_data/Buffer-example/20k_TX_c_SC_AZ.csv',
    ]
    
    results = []
    all_results_df = pd.DataFrame()
    
    for seed in seed_values:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"Running experiment with seed: {seed}")
        
        for num_rows in num_rows_list:
            print(f"Loading {num_rows} rows...")
            
            for rae_file in rae_files:
                print(f"Loading {rae_file}...")
                
                for weather_data in climate_files:
                    print(f"Processing weather file: {weather_data}")
                    
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
                    replay_buffer_2 = ReplayBuffer(capacity=num_rows)
                    replay_buffer_2.load_from_csv(rae_file, num_rows=num_rows)
                    
                    trainer = DQNTrainer(agent, replay_buffer, replay_buffer_2, weather_data, parameters['save_idf'], parameters)
                    trainer.setup_baseline()
                    epoch_results = trainer.train()
                    results_df = trainer.report_results()
                    all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('./data/final_results', exist_ok=True)
    all_results_df.to_csv(f'./data/final_results/all_results_{timestamp}.csv', index=False)
    print(f"Results saved to ./data/final_results/all_results_{timestamp}.csv")

if __name__ == '__main__':
    main()
