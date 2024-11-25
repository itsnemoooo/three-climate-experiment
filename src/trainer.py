import os
import copy
import datetime
import csv
import pandas as pd
import torch
import numpy as np
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

pyenergyplus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../OpenStudio-1.4.0/EnergyPlus'))
sys.path.append(pyenergyplus_path)

print("Updated sys.path:", sys.path)

from pyenergyplus.api import EnergyPlusAPI
from src.callbacks import callback_function_DQN
from src.data_bank import Data_Bank

class DQNTrainer:
    def __init__(self, agent, replay_buffer, replay_buffer_2, weather_data, filename_to_run, parameters):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.replay_buffer_2 = replay_buffer_2
        self.weather_data = weather_data
        self.filename_to_run = filename_to_run
        self.parameters = parameters
        self.api = EnergyPlusAPI()
        self.baseline_energy_consumption = {}
        self.experiment_energy_consumption = {}
        self.epochs = parameters['epochs']
        self.EPLUS = Data_Bank(parameters)
        self.FPS = parameters['FPS']
        self.RL_flag = parameters['RL_flag']
        self.epoch_results = []
        self.weather_identifier = os.path.splitext(os.path.basename(weather_data))[0]

    def setup_baseline(self):
        self.EPLUS.FPS = 10000
        self.EPLUS.RL_flag = False
        E_state = self.api.state_manager.new_state()
        self.api.runtime.set_console_output_status(E_state, False)
        self.api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
            E_state,
            lambda state: callback_function_DQN(
                state, self.api, self.EPLUS, self.agent, self.replay_buffer, self.replay_buffer_2, self.parameters
            )
        )
        print(f"Running EnergyPlus for baseline with weather file: {self.weather_data}")
        self.api.runtime.run_energyplus(E_state, ['-w', self.weather_data, '-d', 'out/', self.filename_to_run])
        print("EnergyPlus run for baseline completed.")
        self.api.state_manager.reset_state(E_state)
        self.save_energy_data_to_csv(self.EPLUS.E_HVAC_all, 'rbc')
        self.baseline_energy_consumption[self.weather_data] = {
            'E_HVAC_all_RBC': copy.deepcopy(self.EPLUS.E_HVAC_all),
            'E_Facility_all_RBC': copy.deepcopy(self.EPLUS.E_Facility)
        }

    def train_epoch(self, epoch):
        print(f"Starting train_epoch for epoch {epoch}")
        self.load_model_weights()
        print("Model weights loaded.")
        
        self.EPLUS = Data_Bank(self.parameters)
        print("Data_Bank initialized.")
        
        self.EPLUS.FPS = self.FPS
        self.EPLUS.RL_flag = True
        print(f"FPS set to {self.FPS}, RL_flag set to {self.EPLUS.RL_flag}")
        
        E_state = self.api.state_manager.new_state()
        print("New EnergyPlus state created.")
        
        self.api.runtime.set_console_output_status(E_state, False)
        print("Console output status set to False.")
        
        self.api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
            E_state,
            lambda state: callback_function_DQN(
                state, self.api, self.EPLUS, self.agent, self.replay_buffer, self.replay_buffer_2, self.parameters
            )
        )
        print("Callback function set.")
        
        print(f"Running EnergyPlus for training epoch {epoch} with weather file: {self.weather_data}")
        self.api.runtime.run_energyplus(E_state, ['-w', self.weather_data, '-d', 'out/', self.filename_to_run])
        print(f"EnergyPlus run for training epoch {epoch} completed.")
        
        self.api.state_manager.reset_state(E_state)
        print("EnergyPlus state reset.")
        
        self.api.state_manager.delete_state(E_state)
        print("EnergyPlus state deleted.")
        
        if epoch >= 0:
            self.save_model_weights(epoch)
            print(f"Model weights saved for epoch {epoch}.")
        
        E_HVAC_all_DQN = copy.deepcopy(self.EPLUS.E_HVAC_all)
        E_Facility_all_DQN = copy.deepcopy(self.EPLUS.E_Facility)
        print("Energy data copied for DQN.")
        
        self.save_energy_data_to_csv(E_HVAC_all_DQN, 'RL_based')
        print("Energy data saved to CSV.")
        
        x_sum_1_HVAC = np.sum(self.baseline_energy_consumption[self.weather_data]['E_HVAC_all_RBC'])
        x_sum_2_HVAC = np.sum(E_HVAC_all_DQN)
        print(f"Sum of baseline HVAC energy: {x_sum_1_HVAC}, Sum of DQN HVAC energy: {x_sum_2_HVAC}")
        
        if x_sum_2_HVAC != 0:
            E_save_HVAC = (x_sum_1_HVAC - x_sum_2_HVAC) / x_sum_1_HVAC
        else:
            E_save_HVAC = 0  # or handle the case appropriately
        print(f"Energy saved HVAC: {E_save_HVAC}")
        
        work_time_length = self.EPLUS.work_time.count(1)
        denominator = len(self.baseline_energy_consumption[self.weather_data]['E_HVAC_all_RBC'])
        print(f"Work time length: {work_time_length}, Denominator: {denominator}")
        
        if denominator != 0:
            work_time_length_ratio = work_time_length / denominator
        else:
            work_time_length_ratio = 0  # or handle the case appropriately
        print(f"Work time length ratio: {work_time_length_ratio}")
        
        T_violation = len(self.EPLUS.T_Violation) / len(self.EPLUS.x) if len(self.EPLUS.x) != 0 else 0
        T_violation_offset = np.mean(self.EPLUS.T_Violation) if self.EPLUS.T_Violation else 0
        print(f"T violation: {T_violation}, T violation offset: {T_violation_offset}")
        
        total_reward = sum(self.EPLUS.episode_reward)
        average_reward = total_reward / len(self.EPLUS.episode_reward) if self.EPLUS.episode_reward else 0
        print(f"Total reward: {total_reward}, Average reward: {average_reward}")

        self.epoch_results.append({
            'epoch': epoch,
            'average_reward': average_reward,
            'E_save_HVAC': E_save_HVAC,
            'work_time_length_ratio': work_time_length_ratio,
            'T_violation': T_violation,
            'T_violation_offset': T_violation_offset
        })

        print(f'Epoch {epoch}: Average reward: {average_reward}')
        print(f"Energy saved: {E_save_HVAC*100:.2f}%")
        print(f"Time violation: {T_violation*100:.2f}%")
        self.agent.decay_epsilon()

    def save_energy_data_to_csv(self, E_HVAC_all, mode):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'./data/energy_data_{mode}{timestamp}.csv'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestep', 'E_HVAC'])
            for timestep, energy in enumerate(E_HVAC_all):
                writer.writerow([timestep, energy])
        print(f'Energy data saved to {filename}')

    def save_model_weights(self, epoch):
        os.makedirs('./data/weights/weights_baseline', exist_ok=True)
        save_path_q_net = os.path.join('./data/weights/weights_baseline', f'TX_{epoch}_q_net.pth')
        save_path_target_q_net = os.path.join('./data/weights/weights_baseline', f'TX_{epoch}_target_q_net.pth')
        torch.save(self.agent.q_net.state_dict(), save_path_q_net)
        torch.save(self.agent.target_q_net.state_dict(), save_path_target_q_net)
        print(f'Model weights saved for epoch {epoch}')

    def load_model_weights(self):
        climate_identifier = os.path.splitext(os.path.basename(self.weather_data))[0].split('_')[0].upper()
        epoch = 19
        load_path_q_net = os.path.join('data/weights', f'{climate_identifier}{epoch}_comfortparams_q_net.pth')
        load_path_target_q_net = os.path.join('data/weights', f'{climate_identifier}{epoch}_comfortparams_target_q_net.pth')
        print(f"Loading Q-Net weights from: {load_path_q_net}")
        print(f"Loading Target Q-Net weights from: {load_path_target_q_net}")
        try:
            self.agent.q_net.load_state_dict(torch.load(load_path_q_net, map_location=self.agent.device))
            self.agent.target_q_net.load_state_dict(torch.load(load_path_target_q_net, map_location=self.agent.device))
            print(f"Successfully loaded weights for {climate_identifier}")
        except FileNotFoundError as e:
            print(f"Error loading weights: {e}")

    def train(self):
        self.setup_baseline()
        print("Starting training...")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            self.train_epoch(epoch)
        return self.epoch_results

    def report_results(self):
        for result in self.epoch_results:
            result['E_save_HVAC'] *= 100
            result['T_violation'] *= 100
        results_df = pd.DataFrame(self.epoch_results)
        results_df['climate_file'] = self.weather_identifier
        return results_df

    def evaluate_model(self, epoch):
        self.EPLUS.RL_flag = True
        self.load_model_weights()
        E_state = self.api.state_manager.new_state()
        self.api.runtime.set_console_output_status(E_state, True)
        self.api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
            E_state,
            lambda state: callback_function_DQN(
                state, self.api, self.EPLUS, self.agent, self.replay_buffer, self.replay_buffer_2, self.parameters
            )
        )
        print(f"Running EnergyPlus for evaluation with weather file: {self.weather_data}")
        self.api.runtime.run_energyplus(E_state, ['-w', self.weather_data, '-d', 'out/', self.filename_to_run])
        print("EnergyPlus run for evaluation completed.")
        self.api.state_manager.reset_state(E_state)
        self.api.state_manager.delete_state(E_state)
        return self.EPLUS
