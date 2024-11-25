import os
import time
import copy
import shutil
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import csv
import sys

sys.path.insert(0, './OpenStudio-1.4.0/EnergyPlus')

from pyenergyplus.api import EnergyPlusAPI
import pyenergyplus

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'


device = torch.device("mps")


def temp_c_to_f(temp_c):#: float, arbitrary_arg=None):
    """Convert temp from C to F. Test function with arbitrary argument, for example."""
    return 1.8 * temp_c + 32

def temp_f_to_c(temp_f):
    return (temp_f-32)/1.8

class DNN_3(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DNN_3, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(state_dim, 512))
        self.layer5 = nn.Sequential(nn.Linear(512, 512))
        self.layer6 = nn.Sequential(nn.Linear(512, action_dim))
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer5(x))
        x = self.layer6(x)        
        return x
    
class ReplayBuffer:
    def __init__(self, capacity):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_path = f'./colab_visualizations/prior{timestamp}.csv'
        self.transition_count = 0  
        self.buffer = deque(maxlen=capacity)  
        self.secondary_buffer = deque(maxlen=capacity)  # Secondary buffer for loaded data
        self.capacity = capacity
        if not os.path.exists(self.save_path):
            with open(self.save_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):  # add data to buffer
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
            df = df.head(num_rows)
            for row in df.itertuples(index=False):
                state = np.array(eval(row.state))
                action = row.action
                reward = row.reward
                next_state = np.array(eval(row.next_state))
                done = row.done
                self.buffer.append((state, action, reward, next_state, done))
        else:
            print(f"[ERROR] No historical data found at {file_path}")

class DQN:
    def __init__(self, state_dim, action_dim, learning_rate, gamma,
                 epsilon, epsilon_min, epsilon_decay, target_update, device):
        self.action_dim = action_dim
        self.q_net = DNN_3(state_dim, action_dim).to(device)  
        self.target_q_net = DNN_3(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # discout
        self.epsilon = epsilon  # epsilon-greedy
        self.epsilon_min = epsilon_min  # minimum epsilon
        self.epsilon_decay = epsilon_decay  # decay rate
        self.target_update = target_update  # target update period
        self.count = 0  # record updates
        self.device = device
        self.losses = []  # To store loss values

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    def update(self, replay_buffer,replay_buffer_2, batch_size):

        half_batch_size = batch_size // 2
        
        #b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
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
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

        return dqn_loss
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def HVAC_action(action, temp):
    if action == 0:
        H_new = F_bottom
        C_new = F_top
    elif action == 1:
        H_new = temp[0]
        C_new = temp[1]
    return int(H_new), int(C_new)

class Data_Bank:
    def __init__(self, parameters):
        self.view_distance = 2000
        self.NUM_HVAC = 7
        self.FPS = parameters['FPS']
        self.E_factor_day = float(parameters['E_factor_day'])
        self.T_factor_day = float(parameters['T_factor_day'])
        self.E_factor_night = float(parameters['E_factor_night'])
        self.T_factor_night = float(parameters['T_factor_night'])
        self.episode_reward = 0
        self.episode_return = 0
        self.RL_flag = bool(parameters['RL_flag'])
        self.time_interval = 0
        self.x = []
        self.years = []
        self.months = []
        self.days = []
        self.hours = []
        self.minutes = []
        self.current_times = []
        self.actual_date_times = []
        self.actual_times = []
        self.weekday = []
        self.isweekday = []
        self.isweekend = []
        self.work_time = []
        self.time_line = []
        self.T_Violation = []
        self.score = []
        self.T_diff = []
        self.T_mean = []
        self.T_var = []
        self.T_map = {}
        self.y_humd = []
        self.y_wind = []
        self.y_solar = []
        self.y_zone_humd = []
        self.y_zone_window = []
        self.y_zone_ventmass = []
        self.y_zone_temp = []
        self.y_outdoor = []
        self.y_zone = []
        self.y_htg = []
        self.y_clg = []
        self.y_zone_temp_2001 = []
        self.y_zone_temp_2002 = []
        self.y_zone_temp_2003 = []
        self.y_zone_temp_2004 = []
        self.y_zone_temp_2005 = []
        self.y_zone_temp_2006 = []
        self.sun_is_up = []
        self.is_raining = []
        self.outdoor_humidity = []
        self.wind_speed = []
        self.diffuse_solar = []
        self.E_Facility = []
        self.E_HVAC = []
        self.E_Heating = []
        self.E_Cooling = []
        self.E_HVAC_all = []
        self.action_list = []
        self.episode_reward = []
        self.hvac_htg_2001 = []
        self.hvac_clg_2001 = []
        self.hvac_htg_2002 = []
        self.hvac_clg_2002 = []
        self.hvac_htg_2003 = []
        self.hvac_clg_2003 = []
        self.hvac_htg_2004 = []
        self.hvac_clg_2004 = []
        self.hvac_htg_2005 = []
        self.hvac_clg_2005 = []
        self.hvac_htg_2006 = []
        self.hvac_clg_2006 = []
        self.initialize_handles()

    def initialize_handles(self):
        self.got_handles = False
        self.oa_temp_handle = -1
        self.oa_humd_handle = -1
        self.oa_windspeed_handle = -1
        self.oa_winddirct_handle = -1
        self.oa_solar_azi_handle = -1
        self.oa_solar_alt_handle = -1
        self.oa_solar_ang_handle = -1
        self.zone_temp_handle = -1
        self.zone_htg_tstat_handle = -1
        self.zone_clg_tstat_handle = -1
        self.zone_humd_handle_2001 = -1
        self.zone_humd_handle_2002 = -1
        self.zone_humd_handle_2003 = -1
        self.zone_humd_handle_2004 = -1
        self.zone_humd_handle_2005 = -1
        self.zone_humd_handle_2006 = -1
        self.zone_window_handle_2001 = -1
        self.zone_window_handle_2002 = -1
        self.zone_window_handle_2003 = -1
        self.zone_window_handle_2004 = -1
        self.zone_window_handle_2005 = -1
        self.zone_window_handle_2006 = -1
        self.zone_ventmass_handle_2001 = -1
        self.zone_ventmass_handle_2002 = -1
        self.zone_ventmass_handle_2003 = -1
        self.zone_ventmass_handle_2004 = -1
        self.zone_ventmass_handle_2005 = -1
        self.zone_ventmass_handle_2006 = -1
        self.zone_temp_handle_2001 = -1
        self.zone_temp_handle_2002 = -1
        self.zone_temp_handle_2003 = -1
        self.zone_temp_handle_2004 = -1
        self.zone_temp_handle_2005 = -1
        self.zone_temp_handle_2006 = -1
        self.hvac_htg_2001_handle = -1
        self.hvac_clg_2001_handle = -1
        self.hvac_htg_2002_handle = -1
        self.hvac_clg_2002_handle = -1
        self.hvac_htg_2003_handle = -1
        self.hvac_clg_2003_handle = -1
        self.hvac_htg_2004_handle = -1
        self.hvac_clg_2004_handle = -1
        self.hvac_htg_2005_handle = -1
        self.hvac_clg_2005_handle = -1
        self.hvac_htg_2006_handle = -1
        self.hvac_clg_2006_handle = -1
        self.E_Facility_handle = -1
        self.E_HVAC_handle = -1
        self.E_Heating_handle = -1
        self.E_Cooling_handle = -1

    def handle_availability(self):
        self.handle_list = [
            self.oa_humd_handle,
            self.oa_windspeed_handle,
            self.oa_winddirct_handle,
            self.oa_solar_azi_handle,
            self.oa_solar_alt_handle,
            self.oa_solar_ang_handle,
            self.oa_temp_handle,
            self.zone_temp_handle,
            self.zone_temp_handle_2001,
            self.zone_temp_handle_2002,
            self.zone_temp_handle_2003,
            self.zone_temp_handle_2004,
            self.zone_temp_handle_2005,
            self.zone_temp_handle_2006,
            self.hvac_htg_2001_handle,
            self.hvac_clg_2001_handle,
            self.hvac_htg_2002_handle,
            self.hvac_clg_2002_handle,
            self.hvac_htg_2003_handle,
            self.hvac_clg_2003_handle,
            self.hvac_htg_2004_handle,
            self.hvac_clg_2004_handle,
            self.hvac_htg_2005_handle,
            self.hvac_clg_2005_handle,
            self.hvac_htg_2006_handle,
            self.hvac_clg_2006_handle,
            self.E_Facility_handle,
            self.E_HVAC_handle,
            self.E_Heating_handle,
            self.E_Cooling_handle
        ]
        return self.handle_list

HVAC_action_list = []
for HC_1 in [0,1]:
    for HC_2 in [0,1]:
        for HC_3 in [0,1]:
            for HC_4 in [0,1]:
                for HC_5 in [0,1]:
                    for HC_6 in [0,1]:
                        HVAC_action_list.append([HC_1,HC_2,HC_3,HC_4,HC_5,HC_6])

class DQNTrainer:
    def __init__(self, agent, replay_buffer, weather_data, filename_to_run, parameters):
        self.agent = agent
        self.replay_buffer = replay_buffer
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
        self.api.runtime.callback_begin_zone_timestep_after_init_heat_balance(E_state, lambda state: callback_function_DQN(state, self.api, self.EPLUS))
        self.api.runtime.run_energyplus(E_state, ['-w', self.weather_data, '-d', 'out/', self.filename_to_run])
        self.api.state_manager.reset_state(E_state)
        self.save_energy_data_to_csv(self.EPLUS.E_HVAC_all, 'rbc')
        self.baseline_energy_consumption[self.weather_data] = {
            'E_HVAC_all_RBC': self.EPLUS.E_HVAC_all[:],
            'E_Facility_all_RBC': self.EPLUS.E_Facility[:]
        }

    def train_epoch(self, epoch):
        self.load_model_weights(self.weather_data)
        self.EPLUS = Data_Bank(self.parameters)
        self.EPLUS.FPS = self.FPS
        self.EPLUS.RL_flag = True
        E_state = self.api.state_manager.new_state()
        
        self.api.runtime.set_console_output_status(E_state, False)
        self.api.runtime.callback_begin_zone_timestep_after_init_heat_balance(E_state, lambda state: callback_function_DQN(state, self.api, self.EPLUS))
        self.api.runtime.run_energyplus(E_state, ['-w', self.weather_data, '-d', 'out/', self.filename_to_run])
        self.api.state_manager.reset_state(E_state)
        self.api.state_manager.delete_state(E_state)
        if epoch > 0:
            self.save_model_weights(epoch)
        else:
           self.save_model_weights(epoch+1)
        E_HVAC_all_DQN = copy.deepcopy(self.EPLUS.E_HVAC_all)
        E_Facility_all_DQN = copy.deepcopy(self.EPLUS.E_Facility)
        self.save_energy_data_to_csv(E_HVAC_all_DQN, 'RL_based')

        x_sum_1_HVAC = np.sum(self.baseline_energy_consumption[self.weather_data]['E_HVAC_all_RBC'])
        x_sum_2_HVAC = np.sum(E_HVAC_all_DQN)
        E_save_HVAC = (x_sum_1_HVAC - x_sum_2_HVAC) / x_sum_1_HVAC
        work_time_length = self.EPLUS.work_time.count(1)
        work_time_length_ratio = work_time_length / len(self.baseline_energy_consumption[self.weather_data]['E_HVAC_all_RBC'])
        T_violation = len(self.EPLUS.T_Violation) / len(self.EPLUS.x) if len(self.EPLUS.x) != 0 else 0
        T_violation_offset = np.mean(self.EPLUS.T_Violation)
        total_reward = sum(self.EPLUS.episode_reward)
        average_reward = total_reward / len(self.EPLUS.episode_reward)
        self.epoch_results.append({
            'epoch': epoch,
            'average_reward': average_reward,
            'E_save_HVAC': E_save_HVAC,
            'work_time_length_ratio': work_time_length_ratio,
            'T_violation': T_violation,
            'T_violation_offset': T_violation_offset
        })
        print(f'Average reward: {average_reward}')
        print("Energy saved: ", E_save_HVAC*100, "%")
        print("Time violation: ", T_violation*100, "%")
        self.agent.decay_epsilon()
    def save_energy_data_to_csv(self, E_HVAC_all, mode):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'energy_data_{mode}_{timestamp}.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestep', 'E_HVAC'])
            for timestep, energy in enumerate(E_HVAC_all):
                writer.writerow([timestep, energy])
        print(f'Energy data saved to {filename}')

    def save_model_weights(self, epoch):
        save_path_q_net = os.path.join('./weights/weights_baseline', f'TX_{epoch}_q_net.pth')
        save_path_target_q_net = os.path.join('./weights/weights_baseline', f'TX_{epoch}_target_q_net.pth')
        torch.save(self.agent.q_net.state_dict(), save_path_q_net)
        torch.save(self.agent.target_q_net.state_dict(), save_path_target_q_net)

    def load_model_weights(self, weather_data):
        # Extract the climate identifier from the weather_data filename
        climate_identifier = os.path.splitext(os.path.basename(weather_data))[0].split('_')[0].upper()
        
        # Define the epoch number you want to load
        epoch = 19
        
        # Construct the file paths for the 19th epoch weights, including the climate identifier
        load_path_q_net = os.path.join('./world_weights', f'{climate_identifier}_{epoch}_comfortparams_q_net.pth')
        load_path_target_q_net = os.path.join('./world_weights', f'{climate_identifier}_{epoch}_comfortparams_target_q_net.pth')
        
        # Print the paths to verify
        print(f"Loading Q-Net weights from: {load_path_q_net}")
        print(f"Loading Target Q-Net weights from: {load_path_target_q_net}")
        
        # Load the weights into the networks
        try:
            self.agent.q_net.load_state_dict(torch.load(load_path_q_net))
            self.agent.target_q_net.load_state_dict(torch.load(load_path_target_q_net))
            print(f"Successfully loaded weights for {climate_identifier}")
        except FileNotFoundError as e:
            print(f"Error loading weights: {e}")


    def train(self):
        self.setup_baseline()
       #print("Training baseline...")
        all_epochs_losses = []
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            all_epochs_losses.extend(self.agent.losses)
            self.agent.losses = []
        self.save_losses_to_csv(all_epochs_losses)
        return self.epoch_results 
    
    def save_losses_to_csv(self, losses):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'./final_results/prior_l{timestamp}.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Loss'])
            for step, loss in enumerate(losses):
                writer.writerow([step, loss])
        print(f'Losses saved to {filename}')
    def report_results(self):
        for result in self.epoch_results:
            result['E_save_HVAC'] *= 100 
            result['T_violation'] *= 100 
        results_df = pd.DataFrame(self.epoch_results)       
        results_df['climate_file'] = self.weather_identifier
        return results_df 

    def evaluate_model(self):
        self.EPLUS.RL_flag = True  # Set to true to use RL actions
        self.load_model_weights(19)  # Load the weights for the specified epoch
        E_state = self.api.state_manager.new_state()
        self.api.runtime.set_console_output_status(E_state, False)
        self.api.runtime.callback_begin_zone_timestep_after_init_heat_balance(E_state, lambda state: callback_function_DQN(state, self.api, self.EPLUS))
        self.api.runtime.run_energyplus(E_state, ['-w', self.weather_data, '-d', 'out/', self.filename_to_run])
        self.api.state_manager.reset_state(E_state)
        self.api.state_manager.delete_state(E_state)
        return self.EPLUS

def callback_function_DQN(state_argument, api, EPLUS):
    start_time = time.time()
    RL_flag = EPLUS.RL_flag
    view_distance = EPLUS.view_distance
    time_interval = EPLUS.time_interval
    NUM_HVAC = EPLUS.NUM_HVAC
    FPS = EPLUS.FPS
    T_factor_day = EPLUS.T_factor_day
    E_factor_day = EPLUS.E_factor_day
    T_factor_night = EPLUS.T_factor_night
    E_factor_night = EPLUS.E_factor_night
    
    if not EPLUS.got_handles:
        if not api.exchange.api_data_fully_ready(state_argument):
            return
        EPLUS.oa_temp_handle = api.exchange.get_variable_handle(state_argument, u"SITE OUTDOOR AIR DRYBULB TEMPERATURE", u"ENVIRONMENT")
        EPLUS.oa_humd_handle = api.exchange.get_variable_handle(state_argument, u"Site Outdoor Air Drybulb Temperature", u"ENVIRONMENT")
        EPLUS.oa_windspeed_handle = api.exchange.get_variable_handle(state_argument, u"Site Wind Speed", u"ENVIRONMENT")
        EPLUS.oa_winddirct_handle = api.exchange.get_variable_handle(state_argument, u"Site Wind Direction", u"ENVIRONMENT")
        EPLUS.oa_solar_azi_handle = api.exchange.get_variable_handle(state_argument, u"Site Solar Azimuth Angle", u"ENVIRONMENT")
        EPLUS.oa_solar_alt_handle = api.exchange.get_variable_handle(state_argument, u"Site Solar Altitude Angle", u"ENVIRONMENT")
        EPLUS.oa_solar_ang_handle = api.exchange.get_variable_handle(state_argument, u"Site Solar Hour Angle", u"ENVIRONMENT")
        EPLUS.zone_temp_handle = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature", 'Thermal Zone 2')
        EPLUS.zone_humd_handle_2001 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity", 'Thermal Zone 1')
        EPLUS.zone_humd_handle_2002 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity", 'Thermal Zone 2')
        EPLUS.zone_humd_handle_2003 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity", 'Thermal Zone 3')
        EPLUS.zone_humd_handle_2004 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity", 'Thermal Zone 4')
        EPLUS.zone_humd_handle_2005 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity", 'Thermal Zone 5')
        EPLUS.zone_humd_handle_2006 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity", 'Thermal Zone 6')
        EPLUS.zone_window_handle_2001 = api.exchange.get_variable_handle(state_argument, "Zone Windows Total Heat Gain Energy", 'Thermal Zone 1')
        EPLUS.zone_window_handle_2002 = api.exchange.get_variable_handle(state_argument, "Zone Windows Total Heat Gain Energy", 'Thermal Zone 2')
        EPLUS.zone_window_handle_2003 = api.exchange.get_variable_handle(state_argument, "Zone Windows Total Heat Gain Energy", 'Thermal Zone 3')
        EPLUS.zone_window_handle_2004 = api.exchange.get_variable_handle(state_argument, "Zone Windows Total Heat Gain Energy", 'Thermal Zone 4')
        EPLUS.zone_window_handle_2005 = api.exchange.get_variable_handle(state_argument, "Zone Windows Total Heat Gain Energy", 'Thermal Zone 5')
        EPLUS.zone_window_handle_2006 = api.exchange.get_variable_handle(state_argument, "Zone Windows Total Heat Gain Energy", 'Thermal Zone 6')
        EPLUS.zone_ventmass_handle_2001 = api.exchange.get_variable_handle(state_argument, "Zone Mechanical Ventilation Mass", 'Thermal Zone 1')
        EPLUS.zone_ventmass_handle_2002 = api.exchange.get_variable_handle(state_argument, "Zone Mechanical Ventilation Mass", 'Thermal Zone 2')
        EPLUS.zone_ventmass_handle_2003 = api.exchange.get_variable_handle(state_argument, "Zone Mechanical Ventilation Mass", 'Thermal Zone 3')
        EPLUS.zone_ventmass_handle_2004 = api.exchange.get_variable_handle(state_argument, "Zone Mechanical Ventilation Mass", 'Thermal Zone 4')
        EPLUS.zone_ventmass_handle_2005 = api.exchange.get_variable_handle(state_argument, "Zone Mechanical Ventilation Mass", 'Thermal Zone 5')
        EPLUS.zone_ventmass_handle_2006 = api.exchange.get_variable_handle(state_argument, "Zone Mechanical Ventilation Mass", 'Thermal Zone 6')
        EPLUS.zone_temp_handle_2001 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature", 'Thermal Zone 1')
        EPLUS.zone_temp_handle_2002 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature", 'Thermal Zone 2')
        EPLUS.zone_temp_handle_2003 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature", 'Thermal Zone 3')
        EPLUS.zone_temp_handle_2004 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature", 'Thermal Zone 4')
        EPLUS.zone_temp_handle_2005 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature", 'Thermal Zone 5')
        EPLUS.zone_temp_handle_2006 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature", 'Thermal Zone 6')     
        EPLUS.hvac_htg_2001_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 1')
        EPLUS.hvac_clg_2001_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 1')
        EPLUS.hvac_htg_2002_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 2')
        EPLUS.hvac_clg_2002_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 2')
        EPLUS.hvac_htg_2003_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 3')
        EPLUS.hvac_clg_2003_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 3')
        EPLUS.hvac_htg_2004_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 4')
        EPLUS.hvac_clg_2004_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 4')
        EPLUS.hvac_htg_2005_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 5')
        EPLUS.hvac_clg_2005_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 5')
        EPLUS.hvac_htg_2006_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Heating Setpoint', 'Thermal Zone 6')
        EPLUS.hvac_clg_2006_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control', 'Cooling Setpoint', 'Thermal Zone 6')
        EPLUS.E_Facility_handle = api.exchange.get_meter_handle(state_argument, 'Electricity:Facility')
        EPLUS.E_HVAC_handle = api.exchange.get_meter_handle(state_argument, 'Electricity:HVAC')
        EPLUS.E_Heating_handle = api.exchange.get_meter_handle(state_argument, 'Heating:Electricity')
        EPLUS.E_Cooling_handle = api.exchange.get_meter_handle(state_argument, 'Cooling:Electricity')
        handle_list = EPLUS.handle_availability()
        if -1 in handle_list:
            print("***Invalid handles, check spelling and sensor/actuator availability")
            sys.exit(1)
        EPLUS.got_handles = True
    if api.exchange.warmup_flag(state_argument):
        return
    ''' Time '''
    year = api.exchange.year(state_argument)
    month = api.exchange.month(state_argument)
    day = api.exchange.day_of_month(state_argument)
    hour = api.exchange.hour(state_argument)
    minute = api.exchange.minutes(state_argument)
    current_time = api.exchange.current_time(state_argument)
    actual_date_time = api.exchange.actual_date_time(state_argument)
    actual_time = api.exchange.actual_time(state_argument)
    time_step = api.exchange.zone_time_step_number(state_argument)
    '''Temperature'''
    oa_humd = api.exchange.get_variable_value(state_argument, EPLUS.oa_humd_handle)
    oa_windspeed = api.exchange.get_variable_value(state_argument, EPLUS.oa_windspeed_handle)
    oa_winddirct = api.exchange.get_variable_value(state_argument, EPLUS.oa_winddirct_handle)
    oa_solar_azi = api.exchange.get_variable_value(state_argument, EPLUS.oa_solar_azi_handle)
    oa_solar_alt = api.exchange.get_variable_value(state_argument, EPLUS.oa_solar_alt_handle)
    oa_solar_ang = api.exchange.get_variable_value(state_argument, EPLUS.oa_solar_ang_handle)
    oa_temp = api.exchange.get_variable_value(state_argument, EPLUS.oa_temp_handle)
    zone_temp = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle)
    zone_temp_2001 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2001)
    zone_temp_2002 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2002)
    zone_temp_2003 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2003)
    zone_temp_2004 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2004)
    zone_temp_2005 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2005)
    zone_temp_2006 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2006)
    hvac_htg_2001 = api.exchange.get_actuator_value(state_argument, EPLUS.hvac_htg_2001_handle)
    hvac_clg_2001 = api.exchange.get_actuator_value(state_argument, EPLUS.hvac_clg_2001_handle)
    hvac_htg_2002 = api.exchange.get_actuator_value(state_argument, EPLUS.hvac_htg_2002_handle)
    hvac_clg_2002 = api.exchange.get_actuator_value(state_argument, EPLUS.hvac_clg_2002_handle)
    hvac_htg_2003 = api.exchange.get_actuator_value(state_argument, EPLUS.hvac_htg_2003_handle)
    hvac_clg_2003 = api.exchange.get_actuator_value(state_argument, EPLUS.hvac_clg_2003_handle)
    hvac_htg_2004 = api.exchange.get_actuator_value(state_argument, EPLUS.hvac_htg_2004_handle)
    hvac_clg_2004 = api.exchange.get_actuator_value(state_argument, EPLUS.hvac_clg_2004_handle)
    hvac_htg_2005 = api.exchange.get_actuator_value(state_argument, EPLUS.hvac_htg_2005_handle)
    hvac_clg_2005 = api.exchange.get_actuator_value(state_argument, EPLUS.hvac_clg_2005_handle)
    hvac_htg_2006 = api.exchange.get_actuator_value(state_argument, EPLUS.hvac_htg_2006_handle)
    hvac_clg_2006 = api.exchange.get_actuator_value(state_argument, EPLUS.hvac_clg_2006_handle)
    zone_humd_2001 = api.exchange.get_variable_value(state_argument, EPLUS.zone_humd_handle_2001)
    zone_humd_2002 = api.exchange.get_variable_value(state_argument, EPLUS.zone_humd_handle_2002)
    zone_humd_2003 = api.exchange.get_variable_value(state_argument, EPLUS.zone_humd_handle_2003)
    zone_humd_2004 = api.exchange.get_variable_value(state_argument, EPLUS.zone_humd_handle_2004)
    zone_humd_2005 = api.exchange.get_variable_value(state_argument, EPLUS.zone_humd_handle_2005)
    zone_humd_2006 = api.exchange.get_variable_value(state_argument, EPLUS.zone_humd_handle_2006)
    zone_window_2001 = api.exchange.get_variable_value(state_argument, EPLUS.zone_window_handle_2001)
    zone_window_2002 = api.exchange.get_variable_value(state_argument, EPLUS.zone_window_handle_2002)
    zone_window_2003 = api.exchange.get_variable_value(state_argument, EPLUS.zone_window_handle_2003)
    zone_window_2004 = api.exchange.get_variable_value(state_argument, EPLUS.zone_window_handle_2004)
    zone_window_2005 = api.exchange.get_variable_value(state_argument, EPLUS.zone_window_handle_2005)
    zone_window_2006 = api.exchange.get_variable_value(state_argument, EPLUS.zone_window_handle_2006)
    zone_ventmass_2001 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2001)
    zone_ventmass_2002 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2002)
    zone_ventmass_2003 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2003)
    zone_ventmass_2004 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2004)
    zone_ventmass_2005 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2005)
    zone_ventmass_2006 = api.exchange.get_variable_value(state_argument, EPLUS.zone_temp_handle_2006)

    EPLUS.y_humd.append(oa_humd)
    EPLUS.y_wind.append([oa_windspeed,oa_winddirct])
    EPLUS.y_solar.append([oa_solar_azi, oa_solar_alt, oa_solar_ang])
    EPLUS.y_zone_humd.append([zone_humd_2001,zone_humd_2002,zone_humd_2003, zone_humd_2004,zone_humd_2005,zone_humd_2006])
    EPLUS.y_zone_window.append([zone_window_2001,zone_window_2002,zone_window_2003, zone_window_2004,zone_window_2005,zone_window_2006])
    EPLUS.y_zone_ventmass.append([zone_ventmass_2001,zone_ventmass_2002,zone_ventmass_2003, zone_ventmass_2004,zone_ventmass_2005,zone_ventmass_2006])
    EPLUS.y_outdoor.append(temp_c_to_f(oa_temp))
    EPLUS.y_zone.append(temp_c_to_f(zone_temp))
    EPLUS.y_zone_temp_2001.append(temp_c_to_f(zone_temp_2001))
    EPLUS.y_zone_temp_2002.append(temp_c_to_f(zone_temp_2002))
    EPLUS.y_zone_temp_2003.append(temp_c_to_f(zone_temp_2003))
    EPLUS.y_zone_temp_2004.append(temp_c_to_f(zone_temp_2004))
    EPLUS.y_zone_temp_2005.append(temp_c_to_f(zone_temp_2005))
    EPLUS.y_zone_temp_2006.append(temp_c_to_f(zone_temp_2006))
    EPLUS.hvac_htg_2001.append(temp_c_to_f(hvac_htg_2001))
    EPLUS.hvac_clg_2001.append(temp_c_to_f(hvac_clg_2001))
    EPLUS.hvac_htg_2002.append(temp_c_to_f(hvac_htg_2002))
    EPLUS.hvac_clg_2002.append(temp_c_to_f(hvac_clg_2002))
    EPLUS.hvac_htg_2003.append(temp_c_to_f(hvac_htg_2003))
    EPLUS.hvac_clg_2003.append(temp_c_to_f(hvac_clg_2003))
    EPLUS.hvac_htg_2004.append(temp_c_to_f(hvac_htg_2004))
    EPLUS.hvac_clg_2004.append(temp_c_to_f(hvac_clg_2004))
    EPLUS.hvac_htg_2005.append(temp_c_to_f(hvac_htg_2005))
    EPLUS.hvac_clg_2005.append(temp_c_to_f(hvac_clg_2005))
    EPLUS.hvac_htg_2006.append(temp_c_to_f(hvac_htg_2006))
    EPLUS.hvac_clg_2006.append(temp_c_to_f(hvac_clg_2006))
    T_list = temp_c_to_f(np.array([zone_temp_2001, zone_temp_2002, zone_temp_2003, zone_temp_2004, zone_temp_2005, zone_temp_2006]))
    EPLUS.y_zone_temp.append(T_list)
    T_mean = np.mean(T_list)
    EPLUS.T_mean.append(T_mean)
    EPLUS.T_diff.append(np.max(T_list)-np.min(T_list))
    EPLUS.T_var.append(np.var(T_list))
    EPLUS.E_Facility.append(api.exchange.get_meter_value(state_argument, EPLUS.E_Facility_handle))
    EPLUS.E_HVAC.append(api.exchange.get_meter_value(state_argument, EPLUS.E_HVAC_handle))
    EPLUS.E_Heating.append(api.exchange.get_meter_value(state_argument, EPLUS.E_Heating_handle))
    EPLUS.E_Cooling.append(api.exchange.get_meter_value(state_argument, EPLUS.E_Cooling_handle))
    EPLUS.E_HVAC_all.append(api.exchange.get_meter_value(state_argument, EPLUS.E_HVAC_handle))
    EPLUS.sun_is_up.append(api.exchange.sun_is_up(state_argument))
    EPLUS.is_raining.append(api.exchange.today_weather_is_raining_at_time(state_argument, hour, time_step))
    EPLUS.outdoor_humidity.append(api.exchange.today_weather_outdoor_relative_humidity_at_time(state_argument, hour, time_step))
    EPLUS.wind_speed.append(api.exchange.today_weather_wind_speed_at_time(state_argument, hour, time_step))
    EPLUS.diffuse_solar.append(api.exchange.today_weather_diffuse_solar_at_time(state_argument, hour, time_step))
    year = 2022
    EPLUS.years.append(year)
    EPLUS.months.append(month)
    EPLUS.days.append(day)
    EPLUS.hours.append(hour)
    EPLUS.minutes.append(minute)
    EPLUS.current_times.append(current_time)
    EPLUS.actual_date_times.append(actual_date_time)
    EPLUS.actual_times.append(actual_time)
    timedelta = datetime.timedelta()
    if hour >= 24.0:
        hour = 23.0
        timedelta += datetime.timedelta(hours=1)
    if minute >= 60.0:
        minute = 59
        timedelta += datetime.timedelta(minutes=1)
    dt = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute)
    dt += timedelta
    EPLUS.x.append(dt)
    EPLUS.time_line.append(dt)
    if dt.weekday() > 4:
        EPLUS.weekday.append(dt.weekday())
        EPLUS.isweekday.append(0)
        EPLUS.isweekend.append(1)
    else:
        EPLUS.weekday.append(dt.weekday())
        EPLUS.isweekday.append(1)
        EPLUS.isweekend.append(0)
    EPLUS.work_time.append(EPLUS.isweekday[-1] * EPLUS.sun_is_up[-1])

    if EPLUS.RL_flag == False:
        EPLUS.episode_reward.append(0)
    if EPLUS.RL_flag == True:
        if time_interval == 0:
            EPLUS.episode_reward.append(0)
            EPLUS.action_list.append(0)
        
        done = False
        is_worktime = EPLUS.work_time[-1]
        O0 = EPLUS.y_outdoor[-2]
        E0 = EPLUS.E_HVAC[-2]
        R0 = EPLUS.is_raining[-2]
        W0 = EPLUS.work_time[-2]
        D0 = EPLUS.weekday[-2]
        M0 = EPLUS.months[-2]
        H0 = EPLUS.hours[-2]
        S0 = EPLUS.sun_is_up[-2]
        T_10 = EPLUS.y_zone_temp_2001[-2] 
        T_20 = EPLUS.y_zone_temp_2002[-2] 
        T_30 = EPLUS.y_zone_temp_2003[-2] 
        T_40 = EPLUS.y_zone_temp_2004[-2] 
        T_50 = EPLUS.y_zone_temp_2005[-2] 
        T_60 = EPLUS.y_zone_temp_2006[-2] 
        H_10 = EPLUS.hvac_htg_2001[-2] 
        H_20 = EPLUS.hvac_htg_2002[-2] 
        H_30 = EPLUS.hvac_htg_2003[-2] 
        H_40 = EPLUS.hvac_htg_2004[-2] 
        H_50 = EPLUS.hvac_htg_2005[-2] 
        H_60 = EPLUS.hvac_htg_2006[-2] 
        state_0 = [O0/100,W0,T_30/100, T_10/100,T_20/100,T_30/100,T_40/100,T_50/100,T_60/100, H_10/100,H_20/100,H_30/100,H_40/100,H_50/100,H_60/100]
        action_0 = EPLUS.action_list[-1]
        O1 = EPLUS.y_outdoor[-1] 
        E1 = EPLUS.E_HVAC[-1]
        W1 = EPLUS.work_time[-1]
        R1 = EPLUS.is_raining[-1]
        D1 = EPLUS.weekday[-1]
        M1 = EPLUS.months[-1]
        H1 = EPLUS.hours[-1]
        S1 = EPLUS.sun_is_up[-1]
        T_11 = EPLUS.y_zone_temp_2001[-1] 
        T_21 = EPLUS.y_zone_temp_2002[-1] 
        T_31 = EPLUS.y_zone_temp_2003[-1] 
        T_41 = EPLUS.y_zone_temp_2004[-1] 
        T_51 = EPLUS.y_zone_temp_2005[-1] 
        T_61 = EPLUS.y_zone_temp_2006[-1] 
        H_11 = EPLUS.hvac_htg_2001[-1]
        H_21 = EPLUS.hvac_htg_2002[-1] 
        H_31 = EPLUS.hvac_htg_2003[-1] 
        H_41 = EPLUS.hvac_htg_2004[-1] 
        H_51 = EPLUS.hvac_htg_2005[-1]
        H_61 = EPLUS.hvac_htg_2006[-1]
        state_1 = [O1/100,W1,T_31/100, T_11/100,T_21/100,T_31/100,T_41/100,T_51/100,T_61/100, H_11/100,H_21/100,H_31/100,H_41/100,H_51/100,H_61/100] 
        action_1 = agent.take_action(state_1)
        try:
            action_map = HVAC_action_list[action_1]
        except IndexError as e:
            print(f"IndexError: {e}")
        set_temp = [71,74]
        H_new_1, C_new_1 = HVAC_action(action_map[0], set_temp)
        H_new_2, C_new_2 = HVAC_action(action_map[1], set_temp)
        H_new_3, C_new_3 = HVAC_action(action_map[2], set_temp)
        H_new_4, C_new_4 = HVAC_action(action_map[3], set_temp)
        H_new_5, C_new_5 = HVAC_action(action_map[4], set_temp)
        H_new_6, C_new_6 = HVAC_action(action_map[5], set_temp)
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2001_handle, temp_f_to_c(H_new_1))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2001_handle, temp_f_to_c(C_new_1))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2002_handle, temp_f_to_c(H_new_2))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2002_handle, temp_f_to_c(C_new_2))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2003_handle, temp_f_to_c(H_new_3))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2003_handle, temp_f_to_c(C_new_3))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2004_handle, temp_f_to_c(H_new_4))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2004_handle, temp_f_to_c(C_new_4))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2005_handle, temp_f_to_c(H_new_5))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2005_handle, temp_f_to_c(C_new_5))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2006_handle, temp_f_to_c(H_new_6))
        api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2006_handle, temp_f_to_c(C_new_6))
        EPLUS.action_list.append(action_1)
        if is_worktime:
            E_factor = E_factor_day
            T_factor = T_factor_day
            work_flag = 0
            reward_signal = 0
        else:
            E_factor = E_factor_night
            T_factor = T_factor_night
            work_flag = 0
            reward_signal = 0
        reward_E = -E1 * E_factor
        reward_T1 = 1 * work_flag if 68 < T_11 < 77 else -(T_11 - 72) ** 2 * T_factor
        reward_T2 = 1 * work_flag if 68 < T_21 < 77 else -(T_21 - 72) ** 2 * T_factor
        reward_T3 = 1 * work_flag if 68 < T_31 < 77 else -(T_31 - 72) ** 2 * T_factor
        reward_T4 = 1 * work_flag if 68 < T_41 < 77 else -(T_41 - 72) ** 2 * T_factor
        reward_T5 = 1 * work_flag if 68 < T_51 < 77 else -(T_51 - 72) ** 2 * T_factor
        reward_T6 = 1 * work_flag if 68 < T_61 < 77 else -(T_61 - 72) ** 2 * T_factor
        reward_T = reward_T1+reward_T2+reward_T3+reward_T4+reward_T5+reward_T6
        current_action = HVAC_action_list[EPLUS.action_list[-1]]
        last_action = HVAC_action_list[EPLUS.action_list[-2]]
        change_action = np.array(current_action) ^ np.array(last_action)
        num_unstable = len(change_action[change_action==1])
        reward_signal = -signal_factor * num_unstable
        if signal_loss == True:
            reward_1 = reward_T + reward_E + reward_signal
        else:
            reward_1 = reward_T + reward_E 
        EPLUS.episode_reward.append(reward_1)
        EPLUS.episode_return = EPLUS.episode_return + reward_1
        if is_worktime:
            if T_mean > 77:
                EPLUS.T_Violation.append(T_mean-77)
            elif T_mean < 68:
                EPLUS.T_Violation.append(68-T_mean)
        if H_new_1<0 or H_new_1>120:
            api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2001_handle, temp_f_to_c(72))
            api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2001_handle, temp_f_to_c(72))
            done = True
        if H_new_2<0 or H_new_2>120:
            api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2002_handle, temp_f_to_c(72))
            api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2002_handle, temp_f_to_c(72))
            done = True
        if H_new_3<0 or H_new_3>120:
            api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2003_handle, temp_f_to_c(72))
            api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2003_handle, temp_f_to_c(72))
            done = True
        if H_new_4<0 or H_new_4>120:
            api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2004_handle, temp_f_to_c(72))
            api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2004_handle, temp_f_to_c(72))
            done = True
        if H_new_5<0 or H_new_5>120:
            api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2005_handle, temp_f_to_c(72))
            api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2005_handle, temp_f_to_c(72))
            done = True
        if H_new_6<0 or H_new_6>120:
            api.exchange.set_actuator_value(state_argument, EPLUS.hvac_htg_2006_handle, temp_f_to_c(72))
            api.exchange.set_actuator_value(state_argument, EPLUS.hvac_clg_2006_handle, temp_f_to_c(72))
            done = True
        if done == True:
            EPLUS.score.append(EPLUS.episode_return)
            EPLUS.episode_return = 0

        replay_buffer.add(state_0, action_0, reward_1, state_1, done)

        if replay_buffer.size() > minimal_size :
            agent.update(replay_buffer, replay_buffer_2, batch_size)
    if time_interval > view_distance:
        y_outdoor = EPLUS.y_outdoor[-view_distance::]
        y_zone = EPLUS.y_zone[-view_distance::]
        y_htg = EPLUS.y_htg[-view_distance::]
        y_clg = EPLUS.y_clg[-view_distance::]
        hvac_htg_2001 = EPLUS.hvac_htg_2001[-view_distance::]        
        hvac_htg_2002 = EPLUS.hvac_htg_2002[-view_distance::]
        hvac_htg_2003 = EPLUS.hvac_htg_2003[-view_distance::]
        hvac_htg_2004 = EPLUS.hvac_htg_2004[-view_distance::]
        hvac_htg_2005 = EPLUS.hvac_htg_2005[-view_distance::]
        hvac_htg_2006 = EPLUS.hvac_htg_2006[-view_distance::]
        y_zone_temp_2001 = EPLUS.y_zone_temp_2001[-view_distance::]
        y_zone_temp_2002 = EPLUS.y_zone_temp_2002[-view_distance::]
        y_zone_temp_2003 = EPLUS.y_zone_temp_2003[-view_distance::]
        y_zone_temp_2004 = EPLUS.y_zone_temp_2004[-view_distance::]
        y_zone_temp_2005 = EPLUS.y_zone_temp_2005[-view_distance::]
        y_zone_temp_2006 = EPLUS.y_zone_temp_2006[-view_distance::]
        T_mean = EPLUS.T_mean[-view_distance::]
        E_HVAC = EPLUS.E_HVAC[-view_distance::]        
        episode_reward = EPLUS.episode_reward[-view_distance::]
        x = EPLUS.x[-view_distance::]
        work_time = EPLUS.work_time[-view_distance::]
    EPLUS.time_interval = EPLUS.time_interval+1
    if time_interval % EPLUS.FPS == 0:
        EPLUS.T_map = {}
        EPLUS.T_map['Thermal Zone 1'] = zone_temp_2001
        EPLUS.T_map['Thermal Zone 2'] = zone_temp_2002
        EPLUS.T_map['Thermal Zone 3'] = zone_temp_2003
        EPLUS.T_map['Thermal Zone 4'] = zone_temp_2004
        EPLUS.T_map['Thermal Zone 5'] = zone_temp_2005
        EPLUS.T_map['Thermal Zone 6'] = zone_temp_2006
        EPLUS.T_map['Outdoor Temp'] = oa_temp

def read_parameters_from_txt(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split(':')
                parameters[key.strip()] = value.strip()
                #print(f"{key.strip()}: {value.strip()}")
    return parameters

if __name__ == '__main__':
    parameters = read_parameters_from_txt('/Users/nathancarey/Replay/parameters.txt')
    F_bottom = int(parameters['F_bottom'])
    F_top = int(parameters['F_top'])
    signal_loss = bool(parameters['signal_loss'])
    signal_factor = float(parameters['signal_factor'])
    minimal_size = int(parameters['minimal_size'])
    batch_size = int(parameters['batch_size'])

    parameters = {key: eval(value) if value.isdigit() else value for key, value in parameters.items()}
    climate_files = [ 
                    './weather_data/ARIZONA.epw',
                    './weather_data/CALIFORNIA.epw',
                    './weather_data/DUBAI.epw',
                    './weather_data/LDN.epw',
                    './weather_data/CAPETOWN.epw',
                    './weather_data/TEXAS.epw',
                    './weather_data/SINGAPORE.epw',
                    './weather_data/SOUTHCAROLINA.epw',
                    './weather_data/TOKYO.epw',
                    './weather_data/VANCOUVER.epw',
                    './weather_modified/ARIZONA_modified.epw',
                    './weather_modified/CALIFORNIA_modified.epw',
                    './weather_modified/DUBAI_modified.epw',
                    './weather_modified/LDN_modified.epw',
                    './weather_modified/CAPETOWN_modified.epw',
                    './weather_modified/SOUTHCAROLINA_modified.epw',
                    './weather_modified/SINGAPORE_modified.epw',
                    './weather_modified/TEXAS_modified.epw',
                    './weather_modified/TOKYO_modified.epw',
                    './weather_modified/VANCOUVER_modified.epw',

                     ]
                     
    seed_values = [ 52] 
    num_rows_list = [100000] 

    rae_files = [
    './Buffer-example/20k_TX_c_SC_AZ.csv',
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
                    agent = DQN(
                        state_dim=parameters['state_dim'],
                        action_dim=parameters['action_dim'],
                        learning_rate=float(parameters['lr']),
                        gamma=float(parameters['gamma']),
                        epsilon=float(parameters['epsilon']),
                        target_update=parameters['target_update'],
                        epsilon_min=float(parameters['epsilon_min']),
                        epsilon_decay=float(parameters['epsilon_decay']),
                        device=device
                    )
                    replay_buffer = ReplayBuffer(parameters['buffer_size'])
                    replay_buffer_2 = ReplayBuffer(capacity=num_rows)
                    replay_buffer_2.load_from_csv(rae_file, num_rows=num_rows)
                    trainer = DQNTrainer(agent, replay_buffer, weather_data, parameters['save_idf'], parameters)
                    trainer.setup_baseline()
                    epoch_results = trainer.train()
                    results_df = trainer.report_results()           
                    all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results_df.to_csv(f'all_results_{timestamp}.csv', index=False)

    print(f"Results saved to all_results_{timestamp}.csv")
