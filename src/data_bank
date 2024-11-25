# src/data_bank.py

import os
import sys
import csv
import copy
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime
from collections import deque
from pyenergyplus.api import EnergyPlusAPI

from src.utils import temp_c_to_f, temp_f_to_c

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
        handle_list = [
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
        return handle_list