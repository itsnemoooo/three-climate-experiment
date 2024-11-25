# src/callbacks.py

import sys
import time
import datetime
import numpy as np
import ctypes

from src.utils import temp_c_to_f, temp_f_to_c

def HVAC_action(action, temp):
    """Map action to HVAC settings."""
    if action == 0:
        H_new = temp[0]
        C_new = temp[1]
    elif action == 1:
        H_new = temp[0]
        C_new = temp[1]
    return int(H_new), int(C_new)

def callback_function_DQN(state_argument, api, EPLUS, agent, replay_buffer, replay_buffer_2, parameters):
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
        # Get all handles
        EPLUS.oa_temp_handle = api.exchange.get_variable_handle(state_argument, "SITE OUTDOOR AIR DRYBULB TEMPERATURE", "ENVIRONMENT")
        EPLUS.oa_humd_handle = api.exchange.get_variable_handle(state_argument, "Site Outdoor Air Drybulb Temperature", "ENVIRONMENT")
        EPLUS.oa_windspeed_handle = api.exchange.get_variable_handle(state_argument, "Site Wind Speed", "ENVIRONMENT")
        EPLUS.oa_winddirct_handle = api.exchange.get_variable_handle(state_argument, "Site Wind Direction", "ENVIRONMENT")
        EPLUS.oa_solar_azi_handle = api.exchange.get_variable_handle(state_argument, "Site Solar Azimuth Angle", "ENVIRONMENT")
        EPLUS.oa_solar_alt_handle = api.exchange.get_variable_handle(state_argument, "Site Solar Altitude Angle", "ENVIRONMENT")
        EPLUS.oa_solar_ang_handle = api.exchange.get_variable_handle(state_argument, "Site Solar Hour Angle", "ENVIRONMENT")
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

        if -1 in EPLUS.handle_availability():
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

    ''' Temperature and Other Variables '''
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
    zone_ventmass_2001 = api.exchange.get_variable_value(state_argument, EPLUS.zone_ventmass_handle_2001)
    zone_ventmass_2002 = api.exchange.get_variable_value(state_argument, EPLUS.zone_ventmass_handle_2002)
    zone_ventmass_2003 = api.exchange.get_variable_value(state_argument, EPLUS.zone_ventmass_handle_2003)
    zone_ventmass_2004 = api.exchange.get_variable_value(state_argument, EPLUS.zone_ventmass_handle_2004)
    zone_ventmass_2005 = api.exchange.get_variable_value(state_argument, EPLUS.zone_ventmass_handle_2005)
    zone_ventmass_2006 = api.exchange.get_variable_value(state_argument, EPLUS.zone_ventmass_handle_2006)

    EPLUS.y_humd.append(oa_humd)
    EPLUS.y_wind.append([oa_windspeed, oa_winddirct])
    EPLUS.y_solar.append([oa_solar_azi, oa_solar_alt, oa_solar_ang])
    EPLUS.y_zone_humd.append([zone_humd_2001, zone_humd_2002, zone_humd_2003, zone_humd_2004, zone_humd_2005, zone_humd_2006])
    EPLUS.y_zone_window.append([zone_window_2001, zone_window_2002, zone_window_2003, zone_window_2004, zone_window_2005, zone_window_2006])
    EPLUS.y_zone_ventmass.append([zone_ventmass_2001, zone_ventmass_2002, zone_ventmass_2003, zone_ventmass_2004, zone_ventmass_2005, zone_ventmass_2006])
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
    EPLUS.T_diff.append(np.max(T_list) - np.min(T_list))
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

    if not EPLUS.RL_flag:
        EPLUS.episode_reward.append(0)
    else:
        if time_interval == 0:
            EPLUS.episode_reward.append(0)
            EPLUS.action_list.append(0)
        
        done = False
        is_worktime = EPLUS.work_time[-1]
        # Previous state
        if len(EPLUS.y_outdoor) < 2:
            return  # Not enough data to take action
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
        state_0 = [O0/100, W0, T_30/100, T_10/100, T_20/100, T_30/100, T_40/100, T_50/100, T_60/100, 
                   H_10/100, H_20/100, H_30/100, H_40/100, H_50/100, H_60/100]
        action_0 = EPLUS.action_list[-1]
        # Current state
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
        state_1 = [O1/100, W1, T_31/100, T_11/100, T_21/100, T_31/100, T_41/100, T_51/100, T_61/100, 
                   H_11/100, H_21/100, H_31/100, H_41/100, H_51/100, H_61/100] 
        action_1 = agent.take_action(state_1)
        
        HVAC_action_list = []
        for HC_1 in [0,1]:
            for HC_2 in [0,1]:
                for HC_3 in [0,1]:
                    for HC_4 in [0,1]:
                        for HC_5 in [0,1]:
                            for HC_6 in [0,1]:
                                HVAC_action_list.append([HC_1,HC_2,HC_3,HC_4,HC_5,HC_6])
        
        try:
            action_map = HVAC_action_list[action_1]
        except IndexError as e:
            print(f"IndexError: {e}")
            action_map = [0] * NUM_HVAC  # Default action
        
        set_temp = [71, 74]
        H_new_list = []
        C_new_list = []
        for i, action_val in enumerate(action_map):
            H_new, C_new = HVAC_action(action_val, set_temp)
            H_new_list.append(H_new)
            C_new_list.append(C_new)
            api.exchange.set_actuator_value(state_argument, getattr(EPLUS, f'hvac_htg_200{i+1}_handle'), temp_f_to_c(H_new))
            api.exchange.set_actuator_value(state_argument, getattr(EPLUS, f'hvac_clg_200{i+1}_handle'), temp_f_to_c(C_new))
        
        EPLUS.action_list.append(action_1)
        if is_worktime:
            E_factor = E_factor_day
            T_factor = T_factor_day
            work_flag = 1
        else:
            E_factor = E_factor_night
            T_factor = T_factor_night
            work_flag = 0

        reward_E = -E1 * E_factor
        reward_T = 0
        for T in [T_11, T_21, T_31, T_41, T_51, T_61]:
            if 68 < T < 77:
                reward_T += 1 * work_flag
            else:
                reward_T -= (T - 72) ** 2 * T_factor
        # Assuming signal_factor and signal_loss are in parameters
        signal_factor = parameters.get('signal_factor', 0.0)
        signal_loss = parameters.get('signal_loss', False)
        if 'signal_factor' in parameters and 'signal_loss' in parameters:
            current_action = HVAC_action_list[EPLUS.action_list[-1]]
            last_action = HVAC_action_list[EPLUS.action_list[-2]] if len(EPLUS.action_list) >=2 else current_action
            change_action = np.array(current_action) ^ np.array(last_action)
            num_unstable = np.sum(change_action == 1)
            reward_signal = -signal_factor * num_unstable
        else:
            reward_signal = 0
        if signal_loss:
            reward_1 = reward_T + reward_E + reward_signal
        else:
            reward_1 = reward_T + reward_E 
        
        EPLUS.episode_reward.append(reward_1)
        EPLUS.episode_return += reward_1
        
        # Temperature Violations
        if is_worktime:
            if T_mean > 77:
                EPLUS.T_Violation.append(T_mean - 77)
            elif T_mean < 68:
                EPLUS.T_Violation.append(68 - T_mean)
        
        # Check for actuator limits
        done = False
        for H_new in H_new_list:
            if H_new < 0 or H_new > 120:
                for i in range(1, NUM_HVAC+1):
                    api.exchange.set_actuator_value(state_argument, getattr(EPLUS, f'hvac_htg_200{i}_handle'), temp_f_to_c(72))
                    api.exchange.set_actuator_value(state_argument, getattr(EPLUS, f'hvac_clg_200{i}_handle'), temp_f_to_c(72))
                done = True
                break

        if done:
            EPLUS.score.append(EPLUS.episode_return)
            EPLUS.episode_return = 0

        replay_buffer.add(state_0, action_0, reward_1, state_1, done)

        minimal_size = parameters.get('minimal_size', 1000)
        batch_size = parameters.get('batch_size', 32)
        if replay_buffer.size() > minimal_size:
            agent.update(replay_buffer, replay_buffer_2, batch_size)
