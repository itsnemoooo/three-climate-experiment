import os
import sys
import sympy
import getpass
import cProfile
import pstats
import ctypes
import numbers
import typing
import inspect
import sys
import os
import datetime

pyenergyplus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../OpenStudio-1.4.0/EnergyPlus'))
sys.path.append(pyenergyplus_path)

from pyenergyplus.api import EnergyPlusAPI



# Initialize the EnergyPlus API
api = EnergyPlusAPI()

def callback_function(state, api):
    # This is a simple callback function for testing
    print("Callback function called.")
    # You can add more logic here if needed

def run_energyplus_test(weather_file, idf_file):
    # Create a new EnergyPlus state
    print("Creating new EnergyPlus state...")
    E_state = api.state_manager.new_state()
    print("New state created.")

    # Set console output status
    print("Setting console output status...")
    api.runtime.set_console_output_status(E_state, False)
    print("Console output status set.")

    # Set up the callback
    print("Setting up callback...")
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
        E_state,
        lambda state: callback_function(state, api)
    )
    print("Callback set up.")

    # Run EnergyPlus
    print(f"Running EnergyPlus with weather file: {weather_file} and IDF file: {idf_file}")
    try:
        api.runtime.run_energyplus(E_state, ['-w', weather_file, '-d', 'out/', idf_file])
        print("EnergyPlus run completed successfully.")
    except Exception as e:
        print(f"Error running EnergyPlus: {e}")

    # Clean up
    api.state_manager.reset_state(E_state)
    api.state_manager.delete_state(E_state)
    print("EnergyPlus state reset and deleted.")

if __name__ == "__main__":
    # Specify the paths to your weather file and IDF file
    weather_file_path = 'data/weather-data/ARIZONA.epw'  # Update this path
    idf_file_path = 'run.idf'  # Update this path

    # Check if files exist
    if not os.path.exists(weather_file_path):
        print(f"Error: Weather file not found at {weather_file_path}")
    if not os.path.exists(idf_file_path):
        print(f"Error: IDF file not found at {idf_file_path}")

    # Run the EnergyPlus test
    run_energyplus_test(weather_file_path, idf_file_path)