import sys
import os
sys.path.clear()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the pyenergyplus directory to the Python path
pyenergyplus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../OpenStudio-1.4.0/EnergyPlus'))
sys.path.append(pyenergyplus_path)

# Optional: Verify the paths
print("Updated sys.path:", sys.path)

try:
    from pyenergyplus.api import EnergyPlusAPI
    print("pyenergyplus is successfully imported!")
except ModuleNotFoundError as e:
    print("Failed to import pyenergyplus:", e)