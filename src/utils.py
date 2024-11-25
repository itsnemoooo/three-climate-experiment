# src/utils.py

def temp_c_to_f(temp_c):
    """Convert temperature from Celsius to Fahrenheit."""
    return 1.8 * temp_c + 32

def temp_f_to_c(temp_f):
    """Convert temperature from Fahrenheit to Celsius."""
    return (temp_f - 32) / 1.8

def read_parameters_from_txt(file_path):
    """Read parameters from a text file."""
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and ':' in line:
                key, value = line.split(':', 1)
                parameters[key.strip()] = value.strip()
    return parameters