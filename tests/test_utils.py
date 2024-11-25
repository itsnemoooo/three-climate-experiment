from src.utils import temp_c_to_f, temp_f_to_c, read_parameters_from_txt

def test_temp_conversion():
    assert temp_c_to_f(0) == 32, "C to F conversion failed for 0°C"
    assert temp_f_to_c(32) == 0, "F to C conversion failed for 32°F"
    assert temp_c_to_f(100) == 212, "C to F conversion failed for 100°C"
    assert temp_f_to_c(212) == 100, "F to C conversion failed for 212°F"
    print("test_temp_conversion passed")

def test_read_parameters_from_txt(tmp_path):
    # Create a temporary parameters.txt file
    content = """state_dim:10
action_dim:4
lr:0.001
gamma:0.99
epsilon:1.0
epsilon_min:0.1
epsilon_decay:0.99
target_update:10
epochs:5
buffer_size:100
minimal_size:10
batch_size:5
signal_factor:0.0
signal_loss:False
RL_flag:True
FPS:1
E_factor_day:1.0
T_factor_day:1.0
E_factor_night:1.0
T_factor_night:1.0
HVAC_action_list:[[0,1],[1,0]]
save_idf:dummy_file.idf
"""
    file = tmp_path / "parameters.txt"
    file.write_text(content)
    parameters = read_parameters_from_txt(file)
    assert parameters['state_dim'] == '10', "Read parameter 'state_dim' incorrect"
    assert parameters['HVAC_action_list'] == '[[0,1],[1,0]]', "Read parameter 'HVAC_action_list' incorrect"
    print("test_read_parameters_from_txt passed")
