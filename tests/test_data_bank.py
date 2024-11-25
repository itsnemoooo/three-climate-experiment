from src.data_bank import Data_Bank

def test_data_bank_initialization():
    parameters = {
        'FPS': 10,
        'E_factor_day': 1.0,
        'T_factor_day': 1.0,
        'E_factor_night': 1.0,
        'T_factor_night': 1.0
    }
    data_bank = Data_Bank(parameters)
    assert data_bank.FPS == 10, "FPS not set correctly"
    assert data_bank.E_factor_day == 1.0, "E_factor_day not set correctly"
    print("test_data_bank_initialization passed")
