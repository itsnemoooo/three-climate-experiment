import torch
from src.models import DNN_3

def test_dnn_3():
    state_dim = 10
    action_dim = 4
    model = DNN_3(state_dim, action_dim)
    input_tensor = torch.rand(1, state_dim)
    output = model(input_tensor)
    assert output.shape == (1, action_dim), f"Expected output shape (1, {action_dim}), got {output.shape}"
    print("test_dnn_3 passed")
