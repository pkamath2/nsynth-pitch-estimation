import json
import torch

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def get_config():
    filepath = 'config/config.json'
    config = {}
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config