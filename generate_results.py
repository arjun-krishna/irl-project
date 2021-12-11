import torch
def generate_results(data_path):
    d = torch.load(data_path)
    print(d['experiment_details'])
    d['success rate']
    d['steps']