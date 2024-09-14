import torch
import os 
path = 'recordings'

for dir in os.listdir(path):
    for file in os.listdir(os.path.join(path, dir)):
        if file.endswith('.pt'):
            data = torch.load(os.path.join(path, dir, file))
            preds = data['preds']
            goal = data['goal']
            input = data['input']
            print(data)