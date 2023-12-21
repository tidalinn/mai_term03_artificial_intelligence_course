'''Загрузка датасета лиц
'''

import h5py
import pyredner
import torch

def load_data(file: str):
    with h5py.File(file, 'r') as hf:
        shape_mean = torch.tensor(
            hf['shape/model/mean'], 
            device = pyredner.get_device()
        )
        
        shape_basis = torch.tensor(
            hf['shape/model/pcaBasis'], # базисный вектор формы
            device = pyredner.get_device()
        )
        
        triangle_list = torch.tensor(
            hf['shape/representer/cells'], 
            device = pyredner.get_device()
        )
        
        color_mean = torch.tensor(
            hf['color/model/mean'], 
            device = pyredner.get_device()
        )
        
        color_basis = torch.tensor(
            hf['color/model/pcaBasis'], # базисный вектор цвета
            device = pyredner.get_device()
        )
    
    return shape_mean, shape_basis, triangle_list, color_mean, color_basis