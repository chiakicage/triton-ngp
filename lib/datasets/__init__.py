import torch
import torch.utils.data as data
import importlib
from lib.config.dataset.nerf_synthetic import SyntheticDatasetConfig
from lib.config.dataset import DatasetConfig

def make_dataset(cfg: DatasetConfig, is_train=True, batch_size=1):
    print(cfg.dataset_module)
    dataset = importlib.import_module(cfg.dataset_module)
    
    return dataset.Dataset(cfg, is_train=is_train, batch_size=batch_size)


