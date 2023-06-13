import torch
import torch.utils.data as data
import importlib
from lib.config.dataset.nerf_synthetic import SyntheticDatasetConfig
from lib.config.dataset import DatasetConfig

def make_dataset(cfg: DatasetConfig, is_train=True):
    dataset = importlib.import_module(cfg.dataset_module)
    
    return dataset.Dataset(cfg, is_train=is_train)


