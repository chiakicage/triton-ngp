import torch.utils.data as data
from lib.config.dataset.nerf_synthetic import SyntheticDatasetConfig

class Dataset(data.Dataset):
    def __init__(self, cfg: SyntheticDatasetConfig):
        super(Dataset, self).__init__()
        
    
