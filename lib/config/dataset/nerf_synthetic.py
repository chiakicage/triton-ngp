from dataclasses import dataclass
from . import DatasetConfig
# from typing import List

@dataclass
class SplitConfig:
	split: str
	view: int
	input_ratio: float

@dataclass
class SyntheticDatasetConfig(DatasetConfig):
	scene: str
	init_rays_num: int
	train_dataset: SplitConfig
	test_dataset: SplitConfig

