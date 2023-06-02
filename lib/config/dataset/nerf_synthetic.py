from dataclasses import dataclass
from . import DatasetConfig

@dataclass
class SplitConfig:
	split: str
	view: int
	input_ratio: float

@dataclass
class SyntheticDatasetConfig(DatasetConfig):
	train_dataset: SplitConfig
	test_dataset: SplitConfig

