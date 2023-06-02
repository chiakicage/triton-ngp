from dataclasses import dataclass

@dataclass
class DatasetConfig:
    name: str
    path: str
    dataset_module: str

