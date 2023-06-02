from dataclasses import dataclass

from .dataset import DatasetConfig
from .task import TaskConfig

@dataclass
class Config:
    dataset: DatasetConfig
    task: TaskConfig
