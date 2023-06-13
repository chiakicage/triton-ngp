from dataclasses import dataclass

@dataclass
class TrainConfig:
    batch_size: int
    lr: float
    weight_decay: float
    num_epochs: int

@dataclass
class TaskConfig:
    # model_module: str
    # loss_module: str
    # evaluator_module: str
    # visualizer_module: str
    device: str
    train: TrainConfig
