from dataclasses import dataclass

@dataclass
class TaskConfig:
    model_module: str
    loss_module: str
    evaluator_module: str
    visualizer_module: str
