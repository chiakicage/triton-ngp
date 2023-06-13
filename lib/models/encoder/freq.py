import torch
from lib.config.task.nerf import FrequencyEncoderConfig

class Encoder:
    def __init__(self, cfg: FrequencyEncoderConfig):
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.input_dim * cfg.L * 2
        self.freqs = 2.**torch.linspace(0, cfg.L-1, steps=cfg.L)
        self.embed_fns = []
        for freq in self.freqs:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(freq * x))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(freq * x))
        # print(self.embed_fns)
    
    def __call__(self, x: torch.Tensor):
        # x = x / torch.norm(x, p=2, dim=-1, keepdim=True)
        return torch.cat([fn(x) for fn in self.embed_fns], dim=-1)

    
