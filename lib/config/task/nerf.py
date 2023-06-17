from dataclasses import dataclass
from ..task import TaskConfig

@dataclass
class EncoderConfig:
    type: str

@dataclass
class FrequencyEncoderConfig(EncoderConfig):
    input_dim: int
    L: int

@dataclass
class SphericalHarmonicsEncoderConfig(EncoderConfig):
    level: int

@dataclass
class HashEncodingConfig(EncoderConfig):
    L: int
    F: int
    logT: int
    Nmin: int
    Nmax: int
    x_min: float
    x_max: float

@dataclass
class NetworkConfig:
    base_mlp_D: int
    base_mlp_W: int
    head_mlp_D: int
    head_mlp_W: int
    geo_feat_dim: int
    xyz_encoder: EncoderConfig
    dir_encoder: EncoderConfig

@dataclass
class NeRFConfig(TaskConfig):
    network: NetworkConfig
