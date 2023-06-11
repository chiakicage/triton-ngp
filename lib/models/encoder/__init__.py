import importlib
from lib.config.task.nerf import EncoderConfig, FrequencyEncoderConfig
def make_encoder(cfg: EncoderConfig):
    if cfg.type == "frequency":
        encoder = importlib.import_module('lib.models.encoder.freq')
        return encoder.Encoder(cfg)
    else:
        raise NotImplementedError
