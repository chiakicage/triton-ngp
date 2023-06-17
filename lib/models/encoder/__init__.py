import importlib
from lib.config.task.nerf import EncoderConfig, FrequencyEncoderConfig
def make_encoder(cfg: EncoderConfig):
    if cfg.type == "frequency":
        encoder = importlib.import_module('lib.models.encoder.freq')
        return encoder.Encoder(cfg)
    elif cfg.type == "spherical":
        encoder = importlib.import_module('lib.models.encoder.spherical')
        return encoder.Encoder(cfg)
    elif cfg.type == "hash":
        encoder = importlib.import_module('lib.models.encoder.hash')
        return encoder.HashGrid(cfg)
    else:
        raise NotImplementedError
