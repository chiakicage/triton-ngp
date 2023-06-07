from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

from lib.config import Config, DatasetConfig
from lib.datasets import make_dataset

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: Config):
	print(cfg.dataset)
	print(OmegaConf.to_yaml(cfg))
	dataset = make_dataset(cfg.dataset)
	print(dataset[0])

if __name__ == "__main__":
	main()
