from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
	print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
	main()
