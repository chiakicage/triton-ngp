import nerfacc
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
import torch

from lib.config import Config, DatasetConfig
from lib.datasets import make_dataset
from lib.models.encoder import make_encoder
from lib.models.nerf import Network
from lib.train.trainer import Trainer
import os

from lib.view.viewer import Viewer

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: Config):
    print(cfg.dataset)
    print(OmegaConf.to_yaml(cfg))
    cfg.task
    train_dataset = make_dataset(cfg.dataset, is_train=True)
    test_dataset = make_dataset(cfg.dataset, is_train=False)
    network = Network(cfg.task)
    estimator = nerfacc.OccGridEstimator(
        roi_aabb=torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]).to(cfg.task.device)
    )
    # trainer = Trainer(network, estimator, cfg.task)
    # trainer.train(train_dataset, test_dataset)
    checkpoint = torch.load("results/model.pth")
    network.load_state_dict(checkpoint['radiance_field_state_dict'])
    estimator.load_state_dict(checkpoint['estimator_state_dict'])
    viewer = Viewer(network, estimator, cfg.task)
    viewer.view(test_dataset)

    # encoder = make_encoder(cfg.task)
    # print(dataset[0])
    # print(os.getcwd())


if __name__ == "__main__":
    main()
