import torch.utils.data as data
from lib.config.dataset.nerf_synthetic import SyntheticDatasetConfig
from torchvision import transforms
import numpy as np
import json
import cv2
import imageio
import os
import matplotlib.pyplot as plt


class Dataset(data.Dataset):
    def __init__(self, cfg: SyntheticDatasetConfig, is_train=True, batch_size=1):
        super(Dataset, self).__init__()
        self.scene = cfg.scene
        self.data_root = os.path.join(cfg.path, self.scene)
        split_dataset = cfg.train_dataset if is_train else cfg.test_dataset
        self.split = split_dataset.split
        self.input_ratio = split_dataset.input_ratio
        self.batch_size = batch_size
        image_paths = []
        json_info = json.load(
            open(os.path.join(self.data_root, f'transforms_{self.split}.json')))
        for frame in json_info['frames']:
            image_paths.append(os.path.join(
                self.data_root, frame['file_path'][2:] + '.png'))
        self.imgs = [np.asarray(imageio.imread(image_path)).astype(
            np.float32) / 255. for image_path in image_paths]
        self.imgs = [img[..., :3] * img[..., -1:] + 1 - img[..., -1:]
                     for img in self.imgs]
        H, W = self.imgs[0].shape[:2]
        self.H = H
        self.W = W
        imgs = [cv2.resize(img, (int(W * self.input_ratio),
                           int(H * self.input_ratio))) for img in self.imgs]
        
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        u, v = X.astype(np.float32) / (W - 1), Y.astype(np.float32) / (H - 1)
        self.uv = np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32)

    def __getitem__(self, index):
        if self.split == 'train':
            ids = np.random.choice(
                len(self.uv), self.batch_size, replace=False)
            uv = self.uv[ids]
            rgb = self.imgs[index].reshape(-1, 3)[ids]
        else:
            uv = self.uv
            rgb = self.imgs[index].reshape(-1, 3)
        ret = {'uv': uv, 'rgb': rgb}
        return ret

    def __len__(self):
        return len(self.imgs)
