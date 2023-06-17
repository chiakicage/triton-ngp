import torch.utils.data as data
from lib.config.dataset.nerf_synthetic import SyntheticDatasetConfig
from torchvision import transforms
import numpy as np
import json
import cv2
import imageio
import os
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("dataset")


class Dataset(data.Dataset):
    def __init__(self, cfg: SyntheticDatasetConfig, is_train=True):
        super(Dataset, self).__init__()
        self.scene = cfg.scene
        self.data_root = os.path.join(cfg.path, self.scene)
        split_dataset = cfg.train_dataset if is_train else cfg.test_dataset
        self.split = split_dataset.split
        self.input_ratio = split_dataset.input_ratio
        self.rays_num = cfg.init_rays_num
        self.c2ws = []
        self.imgs = []
        image_paths = []
        json_info = json.load(
            open(os.path.join(self.data_root, f"transforms_{self.split}.json"))
        )

        for i, frame in enumerate(json_info["frames"]):
            image_path = os.path.join(self.data_root, frame["file_path"][2:] + ".png")
            img = np.asarray(imageio.imread(image_path)).astype(np.float32) / 255.0
            img = img[..., :3] * img[..., -1:] + 1 - img[..., -1:]
            self.imgs.append(img)
            c2w = np.array(frame["transform_matrix"])
            self.c2ws.append(c2w)

        H, W = self.imgs[0].shape[:2]
        FOV = float(json_info["camera_angle_x"])
        self.H = int(H * self.input_ratio)
        self.W = int(W * self.input_ratio)
        self.focal = 0.5 * self.W / np.tan(0.5 * FOV)
        self.K = np.array(
            [[self.focal, 0, 0.5 * self.W], [0, self.focal, 0.5 * self.H], [0, 0, 1]]
        )
        # logger.info(f"internal: {self.internal}")
        self.imgs = [cv2.resize(img, (self.W, self.H)) for img in self.imgs]

        X, Y = np.meshgrid(np.arange(self.W), np.arange(self.H))
        u, v = X.astype(np.float32), Y.astype(np.float32)
        self.uv = np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32)

    def update_rays_num(self, rays_num):
        self.rays_num = rays_num

    def get_rays(self, img, c2w):
        if self.split == "train":
            ids = np.random.choice(len(self.uv), self.rays_num, replace=False)
            uv = self.uv[ids]
            rgb = img.reshape(-1, 3)[ids]
        else:
            uv = self.uv
            rgb = img.reshape(-1, 3)
        rays_x = (uv[..., 0] - self.K[0, 2] + 0.5) / self.K[0, 0]
        rays_y = -(uv[..., 1] - self.K[1, 2] + 0.5) / self.K[1, 1]
        rays_z = -np.ones_like(rays_x)
        rays_d = np.stack([rays_x, rays_y, rays_z], axis=-1)
        rays_d = rays_d @ c2w[:3, :3].T
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
        rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)
        return uv, rays_o, rays_d, rgb

    def __getitem__(self, index):
        uv, rays_o, rays_d, rgb = self.get_rays(self.imgs[index], self.c2ws[index])
        uv = uv.astype(np.float32)
        rays_o = rays_o.astype(np.float32)
        rays_d = rays_d.astype(np.float32)
        rgb = rgb.astype(np.float32)
        ret = {"uv": uv, "rgb": rgb, "rays_o": rays_o, "rays_d": rays_d}
        return ret

    def __len__(self):
        return len(self.imgs)
