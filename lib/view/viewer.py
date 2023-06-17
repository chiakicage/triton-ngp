import cv2
import imageio
import nerfacc
import numpy as np
import torch
from tqdm import tqdm
from lib.config.task import TaskConfig
from lib.datasets.nerf_synthetic import Dataset
from lib.models.nerf import Network
import torch.nn.functional as F


class Viewer:
    def __init__(
        self,
        radiance_field: Network,
        estimator: nerfacc.OccGridEstimator,
        cfg: TaskConfig,
    ):
        self.device = cfg.device
        self.radiance_field = radiance_field.to(self.device)
        self.estimator = estimator.to(self.device)

    def view(self, test_dataset: Dataset):
        # 使用OpenCV展示NeRF生成的图像
        trans_t = lambda t: torch.Tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
        ).float()

        rot_phi = lambda phi: torch.Tensor(
            [
                [1, 0, 0, 0],
                [0, np.cos(phi), -np.sin(phi), 0],
                [0, np.sin(phi), np.cos(phi), 0],
                [0, 0, 0, 1],
            ]
        ).float()

        rot_theta = lambda th: torch.Tensor(
            [
                [np.cos(th), 0, -np.sin(th), 0],
                [0, 1, 0, 0],
                [np.sin(th), 0, np.cos(th), 0],
                [0, 0, 0, 1],
            ]
        ).float()

        def pose_spherical(theta, phi, radius):
            c2w = trans_t(radius)
            c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
            c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
            c2w = (
                torch.Tensor(
                    np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                )
                @ c2w
            )
            return c2w

        render_poses = torch.stack(
            [
                pose_spherical(angle, -30.0, 4.0)
                for angle in np.linspace(-180, 180, 30 + 1)[:-1]
            ],
            0,
        )

        # 初始化OpenCV
        # 创建 VideoWriter 对象
        output_path = "results/video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            output_path, fourcc, 30, (test_dataset.W, test_dataset.H)
        )

        # 推理步骤
        render_step_size = 5e-3
        alpha_thre = 0.0
        cone_angle = 0.0
        render_bkgd = torch.ones(3, device=self.device)
        self.radiance_field.eval()
        self.estimator.eval()
        with torch.no_grad():
            idx = 20
            data = test_dataset[idx]
            for step, render_pose in enumerate(render_poses):
                uv, rays_o, rays_d, rgb = test_dataset.get_rays(
                    test_dataset.imgs[idx], render_pose.to("cpu").numpy()
                )

                chunk_size = 10000
                rays_o_all = torch.tensor(rays_o.astype(np.float32))
                rays_d_all = torch.tensor(rays_d.astype(np.float32))
                rgb_gt_all = torch.tensor(rgb.astype(np.float32))
                test_png = []

                for i in range(0, rays_o_all.shape[0], chunk_size):
                    rays_o = rays_o_all[i : i + chunk_size].to(self.device)
                    rays_d = rays_d_all[i : i + chunk_size].to(self.device)

                    def test_sigma_fn(t_starts, t_ends, ray_indices):
                        t_origins = rays_o[ray_indices]
                        t_directions = rays_d[ray_indices]
                        positions = (
                            t_origins
                            + t_directions * (t_starts + t_ends)[:, None] / 2.0
                        )
                        density, _ = self.radiance_field.get_density(positions)
                        return density

                    def test_rgb_sigma_fn(t_starts, t_ends, ray_indices):
                        t_origins = rays_o[ray_indices]
                        t_directions = rays_d[ray_indices]
                        positions = (
                            t_origins
                            + t_directions * (t_starts + t_ends)[:, None] / 2.0
                        )
                        rgb, density = self.radiance_field(positions, t_directions)
                        return rgb, density

                    ray_indices, t_starts, t_ends = self.estimator.sampling(
                        rays_o,
                        rays_d,
                        sigma_fn=test_sigma_fn,
                        render_step_size=render_step_size,
                        alpha_thre=alpha_thre,
                        cone_angle=cone_angle,
                        stratified=self.radiance_field.training,
                    )

                    rgb, density, depth, extras = nerfacc.rendering(
                        t_starts,
                        t_ends,
                        ray_indices,
                        n_rays=rays_o.shape[0],
                        rgb_sigma_fn=test_rgb_sigma_fn,
                        render_bkgd=render_bkgd,
                    )
                    test_png.append((rgb.cpu().numpy() * 255).astype(np.uint8))
                tqdm.write(f"step: {step}")

                test_png = np.concatenate(test_png, axis=0).reshape(
                    test_dataset.H, test_dataset.W, 3
                )
                output = cv2.cvtColor(test_png, cv2.COLOR_RGB2BGR)
                video_writer.write(output)
            
            video_writer.release()

            
