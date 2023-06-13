import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
import nerfacc
import logging
from lib.models.nerf import Network
from lib.config.task import TaskConfig
import imageio
from plyfile import PlyData, PlyElement

logger = logging.getLogger("trainer")


class Trainer:
    def __init__(self, radiance_field: Network, cfg: TaskConfig):
        self.device = cfg.device
        self.radiance_field = radiance_field
        self.radiance_field = self.radiance_field.to(self.device)
        self.estimator = nerfacc.OccGridEstimator(
            roi_aabb=torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]).to(self.device),
        ).to(self.device)
        self.optimizer = Adam(
            self.radiance_field.parameters(),
            lr=5e-4,
        )
        self.max_steps = 50000
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[
                self.max_steps // 2,
                self.max_steps * 3 // 4,
                self.max_steps * 5 // 6,
                self.max_steps * 9 // 10,
            ],
            gamma=0.33,
        )

    def train(self, train_dataset, test_dataset):
        render_step_size = 5e-3
        alpha_thre = 0.0
        cone_angle = 0.0
        render_bkgd = torch.ones(3, device=self.device)
        max_steps = self.max_steps
        eval_step = 1000
        target_sample_batch_size = 1 << 18

        # model_save_path = "results/model.pth"
        # checkpoint = torch.load(model_save_path)
        # init_step = checkpoint["step"]
        # self.radiance_field.load_state_dict(checkpoint["radiance_field_state_dict"])
        # self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # self.estimator.load_state_dict(checkpoint["estimator_state_dict"])

        for step in tqdm(range(max_steps)):
            self.radiance_field.train()
            self.estimator.train()
            i = torch.randint(0, len(train_dataset), (1,)).item()
            data = train_dataset[i]
            # data = torch.rand.
            # data = { k: v.to(self.device) for k, v in data.items() }
            rays_o = torch.tensor(data["rays_o"]).to(self.device)
            rays_d = torch.tensor(data["rays_d"]).to(self.device)
            rgb_gt = torch.tensor(data["rgb"]).to(self.device)
            # logger.info(f"rays_o: {rays_o}, rays_d: {rays_d}, rgb_gt: {rgb_gt}")

            def sigma_fn(t_starts, t_ends, ray_indices):
                t_origins = rays_o[ray_indices]
                t_directions = rays_d[ray_indices]
                positions = (
                    t_origins + t_directions * (t_starts + t_ends)[:, None] / 2.0
                )
                density, _ = self.radiance_field.get_density(positions)
                return density

            def rgb_sigma_fn(t_starts, t_ends, ray_indices):
                t_origins = rays_o[ray_indices]
                t_directions = rays_d[ray_indices]
                positions = (
                    t_origins + t_directions * (t_starts + t_ends)[:, None] / 2.0
                )
                rgb, density = self.radiance_field(positions, t_directions)
                return rgb, density

            def occ_eval_fn(x):
                # logger.info(f"occ_eval_fn: x: {x}")
                density, _ = self.radiance_field.get_density(x)
                # logger.info(f"density: {density * render_step_size}")
                return density * render_step_size

            self.estimator.update_every_n_steps(
                step=step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=1e-2,
            )
            # t_starts = torch.zeros(rays_o.shape[0], device=rays_o.device)
            # t_ends = torch.ones(rays_o.shape[0], device=rays_o.device)
            # ray_indices = torch.arange(rays_o.shape[0], device=rays_o.device)
            # density = sigma_fn(t_starts, t_ends, ray_indices)
            # logger.info(f"density: {density}")
            #     # logger.info(f"rays_o: {rays_o}, rays_d: {rays_d}")
            ray_indices, t_starts, t_ends = self.estimator.sampling(
                rays_o,
                rays_d,
                sigma_fn=sigma_fn,
                render_step_size=render_step_size,
                alpha_thre=alpha_thre,
                cone_angle=cone_angle,
                stratified=self.radiance_field.training,
            )
            if ray_indices.shape[0] == 0:
                continue
            # logger.info(f"ray_indices: {ray_indices.shape[0]}")
            #     logger.info(f"indices: {ray_indices.shape[0]} t_starts: {t_starts.shape[0]} t_ends: {t_ends.shape[0]}")
            #     # ray_indices = torch.arange(rays_o.shape[0], device=rays_o.device)
            #     # t_starts = torch.repeat_interleave(
            #     #     torch.tensor([-1.0], device=rays_o.device), repeats=rays_o.shape[0]
            #     # )
            #     # t_ends = torch.repeat_interleave(
            #     #     torch.tensor([1.0], device=rays_o.device), repeats=rays_o.shape[0]
            #     # )
            #     # logger.info(
            #     #     f"ray_indices: {ray_indices}, t_starts: {t_starts}, t_ends: {t_ends}"
            #     # )
            rgb, density, depth, extras = nerfacc.rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=rays_o.shape[0],
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=render_bkgd,
            )
            samples = len(ray_indices)
            tqdm.write(f"samples: {samples}")
            rays_num = rays_o.shape[0]
            new_rays_num = int(rays_num * (target_sample_batch_size / samples))
            train_dataset.update_rays_num(new_rays_num)

            #     # logger.info(f"rgb: {rgb}, density: {density}, depth: {depth}, extras: {extras}")
            #     return rgb, density, depth, extras
            # uv = data['uv']
            # logger.info(f"rays_o: {rays_o}")
            # logger.info(f"rays_d: {rays_d}")
            # logger.info(f"rgb_gt: {rgb_gt}")

            # rgb, density, _, _ = self.radiance_field(rays_o, rays_d, self.estimator)
            # logger.info(f"rgb: {rgb}")
            # print(rgb)
            # logger.info(f"density: {density}")
            self.optimizer.zero_grad()
            loss = F.mse_loss(rgb, rgb_gt)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            pnsr = -10.0 * torch.log(loss) / np.log(10.0)
            # logger.info(f"loss: {loss.item()}")
            tqdm.write(f"loss: {loss.item()} pnsr: {pnsr.item()}")
            if (step + 1) % eval_step == 0:
                model_save_path = "results/model.pth"
                torch.save(
                    {
                        "step": step,
                        "radiance_field_state_dict": self.radiance_field.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "estimator_state_dict": self.estimator.state_dict(),
                    },
                    model_save_path,
                )
                self.radiance_field.eval()
                self.estimator.eval()
                with torch.no_grad():
                    # i = torch.randint(0, len(test_dataset), (1,)).item()
                    i = 20
                    data = test_dataset[i]
                    # print(test_dataset.H)
                    # print(test_dataset.imgs[i].shape)
                    chunk_size = 10000
                    rays_o_all = torch.tensor(data["rays_o"])
                    rays_d_all = torch.tensor(data["rays_d"])
                    rgb_gt_all = torch.tensor(data["rgb"])
                    # print(rays_o_all.shape)
                    test_png = []
                    losses = []
                    samples = []
                    rays_sample = np.random.choice(rays_o_all.shape[0], 2000)
                    rays_select = np.arange(
                        test_dataset.H // 2 * test_dataset.W,
                        test_dataset.H // 2 * test_dataset.W + test_dataset.W ,
                        1,
                        dtype=np.int32,
                    )
                    # rays_select = [
                    #     test_dataset.H // 2 * test_dataset.W + test_dataset.W // 16,
                    #     test_dataset.H // 2 * test_dataset.W + test_dataset.W // 8,
                    #     test_dataset.H // 2 * test_dataset.W + test_dataset.W // 4,
                    # ]
                    # print(rays_sample, rays_select)

                    select = []
                    graph = np.zeros((test_dataset.H, test_dataset.W)).reshape(-1)
                    for i in range(0, rays_o_all.shape[0], chunk_size):
                        rays_o = rays_o_all[i : i + chunk_size].to(self.device)
                        rays_d = rays_d_all[i : i + chunk_size].to(self.device)
                        rgb_gt = rgb_gt_all[i : i + chunk_size].to(self.device)

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

                        # print(rays_o.shape)
                        ray_indices, t_starts, t_ends = self.estimator.sampling(
                            rays_o,
                            rays_d,
                            sigma_fn=test_sigma_fn,
                            render_step_size=render_step_size,
                            alpha_thre=alpha_thre,
                            cone_angle=cone_angle,
                            stratified=self.radiance_field.training,
                        )

                        # print(ray_indices)
                        # mask = torch.repeat_interleave(
                        #     torch.tensor(False), ray_indices.shape[0]
                        # ).to(self.device)
                        # for ray in rays_sample:
                        #     mask = torch.logical_or(mask, ray_indices == (ray - i))
                        # t_origins = rays_o[ray_indices[mask]]
                        # t_directions = rays_d[ray_indices[mask]]
                        # positions = (
                        #     t_origins
                        #     + t_directions
                        #     * (t_starts[mask] + t_ends[mask])[:, None]
                        #     / 2.0
                        # )
                        # # print(positions.shape)
                        # samples.append(positions.cpu().numpy())

                        # pixel_indices = np.unique(ray_indices.cpu().numpy()) + i
                        # # print(pixel_indices)
                        # graph[pixel_indices] = 1
                        # # print(np.unique(ray_indices.cpu().numpy()).max())

                        # mask = torch.repeat_interleave(
                        #     torch.tensor(False), ray_indices.shape[0]
                        # ).to(self.device)
                        # # print(ray_indices)
                        # for ray in rays_select:
                        #     mask = torch.logical_or(mask, ray_indices == (ray - i))
                        #     # print((ray_indices == ray).sum())
                        # t_origins = rays_o[ray_indices[mask]]
                        # t_directions = rays_d[ray_indices[mask]]
                        # positions = (
                        #     t_origins
                        #     + t_directions
                        #     * (t_starts[mask] + t_ends[mask])[:, None]
                        #     / 2.0
                        # )
                        # # print(positions.shape)
                        # select.append(positions.cpu().numpy())

                        # print(ray_indices.shape)
                        rgb, density, depth, extras = nerfacc.rendering(
                            t_starts,
                            t_ends,
                            ray_indices,
                            n_rays=rays_o.shape[0],
                            rgb_sigma_fn=test_rgb_sigma_fn,
                            render_bkgd=render_bkgd,
                        )
                        loss = F.mse_loss(rgb, rgb_gt)
                        # print(f"eval loss: {loss.item()}")
                        losses.append(loss.item())
                        # pnsr = -10.0 * torch.log(loss) / np.log(10.0)
                        test_png.append((rgb.cpu().numpy() * 255).astype(np.uint8))
                    # samples = np.concatenate(samples, axis=0)
                    # vertex = np.array(
                    #     list(map(tuple, samples)),
                    #     dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
                    # )
                    # el = PlyElement.describe(vertex, "vertex")
                    # PlyData([el]).write(f"results/sample_binary_{step+1}.ply")
                    # select = np.concatenate(select, axis=0)
                    # vertex = np.array(
                    #     list(map(tuple, select)),
                    #     dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
                    # )
                    # el = PlyElement.describe(vertex, "vertex")
                    # PlyData([el]).write(f"results/select_binary_{step+1}.ply")
                    tqdm.write(f"eval loss: {np.mean(losses)}")

                    test_png = np.concatenate(test_png, axis=0).reshape(
                        test_dataset.H, test_dataset.W, 3
                    )
                    gt_png = (
                        (rgb_gt_all.cpu().numpy() * 255)
                        .astype(np.uint8)
                        .reshape(test_dataset.H, test_dataset.W, 3)
                    )
                    # graph = np.stack([graph, graph, graph], axis=-1)
                    # pixel = (
                    #     (graph * 255)
                    #     .astype(np.uint8)
                    #     .reshape(test_dataset.H, test_dataset.W, 3)
                    # )

                    output = np.concatenate([test_png, gt_png], axis=1)
                    imageio.imwrite(f"results/test_{step+1}.png", output)
            # return
            # print(f'loss: {loss.item()}')
