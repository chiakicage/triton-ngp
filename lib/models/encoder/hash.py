import torch
from torch.autograd import Function
from torch import nn
import math
import triton
import triton.language as tl
import numpy as np
from dataclasses import dataclass
from lib.config.task.nerf import HashEncodingConfig
from torch.cuda.amp.autocast_mode import custom_bwd, custom_fwd

import tinycudann as tcnn

BLOCK_SIZE = 128
@triton.jit
def hash_encoding_fwd_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    weight_ptr,
    index_ptr,
    resolution_ptr,
    n_rows,
    T: tl.constexpr,
    F: tl.constexpr,
    L: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    # **meta
):
    # BLOCK_SIZE = meta["BLOCK_SIZE"]
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    b_ptr = b_ptr + pid1 * T * F
    weight_ptr = weight_ptr + pid1 * n_rows * 3
    index_ptr = index_ptr + pid1 * n_rows * 8

    block_start = pid0 * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x_offsets = offsets * 3
    y_offsets = x_offsets + 1
    z_offsets = x_offsets + 2
    mask = offsets < n_rows
    N = tl.load(resolution_ptr + pid1)
    # input_mask = x_offsets < n_elements

    x = tl.load(a_ptr + x_offsets, mask=mask)
    x = x * N
    x_0 = tl.libdevice.floor(x)
    weight_x = x - x_0
    x_0 = x_0.to(tl.uint32)
    x_1 = x_0 + 1

    y = tl.load(a_ptr + y_offsets, mask=mask)
    y = y * N
    y_0 = tl.libdevice.floor(y)
    weight_y = y - y_0
    y_0 = y_0.to(tl.uint32)
    y_1 = y_0 + 1

    y_0 = y_0 * 2654435761
    y_1 = y_1 * 2654435761

    z = tl.load(a_ptr + z_offsets, mask=mask)
    z = z * N
    z_0 = tl.libdevice.floor(z)
    weight_z = z - z_0
    z_0 = z_0.to(tl.uint32)
    z_1 = z_0 + 1

    z_0 = z_0 * 805459861
    z_1 = z_1 * 805459861

    tl.store(weight_ptr + x_offsets, weight_x, mask=mask)
    tl.store(weight_ptr + y_offsets, weight_y, mask=mask)
    tl.store(weight_ptr + z_offsets, weight_z, mask=mask)

    index_000 = x_0 ^ y_0 ^ z_0
    index_001 = x_0 ^ y_0 ^ z_1
    index_010 = x_0 ^ y_1 ^ z_0
    index_011 = x_0 ^ y_1 ^ z_1
    index_100 = x_1 ^ y_0 ^ z_0
    index_101 = x_1 ^ y_0 ^ z_1
    index_110 = x_1 ^ y_1 ^ z_0
    index_111 = x_1 ^ y_1 ^ z_1

    # t = tl.full(index_000.shape, 1, dtype=tl.int32)
    # t = (t * T).to(tl.uint32)
    # t = tl.full(index_000.shape, T, dtype=tl.int32).to(tl.uint32)

    index_000 = index_000 & (T - 1)
    index_001 = index_001 & (T - 1)
    index_010 = index_010 & (T - 1)
    index_011 = index_011 & (T - 1)
    index_100 = index_100 & (T - 1)
    index_101 = index_101 & (T - 1)
    index_110 = index_110 & (T - 1)
    index_111 = index_111 & (T - 1)

    tl.store(index_ptr + offsets * 8 + 0, index_000, mask=mask)
    tl.store(index_ptr + offsets * 8 + 1, index_001, mask=mask)
    tl.store(index_ptr + offsets * 8 + 2, index_010, mask=mask)
    tl.store(index_ptr + offsets * 8 + 3, index_011, mask=mask)
    tl.store(index_ptr + offsets * 8 + 4, index_100, mask=mask)
    tl.store(index_ptr + offsets * 8 + 5, index_101, mask=mask)
    tl.store(index_ptr + offsets * 8 + 6, index_110, mask=mask)
    tl.store(index_ptr + offsets * 8 + 7, index_111, mask=mask)

    output_0_000 = tl.load(b_ptr + index_000 * 2, mask=mask)
    output_1_000 = tl.load(b_ptr + index_000 * 2 + 1, mask=mask)
    output_0_001 = tl.load(b_ptr + index_001 * 2, mask=mask)
    output_1_001 = tl.load(b_ptr + index_001 * 2 + 1, mask=mask)
    output_0_010 = tl.load(b_ptr + index_010 * 2, mask=mask)
    output_1_010 = tl.load(b_ptr + index_010 * 2 + 1, mask=mask)
    output_0_011 = tl.load(b_ptr + index_011 * 2, mask=mask)
    output_1_011 = tl.load(b_ptr + index_011 * 2 + 1, mask=mask)
    output_0_100 = tl.load(b_ptr + index_100 * 2, mask=mask)
    output_1_100 = tl.load(b_ptr + index_100 * 2 + 1, mask=mask)
    output_0_101 = tl.load(b_ptr + index_101 * 2, mask=mask)
    output_1_101 = tl.load(b_ptr + index_101 * 2 + 1, mask=mask)
    output_0_110 = tl.load(b_ptr + index_110 * 2, mask=mask)
    output_1_110 = tl.load(b_ptr + index_110 * 2 + 1, mask=mask)
    output_0_111 = tl.load(b_ptr + index_111 * 2, mask=mask)
    output_1_111 = tl.load(b_ptr + index_111 * 2 + 1, mask=mask)

    weight_x_n = 1 - weight_x
    weight_y_n = 1 - weight_y
    weight_z_n = 1 - weight_z

    output_0 = (
        output_0_000 * weight_x_n * weight_y_n * weight_z_n
        + output_0_001 * weight_x_n * weight_y_n * weight_z
        + output_0_010 * weight_x_n * weight_y * weight_z_n
        + output_0_011 * weight_x_n * weight_y * weight_z
        + output_0_100 * weight_x * weight_y_n * weight_z_n
        + output_0_101 * weight_x * weight_y_n * weight_z
        + output_0_110 * weight_x * weight_y * weight_z_n
        + output_0_111 * weight_x * weight_y * weight_z
    )
    output_1 = (
        output_1_000 * weight_x_n * weight_y_n * weight_z_n
        + output_1_001 * weight_x_n * weight_y_n * weight_z
        + output_1_010 * weight_x_n * weight_y * weight_z_n
        + output_1_011 * weight_x_n * weight_y * weight_z
        + output_1_100 * weight_x * weight_y_n * weight_z_n
        + output_1_101 * weight_x * weight_y_n * weight_z
        + output_1_110 * weight_x * weight_y * weight_z_n
        + output_1_111 * weight_x * weight_y * weight_z
    )

    # index = (x ^ (y * 2654435761) ^ (z * 805459861))
    # t = tl.full(index.shape, T, dtype=tl.int32).to(tl.uint32)
    # index = index % t
    # output_0 = tl.load(b_ptr + index * 2, mask=mask)
    # output_1 = tl.load(b_ptr + index * 2 + 1, mask=mask)

    # # tl.store(index_ptr + offsets, index, mask=mask)
    tl.store(output_ptr + offsets * 2 * L + pid1 * 2, output_0, mask=mask)
    tl.store(output_ptr + offsets * 2 * L + pid1 * 2 + 1, output_1, mask=mask)

@triton.jit
def hash_encoding_bwd_kernel(
    b_grad_ptr,
    output_ptr,
    weight_ptr,
    index_ptr,
    n_rows,
    T: tl.constexpr,
    F: tl.constexpr,
    L: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    # **meta
):
    # BLOCK_SIZE = meta["BLOCK_SIZE"]
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    b_grad_ptr = b_grad_ptr + pid1 * T * F
    weight_ptr = weight_ptr + pid1 * n_rows * 3
    index_ptr = index_ptr + pid1 * n_rows * 8

    block_start = pid0 * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_rows

    output_0 = tl.load(output_ptr + offsets * 2 * L + pid1 * 2, mask=mask)
    output_1 = tl.load(output_ptr + offsets * 2 * L + pid1 * 2 + 1, mask=mask)

    index_000 = tl.load(index_ptr + offsets * 8 + 0, mask=mask)
    index_001 = tl.load(index_ptr + offsets * 8 + 1, mask=mask)
    index_010 = tl.load(index_ptr + offsets * 8 + 2, mask=mask)
    index_011 = tl.load(index_ptr + offsets * 8 + 3, mask=mask)
    index_100 = tl.load(index_ptr + offsets * 8 + 4, mask=mask)
    index_101 = tl.load(index_ptr + offsets * 8 + 5, mask=mask)
    index_110 = tl.load(index_ptr + offsets * 8 + 6, mask=mask)
    index_111 = tl.load(index_ptr + offsets * 8 + 7, mask=mask)

    weight_x = tl.load(weight_ptr + offsets * 3 + 0, mask=mask)
    weight_y = tl.load(weight_ptr + offsets * 3 + 1, mask=mask)
    weight_z = tl.load(weight_ptr + offsets * 3 + 2, mask=mask)
    weight_x_n = 1 - weight_x
    weight_y_n = 1 - weight_y
    weight_z_n = 1 - weight_z

    output_0_000 = output_0 * weight_x_n * weight_y_n * weight_z_n
    output_0_001 = output_0 * weight_x_n * weight_y_n * weight_z
    output_0_010 = output_0 * weight_x_n * weight_y * weight_z_n
    output_0_011 = output_0 * weight_x_n * weight_y * weight_z
    output_0_100 = output_0 * weight_x * weight_y_n * weight_z_n
    output_0_101 = output_0 * weight_x * weight_y_n * weight_z
    output_0_110 = output_0 * weight_x * weight_y * weight_z_n
    output_0_111 = output_0 * weight_x * weight_y * weight_z

    output_1_000 = output_1 * weight_x_n * weight_y_n * weight_z_n
    output_1_001 = output_1 * weight_x_n * weight_y_n * weight_z
    output_1_010 = output_1 * weight_x_n * weight_y * weight_z_n
    output_1_011 = output_1 * weight_x_n * weight_y * weight_z
    output_1_100 = output_1 * weight_x * weight_y_n * weight_z_n
    output_1_101 = output_1 * weight_x * weight_y_n * weight_z
    output_1_110 = output_1 * weight_x * weight_y * weight_z_n
    output_1_111 = output_1 * weight_x * weight_y * weight_z

    tl.atomic_add(b_grad_ptr + index_000 * 2, output_0_000, mask=mask)
    tl.atomic_add(b_grad_ptr + index_000 * 2 + 1, output_1_000, mask=mask)
    tl.atomic_add(b_grad_ptr + index_001 * 2, output_0_001, mask=mask)
    tl.atomic_add(b_grad_ptr + index_001 * 2 + 1, output_1_001, mask=mask)
    tl.atomic_add(b_grad_ptr + index_010 * 2, output_0_010, mask=mask)
    tl.atomic_add(b_grad_ptr + index_010 * 2 + 1, output_1_010, mask=mask)
    tl.atomic_add(b_grad_ptr + index_011 * 2, output_0_011, mask=mask)
    tl.atomic_add(b_grad_ptr + index_011 * 2 + 1, output_1_011, mask=mask)
    tl.atomic_add(b_grad_ptr + index_100 * 2, output_0_100, mask=mask)
    tl.atomic_add(b_grad_ptr + index_100 * 2 + 1, output_1_100, mask=mask)
    tl.atomic_add(b_grad_ptr + index_101 * 2, output_0_101, mask=mask)
    tl.atomic_add(b_grad_ptr + index_101 * 2 + 1, output_1_101, mask=mask)
    tl.atomic_add(b_grad_ptr + index_110 * 2, output_0_110, mask=mask)
    tl.atomic_add(b_grad_ptr + index_110 * 2 + 1, output_1_110, mask=mask)
    tl.atomic_add(b_grad_ptr + index_111 * 2, output_0_111, mask=mask)
    tl.atomic_add(b_grad_ptr + index_111 * 2 + 1, output_1_111, mask=mask)


# T = 10000
# F = 2
# L = 4
# Nmin = 16
# Nmax = 128
BLOCK_SIZE = 128


class HashEncoding(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        hashmap: torch.Tensor,
        resolution: torch.Tensor,
        T: int,
        F: int,
        L: int,
    ):
        assert len(x.shape) == 2 and x.shape[1] == 3

        n_rows = x.shape[0]
        weight = torch.zeros((L, n_rows, 3), dtype=torch.float32).cuda()
        index = torch.zeros((L, n_rows, 8), dtype=torch.int32).cuda()
        output = torch.zeros((n_rows, L * F), dtype=torch.float32).cuda()
        grid = lambda meta: (triton.cdiv(x.shape[0], meta["BLOCK_SIZE"]), L)
        hash_encoding_fwd_kernel[grid](
            x,
            hashmap,
            output,
            weight,
            index,
            resolution,
            n_rows,
            # n_elements,
            # table_size,
            BLOCK_SIZE=BLOCK_SIZE,
            T=T,
            F=F,
            L=L,
        )
        ctx.save_for_backward(weight, index)
        ctx.grid = grid
        ctx.n_rows = n_rows
        ctx.T = T
        ctx.F = F
        ctx.L = L

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        weight, index = ctx.saved_tensors
        T = ctx.T
        F = ctx.F
        L = ctx.L
        n_rows = ctx.n_rows
        grid = ctx.grid
        b_grad = torch.zeros((L, T, F), dtype=torch.float32).cuda()
        hash_encoding_bwd_kernel[grid](
            b_grad,
            g,
            weight,
            index,
            n_rows,
            BLOCK_SIZE=BLOCK_SIZE,
            T=T,
            F=F,
            L=L,
        )
        return None, b_grad, None, None, None, None


class HashGrid(nn.Module):
    def __init__(self, cfg: HashEncodingConfig):
        super().__init__()
        self.L = cfg.L
        self.T = 2**cfg.logT
        self.F = cfg.F
        self.Nmin = cfg.Nmin
        self.Nmax = cfg.Nmax
        self.x_min = cfg.x_min
        self.x_max = cfg.x_max
        self.output_dim = self.L * self.F
        self.resolution = []
        self.scaler = math.exp(
            (math.log(self.Nmax) - math.log(self.Nmin)) / (self.L - 1)
        )
        for i in range(self.L):
            self.resolution.append(int(self.Nmin * self.scaler**i))
        self.resolution = torch.tensor(self.resolution, dtype=torch.int32).cuda()

        self.hashmap = nn.Parameter(
            torch.zeros((self.L, self.T, self.F)), requires_grad=True
        )
        nn.init.xavier_uniform_(self.hashmap)
        # self.encoder = tcnn.Encoding(
        #     n_input_dims=3,
        #     encoding_config={
        #         "otype": "HashGrid",
        #         "n_levels": cfg.L,
        #         "n_features_per_level": 2,
        #         "log2_hashmap_size": cfg.logT,
        #         "base_resolution": cfg.Nmin,
        #         "per_level_scale": self.scaler,
        #     }
        # )

    def forward(self, x):
        x = (x - self.x_min) / (self.x_max - self.x_min)
        return HashEncoding.apply(
            x, self.hashmap, self.resolution, self.T, self.F, self.L
        )
        # return self.encoder(x)


if __name__ == "__main__":
    # @dataclass
    # class HashEncodingConfig:
    #     type: str
    #     L: int
    #     F: int
    #     logT: int
    #     Nmin: int
    #     Nmax: int

    n_rows = 10
    T = 1024
    L = 4
    F = 2
    Nmin = 16
    Nmax = 128

    torch.random.manual_seed(42)
    a = torch.rand((n_rows, 3), dtype=torch.float32).cuda()
    b = torch.randn((L, T, F), dtype=torch.float16).cuda()
    c = torch.zeros((n_rows, L * F), dtype=torch.float16).cuda()
    gt = torch.randn((n_rows, L * F), dtype=torch.float16).cuda()

    hash_encoder = HashGrid(
        HashEncodingConfig(
            type="hash",
            logT=10,
            L=4,
            F=2,
            Nmin=16,
            Nmax=128,
            x_min=0,
            x_max=1,
        )
    ).cuda()

    scaler = math.exp((math.log(Nmax) - math.log(Nmin)) / (L - 1))
    resolution = []
    for i in range(L):
        resolution.append(int(Nmin * scaler**i))
    print(resolution)
    resolution = torch.tensor(resolution, dtype=torch.int32).cuda()

    # b = nn.Parameter(b, requires_grad=True)

    # c = HashEncoding.apply(a, b, resolution, T, F, L)
    c = hash_encoder(a)

    # print(c)

    loss = torch.nn.functional.mse_loss(c, gt)

    b = hash_encoder.hashmap

    print(loss)

    loss.backward()
    # print(hash_encoder.hashmap.grad)
    print(list(hash_encoder.parameters()))
    # print(b.grad)
    b_grad = hash_encoder.hashmap.grad
    # b_grad = hash_encoder.hashmap.grad

    # b = hash_encoder.hashmap.detach()

    # weight = torch.zeros((L, n_rows, 3), dtype=torch.float32).cuda()
    # index = torch.zeros((L, n_rows, 8), dtype=torch.int32).cuda()

    # # c = torch.zeros(1000, dtype=torch.float).cuda()
    # # d = 10000
    # n_elements = a.shape[0] * a.shape[1]
    # table_size = T * F
    # grid = lambda meta: (triton.cdiv(a.shape[0], meta["BLOCK_SIZE"]), L)
    # # grid = (a.shape[0], )
    # print(a)
    # hash_encoding_fwd_kernel[grid](
    #     a,
    #     b,
    #     c,
    #     weight,
    #     index,
    #     resolution,
    #     n_rows,
    #     # n_elements,
    #     # table_size,
    #     BLOCK_SIZE=256,
    #     T=T,
    #     F=F,
    #     L=L,
    # )
    # # print(index)
    # print(c)
    # c.requires_grad_(True)
    # loss = torch.nn.functional.mse_loss(c, gt)
    # c.retain_grad()
    # loss.backward()
    # print(c.grad)

    # b_grad = torch.zeros((L, T, F), dtype=torch.float16).cuda()

    # hash_encoding_bwd_kernel[grid](
    #     b_grad,
    #     c.grad,
    #     weight,
    #     index,
    #     n_rows,
    #     BLOCK_SIZE=256,
    #     T=T,
    #     F=F,
    #     L=L,
    # )

    # print(weight)
    # print(index)
    a = a.cpu().numpy()
    b = b.detach().cpu().numpy()

    b = torch.tensor(b, requires_grad=True, dtype=torch.float16).cuda()
    b = nn.Parameter(b, requires_grad=True)
    # # c = c.cpu().numpy()
    x = a[:, 0]
    y = a[:, 1]
    z = a[:, 2]

    results = []
    indexes = []

    for i in range(L):
        nx = x * resolution[i].cpu().numpy()
        ny = y * resolution[i].cpu().numpy()
        nz = z * resolution[i].cpu().numpy()
        x_0 = np.floor(nx)
        y_0 = np.floor(ny)
        z_0 = np.floor(nz)
        weight_x = torch.tensor(nx - x_0).cuda()
        weight_y = torch.tensor(ny - y_0).cuda()
        weight_z = torch.tensor(nz - z_0).cuda()
        x_0 = x_0.astype(np.uint32)
        y_0 = y_0.astype(np.uint32)
        z_0 = z_0.astype(np.uint32)
        x_1 = x_0 + 1
        y_1 = y_0 + 1
        z_1 = z_0 + 1

        index_000 = (x_0 ^ (y_0 * 2654435761) ^ (z_0 * 805459861)) % T
        index_001 = (x_0 ^ (y_0 * 2654435761) ^ (z_1 * 805459861)) % T
        index_010 = (x_0 ^ (y_1 * 2654435761) ^ (z_0 * 805459861)) % T
        index_011 = (x_0 ^ (y_1 * 2654435761) ^ (z_1 * 805459861)) % T
        index_100 = (x_1 ^ (y_0 * 2654435761) ^ (z_0 * 805459861)) % T
        index_101 = (x_1 ^ (y_0 * 2654435761) ^ (z_1 * 805459861)) % T
        index_110 = (x_1 ^ (y_1 * 2654435761) ^ (z_0 * 805459861)) % T
        index_111 = (x_1 ^ (y_1 * 2654435761) ^ (z_1 * 805459861)) % T

        weight_x_n = 1 - weight_x
        weight_y_n = 1 - weight_y
        weight_z_n = 1 - weight_z

        index_000 = torch.tensor(index_000.astype(np.int32), dtype=torch.int32).cuda()
        index_001 = torch.tensor(index_001.astype(np.int32), dtype=torch.int32).cuda()
        index_010 = torch.tensor(index_010.astype(np.int32), dtype=torch.int32).cuda()
        index_011 = torch.tensor(index_011.astype(np.int32), dtype=torch.int32).cuda()
        index_100 = torch.tensor(index_100.astype(np.int32), dtype=torch.int32).cuda()
        index_101 = torch.tensor(index_101.astype(np.int32), dtype=torch.int32).cuda()
        index_110 = torch.tensor(index_110.astype(np.int32), dtype=torch.int32).cuda()
        index_111 = torch.tensor(index_111.astype(np.int32), dtype=torch.int32).cuda()

        output_000 = b[i][index_000]
        output_001 = b[i][index_001]
        output_010 = b[i][index_010]
        output_011 = b[i][index_011]
        output_100 = b[i][index_100]
        output_101 = b[i][index_101]
        output_110 = b[i][index_110]
        output_111 = b[i][index_111]

        # print(index_000)

        output = (
            output_000 * (weight_x_n * weight_y_n * weight_z_n).reshape((-1, 1))
            + output_001 * (weight_x_n * weight_y_n * weight_z).reshape((-1, 1))
            + output_010 * (weight_x_n * weight_y * weight_z_n).reshape((-1, 1))
            + output_011 * (weight_x_n * weight_y * weight_z).reshape((-1, 1))
            + output_100 * (weight_x * weight_y_n * weight_z_n).reshape((-1, 1))
            + output_101 * (weight_x * weight_y_n * weight_z).reshape((-1, 1))
            + output_110 * (weight_x * weight_y * weight_z_n).reshape((-1, 1))
            + output_111 * (weight_x * weight_y * weight_z).reshape((-1, 1))
        ).to(torch.float16)
        results.append(output)
        index = torch.stack(
            [
                index_000,
                index_001,
                index_010,
                index_011,
                index_100,
                index_101,
                index_110,
                index_111,
            ],
            dim=-1,
        )
        indexes.append(index)
        # print(index)
        # print(
        #     torch.tensor(np.stack(
        #         [
        #             weight_x,
        #             weight_y,
        #             weight_z,
        #         ],
        #         axis=-1,
        #     ))
        # )

    # results = torch.tensor(np.concatenate(results, axis=-1))
    results = torch.cat(results, dim=-1)
    loss = torch.nn.functional.mse_loss(results, gt)

    print(results)
    print(loss)
    # b.retain_grad()
    results.retain_grad()
    loss.backward()
    if results.grad is not None:
        print(results.grad.shape)
        print(results.grad)
    if b.grad is not None:
        print(b.grad.shape)
        # print(b.grad)
        print(indexes[0][:, 0])
        print(b.grad[0][indexes[0][:, 0]])
        print(b_grad[0][indexes[0][:, 0]])
        print(torch.nn.functional.mse_loss(b.grad, b_grad))
    # print(torch.tensor(results))

    # index = (x ^ (y * 2654435761) ^ (z * 805459861)) % T
    # print(index)
    # results = []
    # for i in range(L):
    #     results.append(b[i][index])
    # results = np.concatenate(results, axis=-1)
    # print(results)
    # print(b[index])
