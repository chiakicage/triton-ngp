import torch
from torch.autograd import Function
from torch import nn
import math
import triton
import triton.language as tl
import numpy as np
from dataclasses import dataclass
# from lib.config.task.nerf import HashEncodingConfig
from torch.cuda.amp.autocast_mode import custom_bwd, custom_fwd, autocast
import tinycudann as tcnn
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

@dataclass
class HashEncodingConfig:
    type: str
    L: int
    F: int
    logT: int
    Nmin: int
    Nmax: int
    x_min: float
    x_max: float

BLOCK_SIZE = 512

# @ti.kernel
# def hash_encoding_fwd_kernel(
#     x: ti.template(),
#     hashmap: ti.template(),
# ) -> ti.types.ndarray:
#     L = hashmap.shape[0]
#     for i in range(L):
#         pass



# T = 10000
# F = 2
# L = 4
# Nmin = 16
# Nmax = 128



class HashEncoding(Function):
    pass
    # @staticmethod
    # @custom_fwd
    # def forward(
    #     ctx,
    #     x: torch.Tensor,
    #     hashmap: torch.Tensor,
    #     resolution: torch.Tensor,
    #     T: int,
    #     F: int,
    #     L: int,
    # ):
    #     assert len(x.shape) == 2 and x.shape[1] == 3

    #     n_rows = x.shape[0]
    #     # weight = torch.zeros((L, n_rows, 3), dtype=torch.float32).cuda()
    #     # index = torch.zeros((L, n_rows, 8), dtype=torch.int32).cuda()
    #     output = torch.zeros(n_rows * L * F, dtype=torch.float32).cuda()
    #     grid = lambda meta: (triton.cdiv(x.shape[0], meta["BLOCK_SIZE"]), L)
    #     hash_encoding_fwd_kernel[grid](
    #         x,
    #         hashmap,
    #         output,
    #         # weight,
    #         # index,
    #         resolution,
    #         n_rows,
    #         # n_elements,
    #         # table_size,
    #         BLOCK_SIZE=512,
    #         T=T,
    #         F=F,
    #         L=L,
    #     )
    #     ctx.save_for_backward(x, resolution)
    #     ctx.grid = grid
    #     ctx.n_rows = n_rows
    #     ctx.T = T
    #     ctx.F = F
    #     ctx.L = L
    #     return output.view(n_rows, L * F)

    # @staticmethod
    # @custom_bwd
    # def backward(ctx, g):
    #     x, resolution = ctx.saved_tensors
    #     T = ctx.T
    #     F = ctx.F
    #     L = ctx.L
    #     n_rows = ctx.n_rows
    #     grid = ctx.grid
    #     b_grad = torch.zeros(L * T * F, dtype=torch.float32).cuda()
    #     hash_encoding_bwd_kernel[grid](
    #         x,
    #         b_grad,
    #         g,
    #         resolution,
    #         # weight,
    #         # index,
    #         n_rows,
    #         BLOCK_SIZE=512,
    #         T=T,
    #         F=F,
    #         L=L,
    #     )
    #     return None, b_grad, None, None, None, None


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

        # self.hashmap = nn.Parameter(
        #     torch.zeros(self.L * self.T * self.F), requires_grad=True
        # )
        # nn.init.xavier_uniform_(self.hashmap.view(self.L, self.T, self.F))
        self.hook = nn.Parameter(
            torch.tensor(0), requires_grad=True
        )
        self.hashmap = ti.field(ti.f32)
        levels = ti.root.pointer(ti.i, self.L)
        levels.dense(ti.i, self.T).dense(ti.i, self.F).place(self.hashmap)

        print(self.hashmap.shape)
        
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
        # return HashEncoding.apply(
        #     x, self.hashmap, self.resolution, self.T, self.F, self.L
        # )
        # return self.encoder(x)
    


# @ti.func
# def get_index(
#     x: ti.template(),
#     y: ti.template(),
#     z: ti.template(),
#     n_rows: ti.i32,
#     N: ti.i32,
#     T: ti.u32,
# ) -> ti.template():
#     index = ti.field(ti.u32, shape=(n_rows,))
#     for i in range(n_rows):
#         index[i] = (a[0] ^ (a[1] * 2654435761) ^ (a[2] * 805459861)) % T
#     return index

@ti.kernel
def init_hashmap(
    hashmap: ti.template(),
    F: ti.i32,
    T: ti.i32,
    L: ti.i32,
):
    for i, j in hashmap:
        for k in range(F):
            hashmap[i, j][k] = ti.randn()

@ti.kernel
def hash_encoding_fwd_kernel(
    a: ti.template(), # (3, n_row)
    hashmap: ti.template(),
    result: ti.template(),
    F: int,
    T: int,
    L: int,
    resolution: ti.template(),
):
    
    for i in range(L):
        N = ti.cast(resolution[i], ti.i32)
        # x = x * N
        # for j in range(x.shape[0]):
        #     x[j] = x[j] * N
        # a = ti.Vector.field(x.shape[0], ti.f32, shape=(3, ))
        x = a[0] * N
        x_0 = ti.floor(x, dtype=ti.f32)
        weight_x = x - x_0
        xn = ti.cast(x_0, ti.u32)

        y = a[1] * N
        y_0 = ti.floor(y, dtype=ti.f32)
        weight_y = y - y_0
        yn = ti.cast(y_0, ti.u32)

        z = a[2] * N
        z_0 = ti.floor(z, dtype=ti.f32)
        weight_z = z - z_0
        zn = ti.cast(z_0, ti.u32)
        
        index_000 = (xn ^ (yn * ti.u32(2654435761)) ^ (zn * ti.u32(805459861))) % T
        index_001 = (xn ^ (yn * ti.u32(2654435761)) ^ ((zn + 1) * ti.u32(805459861))) % T
        index_010 = (xn ^ ((yn + 1) * ti.u32(2654435761)) ^ (zn * ti.u32(805459861))) % T
        index_011 = (xn ^ ((yn + 1) * ti.u32(2654435761)) ^ ((zn + 1) * ti.u32(805459861))) % T
        index_100 = ((xn + 1) ^ (yn * ti.u32(2654435761)) ^ (zn * ti.u32(805459861))) % T
        index_101 = ((xn + 1) ^ (yn * ti.u32(2654435761)) ^ ((zn + 1) * ti.u32(805459861))) % T
        index_110 = ((xn + 1) ^ ((yn + 1) * ti.u32(2654435761)) ^ (zn * ti.u32(805459861))) % T
        index_111 = ((xn + 1) ^ ((yn + 1) * ti.u32(2654435761)) ^ ((zn + 1) * ti.u32(805459861))) % T

        weight_x_n = 1 - weight_x
        weight_y_n = 1 - weight_y
        weight_z_n = 1 - weight_z
        
        output_000 = hashmap[i, index_000] * weight_x_n * weight_y_n * weight_z_n
        output_001 = hashmap[i, index_001] * weight_x_n * weight_y_n * weight_z
        output_010 = hashmap[i, index_010] * weight_x_n * weight_y * weight_z_n
        output_011 = hashmap[i, index_011] * weight_x_n * weight_y * weight_z
        output_100 = hashmap[i, index_100] * weight_x * weight_y_n * weight_z_n
        output_101 = hashmap[i, index_101] * weight_x * weight_y_n * weight_z
        output_110 = hashmap[i, index_110] * weight_x * weight_y * weight_z_n
        output_111 = hashmap[i, index_111] * weight_x * weight_y * weight_z
        output = output_000 + output_001 + output_010 + output_011 + output_100 + output_101 + output_110 + output_111
        for j in range(n_rows):
            for k in range(F):
                result[i][j * F + k] = output[j][k]
    




if __name__ == "__main__":
    
    import time

    n_rows = 10
    logT = 10
    T = 1 << logT
    L = 4
    F = 2
    Nmin = 16
    Nmax = 128
    scaler = math.exp((math.log(Nmax) - math.log(Nmin)) / (L - 1))
    resolution = []
    for i in range(L):
        resolution.append(int(Nmin * scaler**i))
    print(resolution)
    resolution = torch.tensor(resolution, dtype=torch.int32).cuda()
    r = ti.field(ti.i32, shape=(L, ))
    r.from_torch(resolution)
    hashmap = ti.Vector.field(F, ti.f32)
    levels = ti.root.dense(ti.i, L)
    levels.dense(ti.j, T).place(hashmap)
    a = torch.rand((n_rows, 3), dtype=torch.float32).cuda()
    x = ti.Vector.field(n_rows, ti.f32, shape=(3, ))
    x.from_torch(a.T)
    print(x)
    print(hashmap.shape)
    print(x.shape)
    init_hashmap(hashmap, F, T, L)
    a = hashmap[0, 0]
    result = ti.Vector.field(F * L, ti.f32, shape=(x.shape[0], ))
    hash_encoding_fwd_kernel(x, hashmap, result, F, T, L, r)
    # print(hashmap[0, 0])
    # print(hashmap)
#########################

    # torch.random.manual_seed(42)
    # a = torch.rand((n_rows, 3), dtype=torch.float32).cuda()
    # b = torch.randn((L, T, F), dtype=torch.float32).cuda()
    # # c = torch.zeros((n_rows, L * F), dtype=torch.float32).cuda()
    

    # hash_encoder = HashGrid(
    #     HashEncodingConfig(
    #         type="hash",
    #         logT=logT,
    #         L=L,
    #         F=F,
    #         Nmin=Nmin,
    #         Nmax=Nmax,
    #         x_min=0,
    #         x_max=1,
    #     )
    # ).cuda()

    # scaler = math.exp((math.log(Nmax) - math.log(Nmin)) / (L - 1))
    # resolution = []
    # for i in range(L):
    #     resolution.append(int(Nmin * scaler**i))
    # print(resolution)
    # resolution = torch.tensor(resolution, dtype=torch.int32).cuda()

    # # b = nn.Parameter(b, requires_grad=True)

    # # c = HashEncoding.apply(a, b, resolution, T, F, L)

    # start = time.time()
    # with autocast():
    #     c = hash_encoder(a)
    #     gt = torch.rand_like(c)
    #     # print(c)

    # fwd_end = time.time()

    # # print(c)

    # loss = torch.nn.functional.mse_loss(c, gt)

    # # b = hash_encoder.hashmap

    # # print(loss)

    # loss.backward()

    # bwd_end = time.time()

    # print(f'Forward time: {fwd_end - start}')
    # print(f'Backward time: {bwd_end - fwd_end}')

############################

    # b = list(hash_encoder.named_parameters())
    # print(b)
    # print(hash_encoder.hashmap.grad)
    # print(list(hash_encoder.parameters()))
    # print(b.grad)
    # b_grad = hash_encoder.hashmap.grad
    # b_grad = hash_encoder.hashmap.grad

    # b = hash_encoder.hashmap

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

###################################

    # a = a.cpu().numpy()
    # b = b.detach().cpu().numpy()

    # b = torch.tensor(b, requires_grad=True, dtype=torch.float32).cuda()
    # b = nn.Parameter(b, requires_grad=True)
    # # # c = c.cpu().numpy()
    # x = a[:, 0]
    # y = a[:, 1]
    # z = a[:, 2]

    # results = []
    # indexes = []

    # for i in range(L):
    #     nx = x * resolution[i].cpu().numpy()
    #     ny = y * resolution[i].cpu().numpy()
    #     nz = z * resolution[i].cpu().numpy()
    #     x_0 = np.floor(nx)
    #     y_0 = np.floor(ny)
    #     z_0 = np.floor(nz)
    #     weight_x = torch.tensor(nx - x_0).cuda()
    #     weight_y = torch.tensor(ny - y_0).cuda()
    #     weight_z = torch.tensor(nz - z_0).cuda()
    #     x_0 = x_0.astype(np.uint32)
    #     y_0 = y_0.astype(np.uint32)
    #     z_0 = z_0.astype(np.uint32)
    #     x_1 = x_0 + 1
    #     y_1 = y_0 + 1
    #     z_1 = z_0 + 1

    #     index_000 = (x_0 ^ (y_0 * 2654435761) ^ (z_0 * 805459861)) % T
    #     index_001 = (x_0 ^ (y_0 * 2654435761) ^ (z_1 * 805459861)) % T
    #     index_010 = (x_0 ^ (y_1 * 2654435761) ^ (z_0 * 805459861)) % T
    #     index_011 = (x_0 ^ (y_1 * 2654435761) ^ (z_1 * 805459861)) % T
    #     index_100 = (x_1 ^ (y_0 * 2654435761) ^ (z_0 * 805459861)) % T
    #     index_101 = (x_1 ^ (y_0 * 2654435761) ^ (z_1 * 805459861)) % T
    #     index_110 = (x_1 ^ (y_1 * 2654435761) ^ (z_0 * 805459861)) % T
    #     index_111 = (x_1 ^ (y_1 * 2654435761) ^ (z_1 * 805459861)) % T

    #     weight_x_n = 1 - weight_x
    #     weight_y_n = 1 - weight_y
    #     weight_z_n = 1 - weight_z

    #     index_000 = torch.tensor(index_000.astype(np.int32), dtype=torch.int32).cuda()
    #     index_001 = torch.tensor(index_001.astype(np.int32), dtype=torch.int32).cuda()
    #     index_010 = torch.tensor(index_010.astype(np.int32), dtype=torch.int32).cuda()
    #     index_011 = torch.tensor(index_011.astype(np.int32), dtype=torch.int32).cuda()
    #     index_100 = torch.tensor(index_100.astype(np.int32), dtype=torch.int32).cuda()
    #     index_101 = torch.tensor(index_101.astype(np.int32), dtype=torch.int32).cuda()
    #     index_110 = torch.tensor(index_110.astype(np.int32), dtype=torch.int32).cuda()
    #     index_111 = torch.tensor(index_111.astype(np.int32), dtype=torch.int32).cuda()

    #     output_000 = b[i][index_000]
    #     output_001 = b[i][index_001]
    #     output_010 = b[i][index_010]
    #     output_011 = b[i][index_011]
    #     output_100 = b[i][index_100]
    #     output_101 = b[i][index_101]
    #     output_110 = b[i][index_110]
    #     output_111 = b[i][index_111]

    #     # print(index_000)

    #     output = (
    #         output_000 * (weight_x_n * weight_y_n * weight_z_n).reshape((-1, 1))
    #         + output_001 * (weight_x_n * weight_y_n * weight_z).reshape((-1, 1))
    #         + output_010 * (weight_x_n * weight_y * weight_z_n).reshape((-1, 1))
    #         + output_011 * (weight_x_n * weight_y * weight_z).reshape((-1, 1))
    #         + output_100 * (weight_x * weight_y_n * weight_z_n).reshape((-1, 1))
    #         + output_101 * (weight_x * weight_y_n * weight_z).reshape((-1, 1))
    #         + output_110 * (weight_x * weight_y * weight_z_n).reshape((-1, 1))
    #         + output_111 * (weight_x * weight_y * weight_z).reshape((-1, 1))
    #     ).to(torch.float32)
    #     results.append(output)
    #     index = torch.stack(
    #         [
    #             index_000,
    #             index_001,
    #             index_010,
    #             index_011,
    #             index_100,
    #             index_101,
    #             index_110,
    #             index_111,
    #         ],
    #         dim=-1,
    #     )
    #     indexes.append(index)
    #     # print(index)
    #     # print(
    #     #     torch.tensor(np.stack(
    #     #         [
    #     #             weight_x,
    #     #             weight_y,
    #     #             weight_z,
    #     #         ],
    #     #         axis=-1,
    #     #     ))
    #     # )

    # # results = torch.tensor(np.concatenate(results, axis=-1))
    # results = torch.cat(results, dim=-1)
    # loss = torch.nn.functional.mse_loss(results, gt)

    # print(results)
    # print(loss)
    # # b.retain_grad()
    # results.retain_grad()
    # loss.backward()
    # if results.grad is not None:
    #     print(results.grad.shape)
    #     print(results.grad)
    # if b.grad is not None:
    #     print(b.grad.shape)
    #     # print(b.grad)
    #     print(indexes[0][:, 0])
    #     print(b.grad[0][indexes[0][:, 0]])
    #     print(b_grad[0][indexes[0][:, 0]])
    #     print(torch.nn.functional.mse_loss(b.grad, b_grad))

###################
    # print(torch.tensor(results))

    # index = (x ^ (y * 2654435761) ^ (z * 805459861)) % T
    # print(index)
    # results = []
    # for i in range(L):
    #     results.append(b[i][index])
    # results = np.concatenate(results, axis=-1)
    # print(results)
    # print(b[index])
