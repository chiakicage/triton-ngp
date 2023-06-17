import torch
from torch.autograd import Function
from torch import nn
import math
import triton
import triton.language as tl
import numpy as np

@triton.jit
def mod_kernel(
    a_ptr, 
    b_ptr,
    output_ptr,
    # index_ptr,
    n_elements,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
    T: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x_offsets = offsets * 3
    y_offsets = x_offsets + 1
    z_offsets = x_offsets + 2
    mask = offsets < n_rows
    # input_mask = x_offsets < n_elements
    x = tl.load(a_ptr + x_offsets, mask=mask).to(tl.uint32)
    y = tl.load(a_ptr + y_offsets, mask=mask).to(tl.uint32)
    z = tl.load(a_ptr + z_offsets, mask=mask).to(tl.uint32)
    index = (x ^ (y * 2654435761) ^ (z * 805459861))
    t = tl.full(index.shape, T, dtype=tl.int32).to(tl.uint32)
    index = index % t
    output_0 = tl.load(b_ptr + index * 2, mask=mask)
    output_1 = tl.load(b_ptr + index * 2 + 1, mask=mask)

    # tl.store(index_ptr + offsets, index, mask=mask)
    tl.store(output_ptr + offsets * 2, output_0, mask=mask)
    tl.store(output_ptr + offsets * 2 + 1, output_1, mask=mask)

    



class HashEncoding(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        print(x)
        return g * torch.exp(x.clamp(-15, 15))
    
class HashGrid(nn.Module):
    def __init__(
            self, 
            level: int, 
            logT: int, 
            F: int, 
            Nmin: int, 
            Nmax: int
        ):
        super().__init__()
        self.level = level
        self.T = 2**logT
        self.F = F
        self.Nmin = Nmin
        self.Nmax = Nmax
        self.resolution = []
        self.scaler = math.exp((math.log(self.Nmax) - math.log(self.Nmin)) / (self.level - 1))
        for i in range(self.level):
            self.resolution.append(int(self.Nmin * self.scaler**i))
        print(self.resolution)
    def forward(x):
        pass

if __name__ == "__main__":
    n_rows = 10
    T = 10000
    a = torch.randint(0, 1000, (n_rows, 3), dtype=torch.int32).cuda()
    b = torch.randn((10000, 2), dtype=torch.float32).cuda()
    c = torch.zeros((n_rows, 2), dtype=torch.float32).cuda()
    index = torch.zeros(n_rows, dtype=torch.int32).cuda()
    # c = torch.zeros(1000, dtype=torch.float).cuda()
    # d = 10000
    n_elements = a.shape[0] * a.shape[1]
    grid = lambda meta: (triton.cdiv(a.shape[0], meta['BLOCK_SIZE']), )
    # grid = (a.shape[0], )
    print(a)
    mod_kernel[grid](
        a, b, c,
        # index, 
        n_elements, 
        n_rows,
        
        BLOCK_SIZE=256, T=T, F=2
    )
    print(index)
    print(c)
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    # c = c.cpu().numpy()
    x = a[:, 0].astype(np.uint32)
    y = a[:, 1].astype(np.uint32)
    z = a[:, 2].astype(np.uint32)
    index = (x ^ (y * 2654435761) ^ (z * 805459861)) % T
    print(index)
    print(b[index])


    
