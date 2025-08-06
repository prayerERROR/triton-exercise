import torch
import triton
import triton.language as tl

DEVICE = "cuda"
BLOCK_SIZE = 256

# Mul kernel
@triton.jit
def mul_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    a = tl.load(a_ptr + offset, mask=mask)
    b = tl.load(b_ptr + offset, mask=mask)
    res = a * b
    tl.store(out_ptr + offset, res, mask=mask)

def mul(a, b):
    assert a.device == b.device
    assert a.shape == b.shape
    out = torch.empty_like(a)
    N = out.numel()
    grid = triton.cdiv(N, BLOCK_SIZE)
    mul_kernel[(grid,)](a, b, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out

def benchmark_mul():
    sizes = [2**n for n in range(10, 25, 1)]
    quantiles = [0.5, 0.2, 0.8]
    torch_perf, triton_perf = [], []
    for size in sizes:
        torch.cuda.empty_cache()
        x = torch.rand(size, device=DEVICE)
        y = torch.rand(size, device=DEVICE)
        GBps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, ms_max, ms_min = triton.testing.do_bench(lambda: x*y, quantiles=quantiles)
        torch_perf.append((GBps(ms), GBps(ms_max), GBps(ms_min)))
        ms, ms_max, ms_min = triton.testing.do_bench(lambda: mul(x,y), quantiles=quantiles)
        triton_perf.append((GBps(ms), GBps(ms_max), GBps(ms_min)))
    return torch_perf, triton_perf
