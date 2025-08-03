import torch
import triton
import triton.language as tl

DEVICE = "cuda"
BLOCK_SIZE = 256

# Dropout kernel
@triton.jit
def dropout_kernel(a_ptr, out_ptr, n, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n
    random = tl.rand(seed, offset)
    a_mask = random > p
    a = tl.load(a_ptr + offset, mask=mask)
    scale = 1 / (1 - p)
    out = tl.where(a_mask, a*scale, 0.0)
    tl.store(out_ptr + offset, out, mask=mask)

def dropout(a, p, seed):
    assert 0 <= p <= 1
    out = torch.empty_like(a)
    n = out.numel()
    grid = triton.cdiv(n, BLOCK_SIZE)
    dropout_kernel[(grid,)](a, out, n, p, seed, BLOCK_SIZE=BLOCK_SIZE)
    return out

def benchmark_dropout():
    sizes = [2**n for n in range(5, 14, 1)]
    quantiles = [0.5, 0.2, 0.8]
    torch_perf, triton_perf = [], []
    for size in sizes:
        torch.cuda.empty_cache()
        x = torch.rand((size, size), device=DEVICE)
        GBps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, ms_max, ms_min = triton.testing.do_bench(lambda: torch.nn.functional.dropout(x, 0.1), quantiles=quantiles)
        torch_perf.append((GBps(ms), GBps(ms_max), GBps(ms_min)))
        ms, ms_max, ms_min = triton.testing.do_bench(lambda: dropout(x, 0.1, 42), quantiles=quantiles)
        triton_perf.append((GBps(ms), GBps(ms_max), GBps(ms_min)))
    return torch_perf, triton_perf