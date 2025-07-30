# Pytorch
import torch

# Triton
import triton
import triton.language as tl

# Config device
DEVICE = "cuda"

# Add kernel
@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a = tl.load(a_ptr + offset, mask=mask)
    b = tl.load(b_ptr + offset, mask=mask)
    print(type(a))
    res = a + b
    tl.store(out_ptr + offset, res, mask=mask)

def add(a, b):
    assert a.device == b.device
    assert a.shape == b.shape
    out = torch.empty_like(a)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=256)
    return out

def benchmark_add():
    sizes = [2**n for n in range(10, 25, 1)]
    quantiles = [0.5, 0.2, 0.8]
    torch_perf, triton_perf = [], []
    for size in sizes:
        torch.cuda.empty_cache()
        x = torch.rand(size, device=DEVICE)
        y = torch.rand(size, device=DEVICE)
        GBps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, ms_max, ms_min = triton.testing.do_bench(lambda: x+y, quantiles=quantiles)
        torch_perf.append((GBps(ms), GBps(ms_max), GBps(ms_min)))
        ms, ms_max, ms_min = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
        triton_perf.append((GBps(ms), GBps(ms_max), GBps(ms_min)))
    return torch_perf, triton_perf


# Mul kernel
@triton.jit
def mul_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a = tl.load(a_ptr + offset, mask=mask)
    b = tl.load(b_ptr + offset, mask=mask)
    res = a * b
    tl.store(out_ptr + offset, res, mask=mask)

def mul(a, b):
    assert a.device == b.device
    assert a.shape == b.shape
    out = torch.empty_like(a)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    mul_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=256)
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


# Exp kernel
@triton.jit
def exp_kernel(a_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a = tl.load(a_ptr + offset, mask=mask)
    res = tl.exp(a)
    tl.store(out_ptr + offset, res, mask=mask)

def exp(a):
    out = torch.empty_like(a)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    exp_kernel[grid](a, out, n_elements, BLOCK_SIZE=256)
    return out

def benchmark_exp():
    sizes = [2**n for n in range(10, 25, 1)]
    quantiles = [0.5, 0.2, 0.8]
    torch_perf, triton_perf = [], []
    for size in sizes:
        torch.cuda.empty_cache()
        x = torch.rand(size, device=DEVICE)
        GBps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, ms_max, ms_min = triton.testing.do_bench(lambda: torch.exp(x), quantiles=quantiles)
        torch_perf.append((GBps(ms), GBps(ms_max), GBps(ms_min)))
        ms, ms_max, ms_min = triton.testing.do_bench(lambda: exp(x), quantiles=quantiles)
        triton_perf.append((GBps(ms), GBps(ms_max), GBps(ms_min)))
    return torch_perf, triton_perf