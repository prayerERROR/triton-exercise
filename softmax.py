import torch
import triton
import triton.language as tl

DEVICE = "cuda"
# We have set adaptive BLOCK_SIZE in function softmax.
# BLOCK_SIZE = 256

# Softmax operator
@triton.jit
def softmax_kernel(a_ptr, out_ptr, a_stride, out_stride, n, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start_ptr = a_ptr + pid * a_stride
    out_start_ptr = out_ptr + pid * out_stride
    offset = tl.arange(0, BLOCK_SIZE_N)
    mask = offset < n
    row = tl.load(row_start_ptr + offset, mask=mask, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0) 
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    out = numerator / denominator
    tl.store(out_start_ptr + offset, out, mask=mask)

def softmax(a):
    assert a.is_contiguous()
    m, n = a.shape
    out = torch.empty_like(a)
    BLOCK_SIZE = triton.next_power_of_2(n)
    softmax_kernel[(m,)](a, out, a.stride(0), out.stride(0), n, BLOCK_SIZE_N=BLOCK_SIZE)
    return out

def benchmark_softmax():
    sizes = [2**n for n in range(5, 14, 1)]
    quantiles = [0.5, 0.2, 0.8]
    torch_perf, triton_perf = [], []
    for size in sizes:
        torch.cuda.empty_cache()
        x = torch.rand((size, size), device=DEVICE)
        GBps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, ms_max, ms_min = triton.testing.do_bench(lambda: torch.nn.functional.softmax(x, dim=1), quantiles=quantiles)
        torch_perf.append((GBps(ms), GBps(ms_max), GBps(ms_min)))
        ms, ms_max, ms_min = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
        triton_perf.append((GBps(ms), GBps(ms_max), GBps(ms_min)))
    return torch_perf, triton_perf