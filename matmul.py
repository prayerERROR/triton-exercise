import torch
import triton
import triton.language as tl

DEVICE = "cuda"
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 64

# Matmul kernel
@triton.jit
def matmul_kernel(a_ptr, b_ptr, out_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_om, stride_on,
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_blocks_n
    pid_n = pid % num_blocks_n
    offset_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    result = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        offset_k = tl.arange(0, BLOCK_SIZE_K) + k * BLOCK_SIZE_K
        a_addr = a_ptr + (offset_am[:, None] * stride_am + offset_k[None, :] * stride_ak)
        b_addr = b_ptr + (offset_k[:, None] * stride_bk + offset_bn[None, :] * stride_bn)
        k_mask = offset_k < K
        a_mask = (offset_am[:, None] < M) & (k_mask[None, :])
        b_mask = (k_mask[:, None]) & (offset_bn[None, :] < N)
        a = tl.load(a_addr, mask=a_mask, other=0)
        b = tl.load(b_addr, mask=b_mask, other=0)
        result += tl.dot(a, b)

    offset_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_addr = out_ptr + (offset_om[:, None] * stride_om + offset_on[None, :] * stride_on)
    out_mask = (offset_om[:, None] < M) & (offset_on[None, :] < N)
    tl.store(out_addr, result, mask=out_mask)

def matmul(a, b):
    assert a.is_contiguous() and b.is_contiguous()
    assert a.device == b.device
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    N = b.shape[1]
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_om, stride_on = out.stride(0), out.stride(1)
    grid = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)
    matmul_kernel[(grid,)](
        a, b, out,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_om, stride_on,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return out

def benchmark_matmul():
    sizes = [2**n for n in range(5, 14, 1)]
    quantiles = [0.5, 0.2, 0.8]
    torch_perf, triton_perf = [], []
    for size in sizes:
        torch.cuda.empty_cache()
        x = torch.rand((size, size), device=DEVICE, dtype=torch.float16)
        y = torch.rand((size, size), device=DEVICE, dtype=torch.float16)
        # Coef = 3 is problematic here, because we load a and b multiple times.
        # However, we can still compare our kernel with torch kernel.
        # Torch is faster, since cuBLAS are optimized for different matrix size.
        GBps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, ms_max, ms_min = triton.testing.do_bench(lambda: x.matmul(y), quantiles=quantiles)
        torch_perf.append((GBps(ms), GBps(ms_max), GBps(ms_min)))
        ms, ms_max, ms_min = triton.testing.do_bench(lambda: matmul(x, y), quantiles=quantiles)
        triton_perf.append((GBps(ms), GBps(ms_max), GBps(ms_min)))
    return torch_perf, triton_perf