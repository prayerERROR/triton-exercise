# Pytorch
import torch

# Triton
import triton
import triton.language as tl

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