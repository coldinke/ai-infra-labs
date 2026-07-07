import time
import torch

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [batch_size, hidden_size]
    #  weight: [hidden_size]
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight

def benchmark(fn, *args, warmup: int = 20, iters: int = 100):
    for _ in range(warmup):
        fn(*args)
    
    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(iters):
        fn(*args)

    torch.cuda.synchronize()

    end = time.perf_counter()
    avg_ms = (end - start) * 1000 / iters
    return avg_ms

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    device = "cuda"
    dtype = torch.float32

    shapes = [
        (1, 1024),
        (1, 4096),
        (8, 4096),
        (32, 4096),
        (32, 8192),
    ]

    for batch_size, hidden_size in shapes:
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        weight = torch.randn(hidden_size, device=device, dtype=dtype)

        avg_ms = benchmark(rmsnorm_torch, x, weight)

        print(
            f"batch_size={batch_size:<4} "
            f"hidden_size={hidden_size:<6} "
            f"dtype={str(dtype):<15} "
            f"avg={avg_ms:.6f} ms"
        )

if __name__ == "__main__":
    main()