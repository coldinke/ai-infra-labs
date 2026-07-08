#include <climits>
#include <cstdint>
#include <sys/cdefs.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kBlockSize = 256;

__global__ void rmsnorm_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int64_t batch_size,
    int64_t hidden_size,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= batch_size) {
        return;
    }

    const float* x_row = x + static_cast<int64_t>(row) * hidden_size;
    float* y_row = y + static_cast<int64_t>(row) * hidden_size;

    float local_sum = 0.0f;

    for (int64_t col = tid; col < hidden_size; col += blockDim.x) {
        float v = x_row[col];
        local_sum += v * v;
    }

    __shared__ float shared[kBlockSize];
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float mean_square = shared[0] / static_cast<float>(hidden_size);
    float rstd = rsqrtf(mean_square + eps);

    for (int64_t col = tid; col < hidden_size; col += blockDim.x) {
        y_row[col] = x_row[col] * rstd * weight[col];
    }
}

__global__ void rmsnorm_f32x4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int64_t batch_size,
    int64_t hidden_size,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= batch_size) {
        return;
    }

    const int64_t hidden_vec_size = hidden_size / 4;

    const float4* __restrict__ x4 = 
        reinterpret_cast<const float4*>(x + static_cast<int64_t>(row) * hidden_size);
    const float4* __restrict__ w4 =
        reinterpret_cast<const float4*>(weight);
    float4* __restrict__ y4 =
        reinterpret_cast<float4*>(y + static_cast<int64_t>(row) * hidden_size);

    float local_sum = 0.0f;

    for (int64_t col4 = tid; col4 < hidden_vec_size; col4 += blockDim.x) {
        float4 xv = x4[col4];

        local_sum += xv.x * xv.x;
        local_sum += xv.y * xv.y;
        local_sum += xv.z * xv.z;
        local_sum += xv.w * xv.w;
    }

    __shared__ float shared[kBlockSize];
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float mean_square = shared[0] / static_cast<float>(hidden_size);
    float rstd = rsqrtf(mean_square + eps);

    for (int64_t col4 = tid; col4 < hidden_vec_size; col4 += blockDim.x) {
        float4 xv = x4[col4];
        float4 wv = w4[col4];

        float4 out;
        out.x = xv.x * rstd * wv.x;
        out.y = xv.y * rstd * wv.y;
        out.z = xv.z * rstd * wv.z;
        out.w = xv.w * rstd * wv.w;

        y4[col4] = out;
    }
}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void rmsnorm_f32_warp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int64_t batch_size,
    int64_t hidden_size,
    float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= batch_size) {
    return;
  }

  const float* x_row = x + static_cast<int64_t>(row) * hidden_size;
  float* y_row = y + static_cast<int64_t>(row) * hidden_size;

  float local_sum = 0.0f;

  for (int64_t col = tid; col < hidden_size; col += blockDim.x) {
    float v = x_row[col];
    local_sum += v * v;
  }

  const int lane = tid % warpSize;
  const int warp_id = tid / warpSize;
  constexpr int kNumWarps = kBlockSize / 32;

  __shared__ float warp_sums[kNumWarps];

  float warp_sum = warp_reduce_sum(local_sum);

  if (lane == 0) {
    warp_sums[warp_id] = warp_sum;
  }

  __syncthreads();

  float block_sum = 0.0f;

  if (warp_id == 0) {
    block_sum = (lane < kNumWarps) ? warp_sums[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);

    if (lane == 0) {
      warp_sums[0] = block_sum;
    }
  }

  __syncthreads();

  float mean_square = warp_sums[0] / static_cast<float>(hidden_size);
  float rstd = rsqrtf(mean_square + eps);

  for (int64_t col = tid; col < hidden_size; col += blockDim.x) {
    y_row[col] = x_row[col] * rstd * weight[col];
  }
}

} // namespace

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
    const auto batch_size = x.size(0);
    const auto hidden_size = x.size(1);

    auto y = torch::empty_like(x);

    const dim3 grid(batch_size);
    const dim3 block(kBlockSize);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    rmsnorm_f32_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size,
        hidden_size,
        static_cast<float>(eps)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}

torch::Tensor rmsnorm_cuda_vectorized(torch::Tensor x, torch::Tensor weight, double eps) {
  const auto batch_size = x.size(0);
  const auto hidden_size = x.size(1);

  TORCH_CHECK(hidden_size % 4 == 0, "hidden_size must be divisible by 4");

  auto y = torch::empty_like(x);

  const dim3 grid(batch_size);
  const dim3 block(kBlockSize);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  rmsnorm_f32x4_kernel<<<grid, block, 0, stream>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      y.data_ptr<float>(),
      batch_size,
      hidden_size,
      static_cast<float>(eps));

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}

torch::Tensor rmsnorm_cuda_warp(torch::Tensor x, torch::Tensor weight, double eps) {
  const auto batch_size = x.size(0);
  const auto hidden_size = x.size(1);

  auto y = torch::empty_like(x);

  const dim3 grid(batch_size);
  const dim3 block(kBlockSize);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  rmsnorm_f32_warp_kernel<<<grid, block, 0, stream>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      y.data_ptr<float>(),
      batch_size,
      hidden_size,
      static_cast<float>(eps));

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}

int rmsnorm_cuda_block_size() {
    return kBlockSize;
}
