#include <cfloat>
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kBlockSize = 256;

__global__ void softmax_f32_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t batch_size,
    int64_t hidden_size) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= batch_size) {
    return;
  }

  const float* x_row = x + static_cast<int64_t>(row) * hidden_size;
  float* y_row = y + static_cast<int64_t>(row) * hidden_size;

  float local_max = -FLT_MAX;
  for (int64_t col = tid; col < hidden_size; col += blockDim.x) {
    local_max = fmaxf(local_max, x_row[col]);
  }

  __shared__ float shared[kBlockSize];
  shared[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }

  const float row_max = shared[0];

  float local_sum = 0.0f;
  for (int64_t col = tid; col < hidden_size; col += blockDim.x) {
    const float exp_val = expf(x_row[col] - row_max);
    y_row[col] = exp_val;
    local_sum += exp_val;
  }

  shared[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float row_sum = shared[0];
  const float inv_sum = 1.0f / row_sum;

  for (int64_t col = tid; col < hidden_size; col += blockDim.x) {
    y_row[col] *= inv_sum;
  }
}

} // namespace

torch::Tensor softmax_cuda(torch::Tensor x) {
  const auto batch_size = x.size(0);
  const auto hidden_size = x.size(1);

  auto y = torch::empty_like(x);

  const dim3 grid(batch_size);
  const dim3 block(kBlockSize);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  softmax_f32_kernel<<<grid, block, 0, stream>>>(
      x.data_ptr<float>(),
      y.data_ptr<float>(),
      batch_size,
      hidden_size);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}

int softmax_cuda_block_size() {
  return kBlockSize;
}
