#include <cstdint>
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
