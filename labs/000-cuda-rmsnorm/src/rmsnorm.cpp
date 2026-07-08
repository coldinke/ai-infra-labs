#include <torch/extension.h>

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, double eps);
int rmsnorm_cuda_block_size();
torch::Tensor rmsnorm_cuda_vectorized(torch::Tensor x, torch::Tensor weight, double eps);
torch::Tensor rmsnorm_cuda_warp(torch::Tensor x, torch::Tensor weight, double eps);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x)                                                       \
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x);                                                         \
  CHECK_FLOAT32(x)

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);

  TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor [batch_size, hidden_size]");
  TORCH_CHECK(weight.dim() == 1, "weight must be a 1D tensor [hidden_size]");
  TORCH_CHECK(x.size(1) == weight.size(0),
              "x hidden_size must match weight size");

  return rmsnorm_cuda(x, weight, eps);
}

torch::Tensor rmsnorm_forward_vectorized(torch::Tensor x, torch::Tensor weight, double eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);

  TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor [batch_size, hidden_size]");
  TORCH_CHECK(weight.dim() == 1, "weight must be a 1D tensor [hidden_size]");
  TORCH_CHECK(x.size(1) == weight.size(0), "x hidden_size must match weight size");
  TORCH_CHECK(x.size(1) % 4 == 0, "hidden_size must be divisible by 4 for float4 vectorized kernel");

  return rmsnorm_cuda_vectorized(x, weight, eps);
}

torch::Tensor rmsnorm_forward_warp(torch::Tensor x, torch::Tensor weight, double eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);

  TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor [batch_size, hidden_size]");
  TORCH_CHECK(weight.dim() == 1, "weight must be a 1D tensor [hidden_size]");
  TORCH_CHECK(x.size(1) == weight.size(0), "x hidden_size must match weight size");

  return rmsnorm_cuda_warp(x, weight, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rmsnorm_forward, "Naive fused RMSNorm forward (CUDA, fp32)");
  m.def("forward_vectorized", &rmsnorm_forward_vectorized, "Float4 vectorized fused RMSNorm forward (CUDA, fp32)");
  m.def("forward_warp", &rmsnorm_forward_warp, "Warp-level reduction RMSNorm forward (CUDA, fp32)");
  m.def("block_size", &rmsnorm_cuda_block_size,
        "Thread block size used by the RMSNorm CUDA kernel");
}
