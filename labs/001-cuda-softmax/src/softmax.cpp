#include <torch/extension.h>

torch::Tensor softmax_cuda(torch::Tensor x);
int softmax_cuda_block_size();

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x)                                                       \
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x);                                                         \
  CHECK_FLOAT32(x)

torch::Tensor softmax_forward(torch::Tensor x) {
  CHECK_INPUT(x);

  TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor [batch_size, hidden_size]");

  return softmax_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &softmax_forward,
        "Naive fused row-wise softmax forward (CUDA, fp32)");
  m.def("block_size", &softmax_cuda_block_size, "Softmax CUDA block size");
}
