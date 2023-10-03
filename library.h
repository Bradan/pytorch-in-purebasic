#ifndef PBTORCH_LIBRARY_H
#define PBTORCH_LIBRARY_H

#include <torch/torch.h>
#include <torch/script.h>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

void set_grad_mode(uint8_t enabled);
uint8_t is_autograd_enabled();

torch::Tensor *create_tensor(const int64_t *dims, size_t ndims, torch::ScalarType type, uint8_t gradient);
void delete_tensor(torch::Tensor *tensor);

torch::ScalarType get_tensor_type(torch::Tensor *tensor);
void *get_tensor_pointer(torch::Tensor *tensor);
torch::Tensor *tensor_to_dtype(torch::Tensor *tensor, torch::ScalarType type);

torch::jit::Module *load_model(const char *path);
void delete_model(torch::jit::Module *module);

torch::Tensor *forward1(torch::jit::Module *module, torch::Tensor *in1);
torch::Tensor *forward2(torch::jit::Module *module, torch::Tensor *in1, torch::Tensor *in2);
torch::Tensor *forward3(torch::jit::Module *module, torch::Tensor *in1, torch::Tensor *in2, torch::Tensor *in3);

#ifdef __cplusplus
}
#endif

#endif //PBTORCH_LIBRARY_H
