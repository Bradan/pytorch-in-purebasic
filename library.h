#ifndef PBTORCH_LIBRARY_H
#define PBTORCH_LIBRARY_H

#include <torch/torch.h>
#include <torch/script.h>

#ifdef __cplusplus
extern "C" {
#endif

torch::Tensor *create_tensor(const long *dims, const size_t ndims, const torch::ScalarType type);
void delete_tensor(torch::Tensor *tensor);

torch::ScalarType get_tensor_type(torch::Tensor *tensor);
void *get_tensor_pointer(torch::Tensor *tensor);

torch::jit::Module *load_model(const char *path);
void delete_model(torch::jit::Module *module);

torch::Tensor *forward(torch::jit::Module *module, torch::Tensor *input);

#ifdef __cplusplus
}
#endif

#endif //PBTORCH_LIBRARY_H
