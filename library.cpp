#include "library.h"

torch::Tensor *create_tensor(const long *dims, const size_t ndims, const torch::ScalarType type) {
    return new torch::Tensor(torch::zeros(torch::IntArrayRef(dims, ndims), type));
}

void delete_tensor(torch::Tensor *tensor) {
    delete tensor;
}

torch::ScalarType get_tensor_type(torch::Tensor *tensor) {
    return tensor->scalar_type();
}

void *get_tensor_pointer(torch::Tensor *tensor) {
    return tensor->data_ptr();
}

torch::jit::Module *load_model(const char *path) {
    return new torch::jit::Module(torch::jit::load(path));
}

void delete_model(torch::jit::Module *module) {
    delete module;
}

torch::Tensor *forward(torch::jit::Module *module, torch::Tensor *input) {
    return new torch::Tensor(module->forward({*input}).toTensor().contiguous());
}
