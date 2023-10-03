#include "library.h"

void set_grad_mode(uint8_t enabled) {
    torch::GradMode::set_enabled(enabled != 0u);
}

uint8_t is_autograd_enabled() {
    return torch::GradMode::is_enabled() ? 1u : 0u;
}

torch::Tensor *create_tensor(const int64_t *dims, const size_t ndims, const torch::ScalarType type, uint8_t gradient) {
    auto const options = torch::TensorOptions(type).requires_grad(gradient != 0u);
    return new torch::Tensor(torch::zeros(torch::IntArrayRef(dims, ndims), options));
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

torch::Tensor *tensor_to_dtype(torch::Tensor *tensor, torch::ScalarType type) {
    return new torch::Tensor(tensor->to(type));
}

torch::jit::Module *load_model(const char *path) {
    return new torch::jit::Module(torch::jit::load(path));
}

void delete_model(torch::jit::Module *module) {
    delete module;
}

torch::Tensor *forward1(torch::jit::Module *module, torch::Tensor *in1) {
    return new torch::Tensor(module->forward({*in1}).toTensor().contiguous());
}

torch::Tensor *forward2(torch::jit::Module *module, torch::Tensor *in1, torch::Tensor *in2) {
    return new torch::Tensor(module->forward({*in1, *in2}).toTensor().contiguous());
}

torch::Tensor *forward3(torch::jit::Module *module, torch::Tensor *in1, torch::Tensor *in2, torch::Tensor *in3) {
    return new torch::Tensor(module->forward({*in1, *in2, *in3}).toTensor().contiguous());
}
