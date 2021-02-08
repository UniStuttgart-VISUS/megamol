#pragma once

namespace megamol::frontend_resources {

struct CUDA_Context {
    void* ctx_;
    int device_;
};

} // namespace megamol::frontend_resources
