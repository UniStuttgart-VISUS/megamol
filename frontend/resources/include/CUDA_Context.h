#pragma once

namespace megamol::frontend_resources {

static std::string CUDA_Context_Req_Name = "CUDA_Context";

struct CUDA_Context {
    void* ctx_;
    int device_;
};

} // namespace megamol::frontend_resources
