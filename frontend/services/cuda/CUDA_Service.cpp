#include "CUDA_Service.hpp"

#ifdef MM_CUDA_ENABLED

#include <stdexcept>

#include "cuda.h"

#include "mmcore/utility/log/Log.h"


static void log(std::string const& text) {
    const std::string msg = "CUDA_Service: " + text;
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}


bool megamol::frontend::CUDA_Service::init(void* configPtr) {
    cuInit(0);
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[CUDA_Service]: No Cuda device available");
    }
    cuDeviceGet(&ctx_.device_, 0);
    auto ctx_ptr = reinterpret_cast<CUcontext*>(&ctx_);
    cuCtxCreate(ctx_ptr, CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST, ctx_.device_);

    if (ctx_.ctx_ == nullptr)
        return false;

    resourceReferences_ = {{"CUDA_Context", ctx_}};

    log("initialized successfully");

    return true;
}


void megamol::frontend::CUDA_Service::close() {
    if (ctx_.ctx_ != nullptr) {
        cuCtxDestroy(reinterpret_cast<CUcontext>(ctx_.ctx_));
    }
}

#endif
