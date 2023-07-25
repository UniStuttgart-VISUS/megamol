/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "CUDA_Service.hpp"

#ifdef MM_CUDA_ENABLED

#include <stdexcept>

#include "cuda.h"

#include "mmcore/utility/log/Log.h"


static void log(std::string const& text) {
    const std::string msg = "CUDA_Service: " + text;
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}


static void log_error(std::string const& text) {
    const std::string msg = "CUDA_Service: " + text;
    megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
}


bool megamol::frontend::CUDA_Service::init(void* configPtr) {
    auto const cu_ret = cuInit(0);
    if (cu_ret != CUDA_SUCCESS) {
        log_error("Unable to initialize Cuda");
        return false;
    }
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) {
        log_error("No Cuda device available");
        return false;
    }
    cuDeviceGet(&ctx_.device_, 0);
    auto ctx_ptr = reinterpret_cast<CUcontext*>(&ctx_.ctx_);
    cuCtxCreate(ctx_ptr, CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST, ctx_.device_);

    if (ctx_.ctx_ == nullptr) {
        log_error("Unable to create a Cuda context");
        return false;
    }

    resourceReferences_ = {{frontend_resources::CUDA_Context_Req_Name, ctx_}};

    log("initialized successfully");

    return true;
}


void megamol::frontend::CUDA_Service::close() {
    if (ctx_.ctx_ != nullptr) {
        cuCtxDestroy(reinterpret_cast<CUcontext>(ctx_.ctx_));
    }
}

#endif
