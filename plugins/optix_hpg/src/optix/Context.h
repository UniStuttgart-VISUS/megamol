#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"

#include "optix/Utils.h"

#include "cuda.h"
#include "optix.h"

#include "CUDA_Context.h"

namespace megamol::optix_hpg {

class Context {
public:
    Context();

    Context(frontend_resources::CUDA_Context const& ctx);

    Context(Context const& rhs) = delete;

    Context& operator=(Context const& rhs) = delete;

    virtual ~Context();

    CUdevice GetDevice() const {
        return _device;
    }

    CUcontext GetCUDAContext() const {
        return _ctx;
    }

    OptixDeviceContext GetOptiXContext() const {
        return _optix_ctx;
    }

    CUstream GetDataStream() const {
        return _data_stream;
    }

    CUstream GetExecStream() const {
        return _exec_stream;
    }

    OptixModuleCompileOptions const& GetModuleCompileOptions() const {
        return _module_options;
    }

    OptixPipelineCompileOptions const& GetPipelineCompileOptions() const {
        return _pipeline_options;
    }

    OptixPipelineLinkOptions const& GetPipelineLinkOptions() const {
        return _pipeline_link_options;
    }

private:
    CUdevice _device;

    CUcontext _ctx = nullptr;

    OptixDeviceContext _optix_ctx = nullptr;

    CUstream _data_stream = nullptr;

    CUstream _exec_stream = nullptr;

    OptixModuleCompileOptions _module_options;

    OptixPipelineCompileOptions _pipeline_options;

    OptixPipelineLinkOptions _pipeline_link_options;
};

} // namespace megamol::optix_hpg
