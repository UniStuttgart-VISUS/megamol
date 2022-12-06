#pragma once

#include "mmstd/data/AbstractGetDataCall.h"

#include "cuda.h"
#include "optix.h"

namespace megamol::optix_hpg {
class CallContext : public core::AbstractGetDataCall {
public:
    static const char* ClassName(void) {
        return "CallContext";
    }

    static const char* Description(void) {
        return "Transports an OptiX context";
    }

    static unsigned int FunctionCount(void) {
        return core::AbstractGetDataCall::FunctionCount();
    }

    static const char* FunctionName(unsigned int idx) {
        return core::AbstractGetDataCall::FunctionName(idx);
    }

    CUcontext get_cuda_ctx() const {
        return _cuda_ctx;
    }

    void set_cuda_ctx(CUcontext ctx) {
        _cuda_ctx = ctx;
    }

    OptixDeviceContext get_ctx() const {
        return _ctx;
    }

    void set_ctx(OptixDeviceContext ctx) {
        _ctx = ctx;
    }

    CUstream get_exec_stream() const {
        return _exec_stream;
    }

    void set_exec_stream(CUstream stream) {
        _exec_stream = stream;
    }

    OptixModuleCompileOptions const* get_module_options() const {
        return _module_options;
    }

    void set_module_options(OptixModuleCompileOptions const* options) {
        _module_options = options;
    }

    OptixPipelineCompileOptions const* get_pipeline_options() const {
        return _pipeline_options;
    }

    void set_pipeline_options(OptixPipelineCompileOptions const* options) {
        _pipeline_options = options;
    }

    OptixPipelineLinkOptions const* get_pipeline_link_options() const {
        return _pipeline_link_options;
    }

    void set_pipeline_link_options(OptixPipelineLinkOptions const* options) {
        _pipeline_link_options = options;
    }

private:
    CUcontext _cuda_ctx;

    OptixDeviceContext _ctx;

    CUstream _exec_stream;

    OptixModuleCompileOptions const* _module_options;

    OptixPipelineCompileOptions const* _pipeline_options;

    OptixPipelineLinkOptions const* _pipeline_link_options;
};

using CallContextDescription = megamol::core::factories::CallAutoDescription<CallContext>;

} // namespace megamol::optix_hpg
