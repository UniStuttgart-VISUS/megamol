#include "Context.h"

#include "optix/CallContext.h"

#include "optix_stubs.h"


megamol::optix_hpg::Context::Context() {}


megamol::optix_hpg::Context::Context(frontend_resources::CUDA_Context const& ctx) {
    /////////////////////////////////////
    // cuda
    /////////////////////////////////////
    _ctx = reinterpret_cast<CUcontext>(ctx.ctx_);
    _device = ctx.device_;
    CUDA_CHECK_ERROR(cuStreamCreate(&_data_stream, CU_STREAM_NON_BLOCKING));
    CUDA_CHECK_ERROR(cuStreamCreate(&_exec_stream, CU_STREAM_NON_BLOCKING));
    /////////////////////////////////////
    // end cuda
    /////////////////////////////////////

    /////////////////////////////////////
    // optix
    /////////////////////////////////////
    OPTIX_CHECK_ERROR(optixInit());
    OPTIX_CHECK_ERROR(optixDeviceContextCreate(_ctx, 0, &_optix_ctx));
#ifdef DEBUG
    OPTIX_CHECK_ERROR(optixDeviceContextSetLogCallback(_optix_ctx, &optix_log_callback, nullptr, 3));
#else
    OPTIX_CHECK_ERROR(optixDeviceContextSetLogCallback(_optix_ctx, &optix_log_callback, nullptr, 2));
#endif

    _module_options = {};
    _module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef DEBUG
    _module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    _module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
    _module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    _module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif // DEBUG

    _pipeline_options = {};
    _pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    _pipeline_options.numAttributeValues = 2;
    _pipeline_options.numPayloadValues = 2;
    _pipeline_options.pipelineLaunchParamsVariableName = "optixLaunchParams";
    _pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    _pipeline_options.usesMotionBlur = false;
    _pipeline_options.usesPrimitiveTypeFlags = 0;

    _pipeline_link_options = {};
#ifdef DEBUG
    _pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    _pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
    _pipeline_link_options.maxTraceDepth = 2;
    /////////////////////////////////////
    // end optix
    /////////////////////////////////////
}


megamol::optix_hpg::Context::~Context() {
    if (_optix_ctx != nullptr) {
        OPTIX_CHECK_ERROR(optixDeviceContextDestroy(_optix_ctx));
    }
    if (_data_stream != nullptr) {
        CUDA_CHECK_ERROR(cuStreamDestroy(_data_stream));
    }
    if (_exec_stream != nullptr) {
        CUDA_CHECK_ERROR(cuStreamDestroy(_exec_stream));
    }
}
