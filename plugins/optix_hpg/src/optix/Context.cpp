#include "Context.h"

#include "optix/CallContext.h"

#include "optix_stubs.h"


megamol::optix_hpg::Context::Context() : _out_ctx_slot("outCtx", "") {
    _out_ctx_slot.SetCallback(CallContext::ClassName(), CallContext::FunctionName(0), &Context::get_ctx_cb);
    MakeSlotAvailable(&_out_ctx_slot);
}


megamol::optix_hpg::Context::~Context() {
    this->Release();
}


bool megamol::optix_hpg::Context::create() {
    /////////////////////////////////////
    // cuda
    /////////////////////////////////////
    CUDA_CHECK_ERROR(cuInit(0));
    int deviceCount = 0;
    CUDA_CHECK_ERROR(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0) {
        return false;
    }
    CUDA_CHECK_ERROR(cuDeviceGet(&_device, 0));
    CUDA_CHECK_ERROR(cuCtxCreate(&_ctx, CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST, _device));
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

    return true;
}


void megamol::optix_hpg::Context::release() {
    OPTIX_CHECK_ERROR(optixDeviceContextDestroy(_optix_ctx));
    CUDA_CHECK_ERROR(cuStreamDestroy(_data_stream));
    CUDA_CHECK_ERROR(cuStreamDestroy(_exec_stream));
    CUDA_CHECK_ERROR(cuCtxDestroy(_ctx));
}


bool megamol::optix_hpg::Context::get_ctx_cb(core::Call& c) {
    auto outCall = dynamic_cast<CallContext*>(&c);
    if (outCall == nullptr)
        return false;

    outCall->set_cuda_ctx(_ctx);
    outCall->set_ctx(_optix_ctx);
    outCall->set_exec_stream(_exec_stream);
    outCall->set_module_options(&_module_options);
    outCall->set_pipeline_options(&_pipeline_options);
    outCall->set_pipeline_link_options(&_pipeline_link_options);

    return true;
}
