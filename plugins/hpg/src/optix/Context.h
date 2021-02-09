#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"

#include "hpg/optix/Utils.h"

#include "cuda.h"
#include "optix.h"

namespace megamol::hpg::optix {

class Context : public core::Module {
public:
    static const char* ClassName(void) {
        return "OptiXContext";
    }

    static const char* Description(void) {
        return "Context for OptiX";
    }

    static bool IsAvailable(void) {
        return true;
    }

    Context();

    virtual ~Context();

protected:
    bool create() override;

    void release() override;

private:
    bool get_ctx_cb(core::Call& c);

    core::CalleeSlot _out_ctx_slot;

    CUdevice _device;

    CUcontext _ctx = nullptr;

    OptixDeviceContext _optix_ctx = nullptr;

    CUstream _data_stream = nullptr;

    CUstream _exec_stream = nullptr;

    OptixModuleCompileOptions _module_options;

    OptixPipelineCompileOptions _pipeline_options;

    OptixPipelineLinkOptions _pipeline_link_options;
};

} // namespace megamol::hpg::optix
