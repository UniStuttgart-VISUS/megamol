#pragma once

#include <string>
#include <vector>

#include "cuda.h"
#include "optix.h"
#include "optix_stubs.h"

#include "optix/utils_host.h"

namespace megamol::optix_hpg {
class MMOptixModule {
public:
    MMOptixModule();

    MMOptixModule(const char* ptx_code, OptixDeviceContext ctx, OptixModuleCompileOptions const* module_options,
        OptixPipelineCompileOptions const* pipeline_options, OptixProgramGroupKind kind,
        std::vector<std::string> const& names);

    MMOptixModule(const char* ptx_code, OptixDeviceContext ctx, OptixModuleCompileOptions const* module_options,
        OptixPipelineCompileOptions const* pipeline_options, OptixProgramGroupKind kind,
        OptixModule build_in_intersector, std::vector<std::string> const& names);

    MMOptixModule(MMOptixModule const& rhs) = delete;

    MMOptixModule& operator=(MMOptixModule const& rhs) = delete;

    MMOptixModule(MMOptixModule&& rhs) noexcept;

    MMOptixModule& operator=(MMOptixModule&& rhs) noexcept;

    ~MMOptixModule();

    operator OptixProgramGroup() {
        return program_;
    }

    operator OptixProgramGroup*() {
        return &program_;
    }

    void ComputeBounds(CUdeviceptr data_in, CUdeviceptr bounds_out, uint32_t num_elements, CUstream stream) const;

private:
    OptixModule module_ = nullptr;

    OptixProgramGroup program_ = nullptr;

    CUmodule bounds_module_ = nullptr;

    CUfunction bounds_function_ = nullptr;
};
} // namespace megamol::optix_hpg
