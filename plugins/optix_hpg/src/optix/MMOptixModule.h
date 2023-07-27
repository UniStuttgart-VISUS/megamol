#pragma once

#include <string>
#include <vector>

#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>

#include "optix/utils_host.h"

namespace megamol::optix_hpg {
class MMOptixModule {
public:
    enum class MMOptixProgramGroupKind : std::underlying_type_t<OptixProgramGroupKind> {
        MMOPTIX_PROGRAM_GROUP_KIND_RAYGEN = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        MMOPTIX_PROGRAM_GROUP_KIND_MISS = OPTIX_PROGRAM_GROUP_KIND_MISS,
        MMOPTIX_PROGRAM_GROUP_KIND_EXCEPTION = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION,
        MMOPTIX_PROGRAM_GROUP_KIND_HITGROUP = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        MMOPTIX_PROGRAM_GROUP_KIND_CALLABLES = OPTIX_PROGRAM_GROUP_KIND_CALLABLES,
        MMOPTIX_PROGRAM_GROUP_BOUNDS = 0
    };

    enum class MMOptixNameKind : uint8_t {
        MMOPTIX_NAME_GENERIC,
        MMOPTIX_NAME_INTERSECTION,
        MMOPTIX_NAME_CLOSESTHIT,
        MMOPTIX_NAME_ANYHIT,
        MMOPTIX_NAME_BOUNDS,
        MMOPTIX_NAME_DIRECT,
        MMOPTIX_NAME_CONTINOUS
    };

    MMOptixModule();

    MMOptixModule(const char* ptx_code, OptixDeviceContext ctx, OptixModuleCompileOptions const* module_options,
        OptixPipelineCompileOptions const* pipeline_options, MMOptixProgramGroupKind kind,
        std::vector<std::pair<MMOptixNameKind, std::string>> const& names);

    MMOptixModule(const char* ptx_code, OptixDeviceContext ctx, OptixModuleCompileOptions const* module_options,
        OptixPipelineCompileOptions const* pipeline_options, MMOptixProgramGroupKind kind,
        OptixModule built_in_intersector, std::vector<std::pair<MMOptixNameKind, std::string>> const& names);

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
