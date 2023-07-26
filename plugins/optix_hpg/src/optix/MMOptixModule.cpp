#include "MMOptixModule.h"

#include <optional>
#include <sstream>
#include <tuple>

#include <glm/glm.hpp>

#include "mmcore/utility/log/Log.h"
#include "optix/Utils.h"

// bounds kernel embedded in optix kernel inspired by OWL

namespace megamol::optix_hpg {

template<unsigned int N>
class simple_log {
public:
    char const* read() {
        log_size_ = N;
        return log_;
    }

    size_t get_log_size() {
        auto tmp_size = log_size_;
        log_size_ = N;
        return tmp_size;
    }

    operator char*() {
        return log_;
    }

    operator size_t*() {
        return &log_size_;
    }

private:
    char log_[N];

    size_t log_size_ = N;
};


std::string hide_optix_commands(std::string const& ptx_code) {
    std::istringstream ptx_in = std::istringstream(ptx_code);
    std::ostringstream ptx_out;
    for (std::string line; std::getline(ptx_in, line);) {
        if (line.find("_optix_") != std::string::npos) {
            ptx_out << "// skipped: " << line << '\n';
        } else {
            ptx_out << line << '\n';
        }
    }
    return ptx_out.str();
}


std::tuple<CUmodule, CUfunction> get_bounds_function(std::string const& ptx_code, std::string const& name) {
    auto ptx_out = hide_optix_commands(std::string(ptx_code));

    unsigned int log_size = 2048;
    char log[2048];

    CUmodule bounds_module = nullptr;
    CUfunction bounds_function = nullptr;

    CUjit_option options[] = {
        CU_JIT_TARGET_FROM_CUCONTEXT,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    };
    void* optionValues[] = {(void*)0, (char*)log, (unsigned int*)&log_size};
    CUDA_CHECK_ERROR(cuModuleLoadDataEx(&bounds_module, ptx_out.c_str(), 3, options, optionValues));
#if DEBUG
    if (log_size > 1) {
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[MMOptixModule] Bounds Module creation info: %s", log);
    }
#endif
    std::string bounds_name = MM_OPTIX_BOUNDS_ANNOTATION_STRING + name;
    CUDA_CHECK_ERROR(cuModuleGetFunction(&bounds_function, bounds_module, bounds_name.c_str()));

    return std::make_tuple(bounds_module, bounds_function);
}


struct MMOptixProgramGrousDesc {
    OptixProgramGroupDesc desc_;

    std::vector<std::string> names_;
};


MMOptixProgramGrousDesc fill_optix_programgroupdesc(MMOptixModule::MMOptixProgramGroupKind kind, OptixModule mod,
    std::vector<std::pair<MMOptixModule::MMOptixNameKind, std::string>> const& names,
    std::optional<OptixModule> const& built_in_intersector = std::nullopt) {
    MMOptixProgramGrousDesc desc{};

    desc.desc_.kind = static_cast<OptixProgramGroupKind>(kind);

    switch (kind) {
    case MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_RAYGEN: {
        desc.desc_.raygen.module = mod;
        auto& [name_kind, name] = names[0];
        desc.names_.push_back(std::string(MM_OPTIX_RAYGEN_ANNOTATION_STRING) + name);
        desc.desc_.raygen.entryFunctionName = desc.names_.back().c_str();
    } break;
    case MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_MISS: {
        desc.desc_.miss.module = mod;
        auto& [name_kind, name] = names[0];
        desc.names_.push_back(std::string(MM_OPTIX_MISS_ANNOTATION_STRING) + name);
        desc.desc_.miss.entryFunctionName = desc.names_.back().c_str();
    } break;
    case MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_EXCEPTION: {
        desc.desc_.exception.module = mod;
        auto& [name_kind, name] = names[0];
        desc.names_.push_back(std::string(MM_OPTIX_EXCEPTION_ANNOTATION_STRING) + name);
        desc.desc_.exception.entryFunctionName = desc.names_.back().c_str();
    } break;
    case MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_CALLABLES: {
        desc.desc_.callables.moduleDC = mod;
        auto& [name_kind, name] = names[0];
        desc.names_.push_back(std::string(MM_OPTIX_DIRECT_CALLABLE_ANNOTATION_STRING) + name);
        desc.desc_.callables.entryFunctionNameDC = desc.names_.back().c_str();
    } break;
    case MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_HITGROUP: {
        for (auto i = 0; i < names.size(); ++i) {
            auto& [name_kind, name] = names[i];
            if (name_kind == MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_INTERSECTION) {
                desc.desc_.hitgroup.moduleIS = mod;
                desc.names_.push_back(std::string(MM_OPTIX_INTERSECTION_ANNOTATION_STRING) + name);
                desc.desc_.hitgroup.entryFunctionNameIS = desc.names_.back().c_str();
            }
            if (name_kind == MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_CLOSESTHIT) {
                desc.desc_.hitgroup.moduleCH = mod;
                desc.names_.push_back(std::string(MM_OPTIX_CLOSESTHIT_ANNOTATION_STRING) + name);
                desc.desc_.hitgroup.entryFunctionNameCH = desc.names_.back().c_str();
            }
            if (name_kind == MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_ANYHIT) {
                desc.desc_.hitgroup.moduleAH = mod;
                desc.names_.push_back(std::string(MM_OPTIX_ANYHIT_ANNOTATION_STRING) + name);
                desc.desc_.hitgroup.entryFunctionNameAH = desc.names_.back().c_str();
            }
        }
        if (built_in_intersector) {
            desc.desc_.hitgroup.moduleIS = built_in_intersector.value();
            desc.desc_.hitgroup.entryFunctionNameIS = nullptr;
        }
    } break;
    }

    return desc;
}

} // namespace megamol::optix_hpg


megamol::optix_hpg::MMOptixModule::MMOptixModule() {}


megamol::optix_hpg::MMOptixModule::MMOptixModule(const char* ptx_code, OptixDeviceContext ctx,
    OptixModuleCompileOptions const* module_options, OptixPipelineCompileOptions const* pipeline_options,
    MMOptixProgramGroupKind kind, std::vector<std::pair<MMOptixNameKind, std::string>> const& names) {
    simple_log<2048> log;

    OPTIX_CHECK_ERROR(optixModuleCreateFromPTX(
        ctx, module_options, pipeline_options, ptx_code, std::strlen(ptx_code), log, log, &module_));
#if DEBUG
    if (log.get_log_size() > 1) {
        core::utility::log::Log::DefaultLog.WriteInfo("[MMOptixModule] Optix Module creation info: %s", log.read());
    }
#endif

    if (kind == MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_HITGROUP) {
        auto const fit = std::find_if(names.begin(), names.end(),
            [](auto const& el) { return el.first == MMOptixNameKind::MMOPTIX_NAME_BOUNDS; });
        std::string bounds_name = fit != names.end() ? fit->second : "bounds";
        std::tie(bounds_module_, bounds_function_) = get_bounds_function(ptx_code, bounds_name);
    }

    auto const desc = fill_optix_programgroupdesc(kind, module_, names);

    OptixProgramGroupOptions pgOptions = {};

    OPTIX_CHECK_ERROR(optixProgramGroupCreate(ctx, &desc.desc_, 1, &pgOptions, log, log, &program_));
#if DEBUG
    if (log.get_log_size() > 1) {
        core::utility::log::Log::DefaultLog.WriteError("[MMOptixModule] Program group creation info: %s", log.read());
    }
#endif
}

megamol::optix_hpg::MMOptixModule::MMOptixModule(const char* ptx_code, OptixDeviceContext ctx,
    OptixModuleCompileOptions const* module_options, OptixPipelineCompileOptions const* pipeline_options,
    MMOptixProgramGroupKind kind, OptixModule built_in_intersector,
    std::vector<std::pair<MMOptixNameKind, std::string>> const& names) {
    simple_log<2048> log;

    OPTIX_CHECK_ERROR(optixModuleCreateFromPTX(
        ctx, module_options, pipeline_options, ptx_code, std::strlen(ptx_code), log, log, &module_));
#if DEBUG
    if (log.get_log_size() > 1) {
        core::utility::log::Log::DefaultLog.WriteInfo("[MMOptixModule] Optix Module creation info: %s", log.read());
    }
#endif

    auto const desc = fill_optix_programgroupdesc(kind, module_, names, std::make_optional(built_in_intersector));

    OptixProgramGroupOptions pgOptions = {};

    OPTIX_CHECK_ERROR(optixProgramGroupCreate(ctx, &desc.desc_, 1, &pgOptions, log, log, &program_));
#if DEBUG
    if (log.get_log_size() > 1) {
        core::utility::log::Log::DefaultLog.WriteError("[MMOptixModule] Program group creation info: %s", log.read());
    }
#endif
}


megamol::optix_hpg::MMOptixModule::MMOptixModule(MMOptixModule&& rhs) noexcept
        : module_(rhs.module_)
        , program_(rhs.program_)
        , bounds_module_(rhs.bounds_module_)
        , bounds_function_(rhs.bounds_function_) {}


megamol::optix_hpg::MMOptixModule& megamol::optix_hpg::MMOptixModule::operator=(MMOptixModule&& rhs) noexcept {
    if (this != std::addressof(rhs)) {
        std::swap(module_, rhs.module_);
        std::swap(program_, rhs.program_);
        std::swap(bounds_module_, rhs.bounds_module_);
        std::swap(bounds_function_, rhs.bounds_function_);
    }
    return *this;
}


megamol::optix_hpg::MMOptixModule::~MMOptixModule() {
    if (program_ != nullptr) {
        OPTIX_CHECK_ERROR(optixProgramGroupDestroy(program_));
    }
    if (module_ != nullptr) {
        OPTIX_CHECK_ERROR(optixModuleDestroy(module_));
    }
    if (bounds_module_ != nullptr) {
        CUDA_CHECK_ERROR(cuModuleUnload(bounds_module_));
    }
}


void megamol::optix_hpg::MMOptixModule::ComputeBounds(
    CUdeviceptr data_in, CUdeviceptr bounds_out, uint32_t num_elements, CUstream stream) const {
    glm::vec3 blockDims(32, 32, 1);
    uint32_t threadsPerBlock = blockDims.x * blockDims.y * blockDims.z;

    uint32_t numBlocks = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    uint32_t numBlocks_x = 1 + uint32_t(powf((float)numBlocks, 1.f / 3.f));
    uint32_t numBlocks_y = 1 + uint32_t(sqrtf((float)(numBlocks / numBlocks_x)));
    uint32_t numBlocks_z = (numBlocks + numBlocks_x * numBlocks_y - 1) / numBlocks_x * numBlocks_y;

    glm::uvec3 gridDims(numBlocks_x, numBlocks_y, numBlocks_z);

    void* args[] = {&data_in, &bounds_out, (void*)&num_elements};

    CUDA_CHECK_ERROR(cuLaunchKernel(bounds_function_, gridDims.x, gridDims.y, gridDims.z, blockDims.x, blockDims.y,
        blockDims.z, 0, stream, args, nullptr));
}
