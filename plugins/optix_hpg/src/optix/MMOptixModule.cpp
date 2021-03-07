#include "MMOptixModule.h"

#include <sstream>

#include "optix/Utils.h"

#include "mmcore/utility/log/Log.h"

#include "glm/glm.hpp"

// bounds kernel embedded in optix kernel inspired by OWL


megamol::optix_hpg::MMOptixModule::MMOptixModule() {}


megamol::optix_hpg::MMOptixModule::MMOptixModule(const char* ptx_code, OptixDeviceContext ctx,
    OptixModuleCompileOptions const* module_options, OptixPipelineCompileOptions const* pipeline_options,
    OptixProgramGroupKind kind, std::vector<std::string> const& names) {
    char log[2048];
    std::size_t log_size = 2048;

    OPTIX_CHECK_ERROR(optixModuleCreateFromPTX(
        ctx, module_options, pipeline_options, ptx_code, std::strlen(ptx_code), log, &log_size, &module_));
#if DEBUG
    if (log_size > 1) {
        core::utility::log::Log::DefaultLog.WriteInfo("[MMOptixModule] Optix Module creation info: %s", log);
    }
#endif

    std::istringstream ptx_in = std::istringstream(std::string(ptx_code));
    std::ostringstream ptx_out;
    for (std::string line; std::getline(ptx_in, line);) {
        if (line.find(" _optix_") != std::string::npos) {
            ptx_out << "// skipped: " << line << '\n';
        } else {
            ptx_out << line << '\n';
        }
    }

    log_size = 2048;

    if (kind == OPTIX_PROGRAM_GROUP_KIND_HITGROUP) {
        CUjit_option options[] = {
            CU_JIT_TARGET_FROM_CUCONTEXT,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        };
        void* optionValues[] = {(void*) 0, (void*) log, (void*) log_size};
        CUDA_CHECK_ERROR(cuModuleLoadDataEx(&bounds_module_, ptx_out.str().c_str(), 3, options, optionValues));
#if DEBUG
        if (log_size > 1) {
            core::utility::log::Log::DefaultLog.WriteInfo("[MMOptixModule] Bounds Module creation info: %s", log);
        }
#endif

        CUDA_CHECK_ERROR(
            cuModuleGetFunction(&bounds_function_, bounds_module_, MM_OPTIX_BOUNDS_ANNOTATION_STRING "sphere_bounds"));
    }

    std::vector<std::string> cb_names;

    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = kind;
    switch (kind) {
    case OPTIX_PROGRAM_GROUP_KIND_RAYGEN: {
        pgDesc.raygen.module = module_;
        cb_names.push_back(std::string(MM_OPTIX_RAYGEN_ANNOTATION_STRING) + names[0]);
        pgDesc.raygen.entryFunctionName = cb_names.back().c_str();
    } break;
    case OPTIX_PROGRAM_GROUP_KIND_MISS: {
        pgDesc.miss.module = module_;
        cb_names.push_back(std::string(MM_OPTIX_MISS_ANNOTATION_STRING) + names[0]);
        pgDesc.miss.entryFunctionName = cb_names.back().c_str();
    } break;
    case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION: {
        pgDesc.exception.module = module_;
        cb_names.push_back(std::string(MM_OPTIX_EXCEPTION_ANNOTATION_STRING) + names[0]);
        pgDesc.exception.entryFunctionName = cb_names.back().c_str();
    } break;
    case OPTIX_PROGRAM_GROUP_KIND_CALLABLES: {
        pgDesc.callables.moduleDC = module_;
        cb_names.push_back(std::string(MM_OPTIX_DIRECT_CALLABLE_ANNOTATION_STRING) + names[0]);
        pgDesc.callables.entryFunctionNameDC = cb_names.back().c_str();
    } break;
    case OPTIX_PROGRAM_GROUP_KIND_HITGROUP: {
        pgDesc.hitgroup.moduleIS = module_;
        cb_names.push_back(std::string(MM_OPTIX_INTERSECTION_ANNOTATION_STRING) + names[0]);
        pgDesc.hitgroup.entryFunctionNameIS = cb_names.back().c_str();
        pgDesc.hitgroup.moduleCH = module_;
        cb_names.push_back(std::string(MM_OPTIX_CLOSESTHIT_ANNOTATION_STRING) + names[1]);
        pgDesc.hitgroup.entryFunctionNameCH = cb_names.back().c_str();
        pgDesc.hitgroup.moduleAH = nullptr;
    } break;
    }

    OptixProgramGroupOptions pgOptions = {};

    log_size = 2048;

    OPTIX_CHECK_ERROR(optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOptions, log, &log_size, &program_));
#if DEBUG
    if (log_size > 1) {
        core::utility::log::Log::DefaultLog.WriteError("[MMOptixModule] Program group creation info: %s", log);
    }
#endif
}

megamol::optix_hpg::MMOptixModule::MMOptixModule(const char* ptx_code, OptixDeviceContext ctx,
    OptixModuleCompileOptions const* module_options, OptixPipelineCompileOptions const* pipeline_options,
    OptixProgramGroupKind kind, OptixModule build_in_intersector, std::vector<std::string> const& names) {
    char log[2048];
    std::size_t log_size = 2048;

    OPTIX_CHECK_ERROR(optixModuleCreateFromPTX(
        ctx, module_options, pipeline_options, ptx_code, std::strlen(ptx_code), log, &log_size, &module_));
#if DEBUG
    if (log_size > 1) {
        core::utility::log::Log::DefaultLog.WriteInfo("[MMOptixModule] Optix Module creation info: %s", log);
    }
#endif

    std::istringstream ptx_in = std::istringstream(std::string(ptx_code));
    std::ostringstream ptx_out;
    for (std::string line; std::getline(ptx_in, line);) {
        if (line.find(" _optix_") != std::string::npos) {
            ptx_out << "// skipped: " << line << '\n';
        } else {
            ptx_out << line << '\n';
        }
    }

    log_size = 2048;

    /*if (kind == OPTIX_PROGRAM_GROUP_KIND_HITGROUP) {
        CUjit_option options[] = {
            CU_JIT_TARGET_FROM_CUCONTEXT,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        };
        void* optionValues[] = {(void*) 0, (void*) log, (void*) log_size};
        CUDA_CHECK_ERROR(cuModuleLoadDataEx(&bounds_module_, ptx_out.str().c_str(), 3, options, optionValues));
#if DEBUG
        if (log_size > 1) {
            core::utility::log::Log::DefaultLog.WriteInfo("[MMOptixModule] Bounds Module creation info: %s", log);
        }
#endif

        CUDA_CHECK_ERROR(
            cuModuleGetFunction(&bounds_function_, bounds_module_, MM_OPTIX_BOUNDS_ANNOTATION_STRING "sphere_bounds"));
    }*/

    std::vector<std::string> cb_names;

    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = kind;
    switch (kind) {
    case OPTIX_PROGRAM_GROUP_KIND_RAYGEN: {
        pgDesc.raygen.module = module_;
        cb_names.push_back(std::string(MM_OPTIX_RAYGEN_ANNOTATION_STRING) + names[0]);
        pgDesc.raygen.entryFunctionName = cb_names.back().c_str();
    } break;
    case OPTIX_PROGRAM_GROUP_KIND_MISS: {
        pgDesc.miss.module = module_;
        cb_names.push_back(std::string(MM_OPTIX_MISS_ANNOTATION_STRING) + names[0]);
        pgDesc.miss.entryFunctionName = cb_names.back().c_str();
    } break;
    case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION: {
        pgDesc.exception.module = module_;
        cb_names.push_back(std::string(MM_OPTIX_EXCEPTION_ANNOTATION_STRING) + names[0]);
        pgDesc.exception.entryFunctionName = cb_names.back().c_str();
    } break;
    case OPTIX_PROGRAM_GROUP_KIND_CALLABLES: {
        pgDesc.callables.moduleDC = module_;
        cb_names.push_back(std::string(MM_OPTIX_DIRECT_CALLABLE_ANNOTATION_STRING) + names[0]);
        pgDesc.callables.entryFunctionNameDC = cb_names.back().c_str();
    } break;
    case OPTIX_PROGRAM_GROUP_KIND_HITGROUP: {
        if (build_in_intersector != nullptr) {
            pgDesc.hitgroup.moduleIS = build_in_intersector;
            cb_names.push_back(std::string(MM_OPTIX_INTERSECTION_ANNOTATION_STRING) + names[0]);
            pgDesc.hitgroup.entryFunctionNameIS = nullptr;
        } else {
            pgDesc.hitgroup.moduleIS = module_;
            cb_names.push_back(std::string(MM_OPTIX_INTERSECTION_ANNOTATION_STRING) + names[0]);
            pgDesc.hitgroup.entryFunctionNameIS = cb_names.back().c_str();
        }
        pgDesc.hitgroup.moduleCH = module_;
        cb_names.push_back(std::string(MM_OPTIX_CLOSESTHIT_ANNOTATION_STRING) + names[1]);
        pgDesc.hitgroup.entryFunctionNameCH = cb_names.back().c_str();
        pgDesc.hitgroup.moduleAH = nullptr;
    } break;
    }

    OptixProgramGroupOptions pgOptions = {};

    log_size = 2048;

    OPTIX_CHECK_ERROR(optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOptions, log, &log_size, &program_));
#if DEBUG
    if (log_size > 1) {
        core::utility::log::Log::DefaultLog.WriteError("[MMOptixModule] Program group creation info: %s", log);
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
    uint32_t numBlocks_x = 1 + uint32_t(powf((float) numBlocks, 1.f / 3.f));
    uint32_t numBlocks_y = 1 + uint32_t(sqrtf((float) (numBlocks / numBlocks_x)));
    uint32_t numBlocks_z = (numBlocks + numBlocks_x * numBlocks_y - 1) / numBlocks_x * numBlocks_y;

    glm::uvec3 gridDims(numBlocks_x, numBlocks_y, numBlocks_z);

    void* args[] = {&data_in, &bounds_out, (void*) &num_elements};

    CUDA_CHECK_ERROR(cuLaunchKernel(bounds_function_, gridDims.x, gridDims.y, gridDims.z, blockDims.x, blockDims.y,
        blockDims.z, 0, stream, args, nullptr));
}
