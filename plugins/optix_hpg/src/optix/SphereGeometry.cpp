#include "SphereGeometry.h"

#include <sstream>

#include "sphere.h"

#include "optix/CallGeometry.h"
#include "optix/Utils.h"

#include "optix_stubs.h"


namespace megamol::optix_hpg {
extern "C" const char embedded_sphere_programs[];
extern "C" const char embedded_sphere_occlusion_programs[];
} // namespace megamol::optix_hpg


megamol::optix_hpg::SphereGeometry::SphereGeometry() : _out_geo_slot("outGeo", ""), _in_data_slot("inData", "") {
    _out_geo_slot.SetCallback(CallGeometry::ClassName(), CallGeometry::FunctionName(0), &SphereGeometry::get_data_cb);
    _out_geo_slot.SetCallback(
        CallGeometry::ClassName(), CallGeometry::FunctionName(1), &SphereGeometry::get_extents_cb);
    MakeSlotAvailable(&_out_geo_slot);

    _in_data_slot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&_in_data_slot);
}


megamol::optix_hpg::SphereGeometry::~SphereGeometry() {
    this->Release();
}


bool megamol::optix_hpg::SphereGeometry::create() {
    return true;
}


void megamol::optix_hpg::SphereGeometry::release() {
    if (_geo_buffer != 0) {
        CUDA_CHECK_ERROR(cuMemFree(_geo_buffer));
    }
    /*if (_particle_data != 0) {
        CUDA_CHECK_ERROR(cuMemFree(_particle_data));
    }
    if (color_data_ != 0) {
        CUDA_CHECK_ERROR(cuMemFree(color_data_));
    }*/
    for (auto const& el : particle_data_) {
        CUDA_CHECK_ERROR(cuMemFree(el));
    }
    for (auto const& el : color_data_) {
        CUDA_CHECK_ERROR(cuMemFree(el));
    }
}


void megamol::optix_hpg::SphereGeometry::init(Context const& ctx) {
    sphere_module_ = MMOptixModule(embedded_sphere_programs, ctx.GetOptiXContext(), &ctx.GetModuleCompileOptions(),
        &ctx.GetPipelineCompileOptions(), MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_INTERSECTION, "sphere_intersect"},
            {MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_CLOSESTHIT, "sphere_closesthit"},
            {MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_BOUNDS, "sphere_bounds"}});
    sphere_occlusion_module_ = MMOptixModule(embedded_sphere_occlusion_programs, ctx.GetOptiXContext(),
        &ctx.GetModuleCompileOptions(), &ctx.GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_INTERSECTION, "sphere_intersect"},
            {MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_CLOSESTHIT, "sphere_closesthit_occlusion"},
            {MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_BOUNDS, "sphere_bounds_occlusion"}});

    ++program_version;

    // OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(sphere_module_, &_sbt_record));
}


bool megamol::optix_hpg::SphereGeometry::assertData(geocalls::MultiParticleDataCall& call, Context const& ctx) {
    auto const pl_count = call.GetParticleListCount();

    for (auto const& el : particle_data_) {
        CUDA_CHECK_ERROR(cuMemFree(el));
    }
    for (auto const& el : color_data_) {
        CUDA_CHECK_ERROR(cuMemFree(el));
    }

    particle_data_.resize(pl_count, 0);
    color_data_.resize(pl_count, 0);
    std::vector<CUdeviceptr> bounds_data(pl_count);
    std::vector<OptixBuildInput> build_inputs;
    sbt_records_.clear();

    for (unsigned int pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        // for now only the first geometry
        auto const& particles = call.AccessParticles(pl_idx);

        auto const p_count = particles.GetCount();
        if (p_count == 0)
            continue;

        auto const color_type = particles.GetColourDataType();
        auto const has_color = (color_type != geocalls::SimpleSphericalParticles::COLDATA_NONE) &&
                               (color_type != geocalls::SimpleSphericalParticles::COLDATA_DOUBLE_I) &&
                               (color_type != geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I);

        std::vector<device::Particle> data(p_count);
        auto x_acc = particles.GetParticleStore().GetXAcc();
        auto y_acc = particles.GetParticleStore().GetYAcc();
        auto z_acc = particles.GetParticleStore().GetZAcc();
        auto rad_acc = particles.GetParticleStore().GetRAcc();

        auto cr_acc = particles.GetParticleStore().GetCRAcc();
        auto cg_acc = particles.GetParticleStore().GetCGAcc();
        auto cb_acc = particles.GetParticleStore().GetCBAcc();
        auto ca_acc = particles.GetParticleStore().GetCAAcc();

        for (std::size_t p_idx = 0; p_idx < p_count; ++p_idx) {
            data[p_idx].pos.x = x_acc->Get_f(p_idx);
            data[p_idx].pos.y = y_acc->Get_f(p_idx);
            data[p_idx].pos.z = z_acc->Get_f(p_idx);
            data[p_idx].pos.w = rad_acc->Get_f(p_idx);
        }

        auto col_count = p_count;
        if (!has_color) {
            col_count = 0;
        }
        std::vector<glm::vec4> color_data(col_count);
        if (has_color) {
            for (std::size_t p_idx = 0; p_idx < col_count; ++p_idx) {
                color_data[p_idx].r = cr_acc->Get_f(p_idx);
                color_data[p_idx].g = cg_acc->Get_f(p_idx);
                color_data[p_idx].b = cb_acc->Get_f(p_idx);
                color_data[p_idx].a = ca_acc->Get_f(p_idx);
            }
            // CUDA_CHECK_ERROR(cuMemFree(color_data_));
            CUDA_CHECK_ERROR(cuMemAlloc(&color_data_[pl_idx], col_count * sizeof(glm::vec4)));
            CUDA_CHECK_ERROR(cuMemcpyHtoDAsync(
                color_data_[pl_idx], color_data.data(), col_count * sizeof(glm::vec4), ctx.GetExecStream()));
        }
        // CUDA_CHECK_ERROR(cuMemFree(_particle_data));
        CUDA_CHECK_ERROR(cuMemAlloc(&particle_data_[pl_idx], p_count * sizeof(device::Particle)));
        CUDA_CHECK_ERROR(cuMemcpyHtoDAsync(
            particle_data_[pl_idx], data.data(), p_count * sizeof(device::Particle), ctx.GetExecStream()));

        CUDA_CHECK_ERROR(cuMemAlloc(&bounds_data[pl_idx], p_count * sizeof(box3f)));

        sphere_module_.ComputeBounds(particle_data_[pl_idx], bounds_data[pl_idx], p_count, ctx.GetExecStream());

        //////////////////////////////////////
        // geometry
        //////////////////////////////////////

        build_inputs.emplace_back();
        unsigned int geo_flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        OptixBuildInput& buildInput = build_inputs.back();
        memset(&buildInput, 0, sizeof(OptixBuildInput));
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        auto& cp_input = buildInput.customPrimitiveArray;
        cp_input.aabbBuffers = &bounds_data[pl_idx];
        cp_input.numPrimitives = p_count;
        cp_input.primitiveIndexOffset = 0;
        cp_input.numSbtRecords = 1;
        cp_input.flags = &geo_flag;
        cp_input.sbtIndexOffsetBuffer = NULL;
        cp_input.sbtIndexOffsetSizeInBytes = 0;
        cp_input.sbtIndexOffsetStrideInBytes = 0;
        cp_input.strideInBytes = 0;

        SBTRecord<device::SphereGeoData> sbt_record;
        OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(sphere_module_, &sbt_record));

        sbt_record.data.particleBufferPtr = (device::Particle*) particle_data_[pl_idx];
        sbt_record.data.colorBufferPtr = nullptr;
        sbt_record.data.radius = particles.GetGlobalRadius();
        sbt_record.data.hasColorData = has_color;
        sbt_record.data.globalColor =
            glm::vec4(particles.GetGlobalColour()[0] / 255.f, particles.GetGlobalColour()[1] / 255.f,
                particles.GetGlobalColour()[2] / 255.f, particles.GetGlobalColour()[3] / 255.f);

        if (has_color) {
            sbt_record.data.colorBufferPtr = (glm::vec4*) color_data_[pl_idx];
        }
        sbt_records_.push_back(sbt_record);

        // occlusion stuff
        SBTRecord<device::SphereGeoData> sbt_record_occlusion;
        OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(sphere_occlusion_module_, &sbt_record_occlusion));
        sbt_record_occlusion.data = sbt_record.data;
        sbt_records_.push_back(sbt_record_occlusion);

        ++sbt_version;
    }

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    accelOptions.motionOptions.numKeys = 0;

    OptixAccelBufferSizes bufferSizes = {};
    OPTIX_CHECK_ERROR(optixAccelComputeMemoryUsage(
        ctx.GetOptiXContext(), &accelOptions, build_inputs.data(), build_inputs.size(), &bufferSizes));

    CUdeviceptr geo_temp;
    CUDA_CHECK_ERROR(cuMemFree(_geo_buffer));
    CUDA_CHECK_ERROR(cuMemAlloc(&_geo_buffer, bufferSizes.outputSizeInBytes));
    CUDA_CHECK_ERROR(cuMemAlloc(&geo_temp, bufferSizes.tempSizeInBytes));

    OptixTraversableHandle geo_handle = 0;
    OPTIX_CHECK_ERROR(optixAccelBuild(ctx.GetOptiXContext(), ctx.GetExecStream(), &accelOptions, build_inputs.data(),
        build_inputs.size(), geo_temp, bufferSizes.tempSizeInBytes, _geo_buffer, bufferSizes.outputSizeInBytes,
        &_geo_handle, nullptr, 0));

    CUDA_CHECK_ERROR(cuMemFree(geo_temp));
    // CUDA_CHECK_ERROR(cuMemFree(bounds_data));
    for (auto const& el : bounds_data) {
        CUDA_CHECK_ERROR(cuMemFree(el));
    }

    //////////////////////////////////////
    // end geometry
    //////////////////////////////////////

    return true;
}


bool megamol::optix_hpg::SphereGeometry::get_data_cb(core::Call& c) {
    auto out_geo = dynamic_cast<CallGeometry*>(&c);
    if (out_geo == nullptr)
        return false;
    auto in_data = _in_data_slot.CallAs<geocalls::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    auto const ctx = out_geo->get_ctx();

    static bool not_init = true;
    if (not_init) {
        init(*ctx);
        not_init = false;
    }

    in_data->SetFrameID(out_geo->FrameID());
    if (!(*in_data)(1))
        return false;
    if (!(*in_data)(0))
        return false;

    if (in_data->FrameID() != _frame_id || in_data->DataHash() != _data_hash) {
        if (!assertData(*in_data, *ctx))
            return false;
        _frame_id = in_data->FrameID();
        _data_hash = in_data->DataHash();
    }

    program_groups_[0] = sphere_module_;
    program_groups_[1] = sphere_occlusion_module_;

    out_geo->set_handle(&_geo_handle);
    out_geo->set_program_groups(program_groups_.data(), program_groups_.size(), program_version);
    out_geo->set_record(
        sbt_records_.data(), sbt_records_.size(), sizeof(SBTRecord<device::SphereGeoData>), sbt_version);

    return true;
}


bool megamol::optix_hpg::SphereGeometry::get_extents_cb(core::Call& c) {
    auto out_geo = dynamic_cast<CallGeometry*>(&c);
    if (out_geo == nullptr)
        return false;
    auto in_data = _in_data_slot.CallAs<geocalls::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    if ((*in_data)(1)) {
        out_geo->SetFrameCount(in_data->FrameCount());
        out_geo->AccessBoundingBoxes() = in_data->AccessBoundingBoxes();
    }

    return true;
}
