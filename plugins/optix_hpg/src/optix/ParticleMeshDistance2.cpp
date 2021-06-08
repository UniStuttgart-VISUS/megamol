#include "ParticleMeshDistance2.h"

#include "mmcore/param/IntParam.h"


#include "RaysFromParticles.h"


namespace megamol::optix_hpg {
extern "C" const char embedded_particlemeshdistance2_programs[];
}


megamol::optix_hpg::ParticleMeshDistance2::ParticleMeshDistance2()
        : out_stats_slot_("outStats", "")
        , in_data_slot_("inData", "")
        , in_part_mesh_slot_("inPartMesh", "")
        , in_inter_mesh_slot_("inInterMesh", "")
        , frame_skip_slot_("frame skip", "") {
    out_stats_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &ParticleMeshDistance2::get_data_cb);
    out_stats_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &ParticleMeshDistance2::get_extent_cb);
    MakeSlotAvailable(&out_stats_slot_);

    in_data_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&in_data_slot_);

    in_part_mesh_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&in_part_mesh_slot_);

    in_inter_mesh_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&in_inter_mesh_slot_);

    frame_skip_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&frame_skip_slot_);
}


megamol::optix_hpg::ParticleMeshDistance2::~ParticleMeshDistance2() {
    this->Release();
}


bool megamol::optix_hpg::ParticleMeshDistance2::create() {
    auto& cuda_res = frontend_resources.get<frontend_resources::CUDA_Context>();
    if (cuda_res.ctx_ != nullptr) {
        optix_ctx_ = std::make_unique<Context>(cuda_res);
    } else {
        return false;
    }

    init();

    return true;
}


void megamol::optix_hpg::ParticleMeshDistance2::release() {}


bool megamol::optix_hpg::ParticleMeshDistance2::init() {
    OptixBuiltinISOptions opts = {};
    opts.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
    opts.usesMotionBlur = false;
    OPTIX_CHECK_ERROR(optixBuiltinISModuleGet(optix_ctx_->GetOptiXContext(), &optix_ctx_->GetModuleCompileOptions(),
        &optix_ctx_->GetPipelineCompileOptions(), &opts, &builtin_triangle_intersector_));

    mesh_module_ = MMOptixModule(embedded_particlemeshdistance2_programs, optix_ctx_->GetOptiXContext(),
        &optix_ctx_->GetModuleCompileOptions(), &optix_ctx_->GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_HITGROUP, builtin_triangle_intersector_,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_INTERSECTION, "pmd_intersect"},
            {MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_CLOSESTHIT, "pmd_closesthit"},
            {MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_ANYHIT, "pmd_anyhit"}});

    raygen_module_ = MMOptixModule(embedded_particlemeshdistance2_programs, optix_ctx_->GetOptiXContext(),
        &optix_ctx_->GetModuleCompileOptions(), &optix_ctx_->GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_GENERIC, "pmd_raygen_program"}});

    miss_module_ = MMOptixModule(embedded_particlemeshdistance2_programs, optix_ctx_->GetOptiXContext(),
        &optix_ctx_->GetModuleCompileOptions(), &optix_ctx_->GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_MISS,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_GENERIC, "pmd_miss_program"}});

    std::array<OptixProgramGroup, 3> groups = {raygen_module_, miss_module_, mesh_module_};

    char log[2048];
    std::size_t log_size = 2048;

    OPTIX_CHECK_ERROR(optixPipelineCreate(optix_ctx_->GetOptiXContext(), &optix_ctx_->GetPipelineCompileOptions(),
        &optix_ctx_->GetPipelineLinkOptions(), groups.data(), 3, log, &log_size, &pipeline_));

    return true;
}


std::tuple<OptixTraversableHandle,
    std::vector<megamol::optix_hpg::SBTRecord<megamol::optix_hpg::device::ParticleMeshDistanceData2>>>
megamol::optix_hpg::ParticleMeshDistance2::setupMeshGeometry(mesh::CallMesh& mesh) {
    auto const mesh_coll = mesh.getData();

    std::vector<OptixBuildInput> build_inputs;
    std::vector<SBTRecord<device::ParticleMeshDistanceData2>> sbt_records;

    for (auto const& [name, mesh] : mesh_coll->accessMeshes()) {
        if (mesh.primitive_type != mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES)
            continue;

        if (mesh.indices.type != mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT)
            continue;

        auto const num_indices = mesh.indices.byte_size / sizeof(unsigned int);

        CUdeviceptr mesh_idx_data;
        CUDA_CHECK_ERROR(cuMemAlloc(&mesh_idx_data, mesh.indices.byte_size));
        CUDA_CHECK_ERROR(
            cuMemcpyHtoDAsync(mesh_idx_data, mesh.indices.data, mesh.indices.byte_size, optix_ctx_->GetExecStream()));

        auto const attributes = mesh.attributes;

        auto const pos_fit = std::find_if(attributes.begin(), attributes.end(), [](auto const& el) {
            return el.semantic == mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION;
        });

        if (pos_fit == attributes.end() ||
            pos_fit->component_type != mesh::MeshDataAccessCollection::ValueType::FLOAT ||
            (pos_fit->stride != 0 && pos_fit->stride != 3 * sizeof(float)))
            continue;

        auto const num_vertices = pos_fit->byte_size / (3 * sizeof(float));

        CUdeviceptr mesh_pos_data;
        CUDA_CHECK_ERROR(cuMemAlloc(&mesh_pos_data, pos_fit->byte_size));
        CUDA_CHECK_ERROR(
            cuMemcpyHtoDAsync(mesh_pos_data, pos_fit->data, pos_fit->byte_size, optix_ctx_->GetExecStream()));

        OptixBuildInput build_input = {};
        unsigned int geo_flag = OPTIX_GEOMETRY_FLAG_NONE;
        // unsigned int geo_flag = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
        memset(&build_input, 0, sizeof(OptixBuildInput));
        build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        auto& tr_input = build_input.triangleArray;
        tr_input.indexBuffer = mesh_idx_data;
        tr_input.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        tr_input.indexStrideInBytes = 3 * sizeof(unsigned int);
        tr_input.numIndexTriplets = num_indices / 3;
        tr_input.numSbtRecords = 1;
        tr_input.primitiveIndexOffset = 0;
        tr_input.numVertices = num_vertices;
        tr_input.vertexBuffers = &mesh_pos_data;
        tr_input.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        tr_input.vertexStrideInBytes = 3 * sizeof(float);
        tr_input.flags = &geo_flag;
        tr_input.preTransform = 0;
        tr_input.sbtIndexOffsetBuffer = NULL;
        tr_input.sbtIndexOffsetSizeInBytes = 0;
        tr_input.sbtIndexOffsetStrideInBytes = 0;

        build_inputs.push_back(build_input);

        SBTRecord<device::ParticleMeshDistanceData2> mesh_sbt_record;
        OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(mesh_module_, &mesh_sbt_record));
        mesh_sbt_record.data.index_buffer = (glm::uvec3*) mesh_idx_data;
        mesh_sbt_record.data.vertex_buffer = (glm::vec3*) mesh_pos_data;
        mesh_sbt_record.data.num_tris = num_vertices / 3;

        sbt_records.push_back(mesh_sbt_record);
    }

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    accelOptions.motionOptions.numKeys = 0;

    OptixAccelBufferSizes bufferSizes = {};
    OPTIX_CHECK_ERROR(optixAccelComputeMemoryUsage(
        optix_ctx_->GetOptiXContext(), &accelOptions, build_inputs.data(), build_inputs.size(), &bufferSizes));

    CUdeviceptr geo_temp;
    CUdeviceptr geo_buffer;
    // OptixTraversableHandle geo_handle;
    // CUDA_CHECK_ERROR(cuMemFree(geo_buffer));
    CUDA_CHECK_ERROR(cuMemAlloc(&geo_buffer, bufferSizes.outputSizeInBytes));
    CUDA_CHECK_ERROR(cuMemAlloc(&geo_temp, bufferSizes.tempSizeInBytes));

    OptixTraversableHandle geo_handle = 0;
    OPTIX_CHECK_ERROR(optixAccelBuild(optix_ctx_->GetOptiXContext(), optix_ctx_->GetExecStream(), &accelOptions,
        build_inputs.data(), build_inputs.size(), geo_temp, bufferSizes.tempSizeInBytes, geo_buffer,
        bufferSizes.outputSizeInBytes, &geo_handle, nullptr, 0));

    CUDA_CHECK_ERROR(cuMemFree(geo_temp));

    // std::for_each(sbt_records.begin(), sbt_records.end(), [&geo_handle](auto& el) { el.data.world = geo_handle; });

    return std::make_tuple(geo_handle, sbt_records);
}


bool megamol::optix_hpg::ParticleMeshDistance2::assertData(mesh::CallMesh& part_mesh, mesh::CallMesh& inter_mesh,
    core::moldyn::MultiParticleDataCall& particles, unsigned int frameID) {
    auto [part_world, part_sbt_records] = setupMeshGeometry(part_mesh);
    auto [inter_world, inter_sbt_records] = setupMeshGeometry(inter_mesh);

    ///----------------------------------------------

    auto rays = rays_from_particles_rng(particles, frameID);

    part_mesh_distances_.resize(rays.size());
    part_mesh_occup_.resize(rays.size());
    pmd_minmax_.resize(rays.size());
    pmd_occup_minmax_.resize(rays.size());
    positions_.resize(rays.size());
    // radius_.resize(rays.size());

    for (unsigned int plIdx = 0; plIdx < rays.size(); ++plIdx) {
        auto& ray_vec = rays[plIdx];
        auto& part_mesh_dis = part_mesh_distances_[plIdx];
        auto& part_mesh_occup = part_mesh_occup_[plIdx];

        std::for_each(ray_vec.begin(), ray_vec.end(), [](auto& el) { el.tMax = std::numeric_limits<float>::max(); });

        CUdeviceptr ray_buffer = 0;
        CUDA_CHECK_ERROR(cuMemAlloc(&ray_buffer, ray_vec.size() * sizeof(RayH)));
        CUDA_CHECK_ERROR(
            cuMemcpyHtoDAsync(ray_buffer, ray_vec.data(), ray_vec.size() * sizeof(RayH), optix_ctx_->GetExecStream()));

        CUdeviceptr ray_distances = 0;
        CUDA_CHECK_ERROR(cuMemAlloc(&ray_distances, ray_vec.size() * sizeof(float)));
        CUDA_CHECK_ERROR(cuMemsetD32Async(ray_distances, 0, ray_vec.size(), optix_ctx_->GetExecStream()));
        CUdeviceptr ray_inter_count = 0;
        CUDA_CHECK_ERROR(cuMemAlloc(&ray_inter_count, ray_vec.size() * sizeof(unsigned int)));
        CUDA_CHECK_ERROR(cuMemsetD32Async(ray_inter_count, 0, ray_vec.size(), optix_ctx_->GetExecStream()));

        // SBTRecord<device::ParticleMeshDistanceData> mesh_sbt_record;
        std::for_each(part_sbt_records.begin(), part_sbt_records.end(), [&ray_distances, &ray_inter_count](auto& el) {
            el.data.distances = (float*) ray_distances;
            el.data.inter_count = (unsigned int*) ray_inter_count;
        });
        std::for_each(inter_sbt_records.begin(), inter_sbt_records.end(), [&ray_distances, &ray_inter_count](auto& el) {
            el.data.distances = (float*) ray_distances;
            el.data.inter_count = (unsigned int*) ray_inter_count;
        });
        /*mesh_sbt_record.data.distances = (float*) ray_distances;
        mesh_sbt_record.data.ray_buffer = (void*) ray_buffer;
        mesh_sbt_record.data.num_rays = ray_vec.size();*/

        SBTRecord<device::PMDRayGenData> raygen_record;
        OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(raygen_module_, &raygen_record));
        raygen_record.data.ray_buffer = (void*) ray_buffer;
        raygen_record.data.num_rays = ray_vec.size();
        raygen_record.data.world = part_world;
        SBTRecord<device::PMDRayGenData> miss_record;
        OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(miss_module_, &miss_record));
        miss_record.data = raygen_record.data;
        SBTRecord<device::PMDRayGenData> inter_raygen_record;
        OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(raygen_module_, &inter_raygen_record));
        inter_raygen_record.data.ray_buffer = (void*) ray_buffer;
        inter_raygen_record.data.num_rays = ray_vec.size();
        inter_raygen_record.data.world = inter_world;
        SBTRecord<device::PMDRayGenData> inter_miss_record;
        OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(miss_module_, &inter_miss_record));
        miss_record.data = inter_raygen_record.data;

        part_sbt_.SetSBT(&raygen_record, sizeof(raygen_record), nullptr, 0, &miss_record, sizeof(miss_record), 1,
            part_sbt_records.data(), sizeof(decltype(part_sbt_records)::value_type), part_sbt_records.size(), nullptr,
            0, 0, optix_ctx_->GetExecStream());

        inter_sbt_.SetSBT(&inter_raygen_record, sizeof(inter_raygen_record), nullptr, 0, &inter_miss_record, sizeof(inter_miss_record), 1,
            inter_sbt_records.data(), sizeof(decltype(inter_sbt_records)::value_type), inter_sbt_records.size(),
            nullptr, 0, 0, optix_ctx_->GetExecStream());

        glm::uvec2 launch_dim = glm::uvec2(std::ceilf(std::sqrtf(static_cast<float>(ray_vec.size()))));
        OPTIX_CHECK_ERROR(
            optixLaunch(pipeline_, optix_ctx_->GetExecStream(), 0, 0, part_sbt_, launch_dim.x, launch_dim.y, 1));

        part_mesh_dis.resize(ray_vec.size());

        CUDA_CHECK_ERROR(cuMemcpyDtoHAsync(
            part_mesh_dis.data(), ray_distances, part_mesh_dis.size() * sizeof(float), optix_ctx_->GetExecStream()));

        std::vector<unsigned int> part_occup(ray_vec.size());
        CUDA_CHECK_ERROR(cuMemcpyDtoHAsync(
            part_occup.data(), ray_inter_count, part_occup.size() * sizeof(unsigned int), optix_ctx_->GetExecStream()));

        CUDA_CHECK_ERROR(cuMemsetD32Async(ray_distances, 0, ray_vec.size(), optix_ctx_->GetExecStream()));
        CUDA_CHECK_ERROR(cuMemsetD32Async(ray_inter_count, 0, ray_vec.size(), optix_ctx_->GetExecStream()));

        OPTIX_CHECK_ERROR(
            optixLaunch(pipeline_, optix_ctx_->GetExecStream(), 0, 0, inter_sbt_, launch_dim.x, launch_dim.y, 1));

        std::vector<unsigned int> inter_occup(ray_vec.size());
        CUDA_CHECK_ERROR(cuMemcpyDtoHAsync(inter_occup.data(), ray_inter_count,
            inter_occup.size() * sizeof(unsigned int), optix_ctx_->GetExecStream()));

        CUDA_CHECK_ERROR(cuMemFree(ray_buffer));
        CUDA_CHECK_ERROR(cuMemFree(ray_distances));
        CUDA_CHECK_ERROR(cuMemFree(ray_inter_count));

        auto const minmax_el = std::minmax_element(part_mesh_dis.begin(), part_mesh_dis.end());
        pmd_minmax_[plIdx] = std::make_pair(*minmax_el.first, *minmax_el.second);

        part_mesh_occup.resize(ray_vec.size());
        for (size_t i = 0; i < ray_vec.size(); ++i) {
            auto const part_val = part_occup[i];
            auto const inter_val = inter_occup[i];

            auto const part_cond = (part_val % 2) == 0;
            auto const inter_cond = (inter_val % 2) == 1;

            part_mesh_occup[i] = 0.0f;

            if (part_cond && inter_cond) {
                part_mesh_occup[i] = 1.0f;
            } else {
                if (!part_cond)
                    part_mesh_occup[i] = 2.0f;
                if (!inter_cond)
                    part_mesh_occup[i] = 3.0f;
                //part_mesh_occup[i] = 0.0f;
            }
        }

        pmd_occup_minmax_[plIdx] = std::make_pair(0.0f, 3.0f);

        auto const& parts = particles.AccessParticles(plIdx);
        auto const xAcc = parts.GetParticleStore().GetXAcc();
        auto const yAcc = parts.GetParticleStore().GetYAcc();
        auto const zAcc = parts.GetParticleStore().GetZAcc();

        auto& pos = positions_[plIdx];
        pos.reserve(parts.GetCount() * 3);

        for (uint64_t pidx = 0; pidx < parts.GetCount(); ++pidx) {
            pos.push_back(xAcc->Get_f(pidx));
            pos.push_back(yAcc->Get_f(pidx));
            pos.push_back(zAcc->Get_f(pidx));
        }
    }

    ++out_data_hash_;

    return true;
}


bool megamol::optix_hpg::ParticleMeshDistance2::get_data_cb(core::Call& c) {
    auto out_data = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_data == nullptr)
        return false;
    auto in_part_mesh = in_part_mesh_slot_.CallAs<mesh::CallMesh>();
    if (in_part_mesh == nullptr)
        return false;
    auto in_inter_mesh = in_inter_mesh_slot_.CallAs<mesh::CallMesh>();
    if (in_inter_mesh == nullptr)
        return false;
    auto in_data = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    /*static bool not_init = true;
    if (not_init) {
        init();
        not_init = false;
    }*/

    auto req_frame_id = out_data->FrameID();

    auto in_part_meta = in_part_mesh->getMetaData();
    in_part_meta.m_frame_ID = req_frame_id;
    in_part_mesh->setMetaData(in_part_meta);

    auto in_inter_meta = in_inter_mesh->getMetaData();
    in_inter_meta.m_frame_ID = req_frame_id;
    in_inter_mesh->setMetaData(in_inter_meta);

    in_data->SetFrameID(req_frame_id);

    if (!(*in_part_mesh)(1))
        return false;
    if (!(*in_part_mesh)(0))
        return false;
    if (!(*in_inter_mesh)(1))
        return false;
    if (!(*in_inter_mesh)(0))
        return false;
    if (!(*in_data)(1))
        return false;
    if (!(*in_data)(0))
        return false;

    static bool first = true;
    // if (/*in_data->hasUpdate()*/ meta.m_frame_ID != frame_id_ /*|| meta.m_data_hash != _data_hash*/) {
    /*if (in_part_mesh->hasUpdate() || in_inter_mesh->hasUpdate() || in_data->DataHash() != in_data_hash_ ||
        in_data->FrameID() != frame_id_)*/if(first) {
        if (!assertData(*in_part_mesh, *in_inter_mesh, *in_data, in_data->FrameID()))
            return false;
        frame_id_ = in_data->FrameID();
        in_data_hash_ = in_data->DataHash();
        first = false;
    }

    out_data->SetParticleListCount(part_mesh_distances_.size());
    out_data->SetDataHash(out_data_hash_);
    out_data->SetFrameID(frame_id_);

    for (unsigned int plIdx = 0; plIdx < part_mesh_distances_.size(); ++plIdx) {
        auto const& in_parts = in_data->AccessParticles(plIdx);
        auto& out_parts = out_data->AccessParticles(plIdx);
        auto const& pos = positions_[plIdx];
        //auto const& col = part_mesh_distances_[plIdx];
        //auto const& minmax_el = pmd_minmax_[plIdx];
        auto const& col = part_mesh_occup_[plIdx];
        auto const& minmax_el = pmd_occup_minmax_[plIdx];

        out_parts.SetCount(pos.size() / 3);
        out_parts.SetGlobalRadius(in_parts.GetGlobalRadius());
        out_parts.SetVertexData(core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, pos.data());
        out_parts.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, col.data());
        out_parts.SetColourMapIndexValues(minmax_el.first, minmax_el.second);
    }

    return true;
}


bool megamol::optix_hpg::ParticleMeshDistance2::get_extent_cb(core::Call& c) {
    auto out_data = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_data == nullptr)
        return false;
    auto in_part_mesh = in_part_mesh_slot_.CallAs<mesh::CallMesh>();
    if (in_part_mesh == nullptr)
        return false;
    auto in_inter_mesh = in_inter_mesh_slot_.CallAs<mesh::CallMesh>();
    if (in_inter_mesh == nullptr)
        return false;
    auto in_data = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    auto req_frame_id = out_data->FrameID();

    auto in_part_meta = in_part_mesh->getMetaData();
    in_part_meta.m_frame_ID = req_frame_id;
    in_part_mesh->setMetaData(in_part_meta);

    auto in_inter_meta = in_inter_mesh->getMetaData();
    in_inter_meta.m_frame_ID = req_frame_id;
    in_inter_mesh->setMetaData(in_inter_meta);

    in_data->SetFrameID(req_frame_id);

    if (!(*in_part_mesh)(1))
        return false;
    if (!(*in_inter_mesh)(1))
        return false;
    if (!(*in_data)(1))
        return false;

    out_data->SetFrameCount(in_data->FrameCount());
    out_data->SetFrameID(in_data->FrameID());
    out_data->AccessBoundingBoxes() = in_data->AccessBoundingBoxes();

    return true;
}
