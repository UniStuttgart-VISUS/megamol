#include "ParticleMeshDistance.h"

#include "mmcore/param/IntParam.h"

#include "particlemeshdistance_device.h"

#include "RaysFromParticles.h"


namespace megamol::optix_hpg {
extern "C" const char embedded_particlemeshdistance_programs[];
}


megamol::optix_hpg::ParticleMeshDistance::ParticleMeshDistance()
        : out_stats_slot_("outStats", "")
        , in_data_slot_("inData", "")
        , in_mesh_slot_("inMesh", "")
        , frame_skip_slot_("frame skip", "") {
    out_stats_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &ParticleMeshDistance::get_data_cb);
    out_stats_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &ParticleMeshDistance::get_extent_cb);
    MakeSlotAvailable(&out_stats_slot_);

    in_data_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&in_data_slot_);

    in_mesh_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&in_mesh_slot_);

    frame_skip_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&frame_skip_slot_);
}


megamol::optix_hpg::ParticleMeshDistance::~ParticleMeshDistance() {
    this->Release();
}


bool megamol::optix_hpg::ParticleMeshDistance::create() {
    auto const fit = std::find_if(this->frontend_resources.begin(), this->frontend_resources.end(),
        [](auto const& el) { return el.getIdentifier() == frontend_resources::CUDA_Context_Req_Name; });

    if (fit == this->frontend_resources.end())
        return false;

    optix_ctx_ = std::make_unique<Context>(fit->getResource<frontend_resources::CUDA_Context>());

    init();

    return true;
}


void megamol::optix_hpg::ParticleMeshDistance::release() {}


bool megamol::optix_hpg::ParticleMeshDistance::init() {
    OptixBuiltinISOptions opts = {};
    opts.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
    opts.usesMotionBlur = false;
    OPTIX_CHECK_ERROR(optixBuiltinISModuleGet(optix_ctx_->GetOptiXContext(), &optix_ctx_->GetModuleCompileOptions(),
        &optix_ctx_->GetPipelineCompileOptions(), &opts, &builtin_triangle_intersector_));

    mesh_module_ = MMOptixModule(embedded_particlemeshdistance_programs, optix_ctx_->GetOptiXContext(),
        &optix_ctx_->GetModuleCompileOptions(), &optix_ctx_->GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_HITGROUP, builtin_triangle_intersector_,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_INTERSECTION, "pmd_intersect"},
            {MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_CLOSESTHIT, "pmd_closesthit"}});

    raygen_module_ = MMOptixModule(embedded_particlemeshdistance_programs, optix_ctx_->GetOptiXContext(),
        &optix_ctx_->GetModuleCompileOptions(), &optix_ctx_->GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_GENERIC, "pmd_raygen_program"}});

    miss_module_ = MMOptixModule(embedded_particlemeshdistance_programs, optix_ctx_->GetOptiXContext(),
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


bool megamol::optix_hpg::ParticleMeshDistance::assertData(
    mesh::CallMesh& mesh, core::moldyn::MultiParticleDataCall& particles, unsigned int frameID) {
    auto mesh_collection = mesh.getData();

    auto const& meshes = mesh_collection->accessMeshes();

    auto const& mesh_data = meshes.begin()->second; // TODO HAZARD

    if (mesh_data.primitive_type != mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES)
        return false;

    if (mesh_data.indices.type != mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT &&
        mesh_data.indices.type != mesh::MeshDataAccessCollection::ValueType::UNSIGNED_SHORT)
        return false;

    CUdeviceptr mesh_idx_data;
    CUDA_CHECK_ERROR(cuMemAlloc(&mesh_idx_data, mesh_data.indices.byte_size));
    CUDA_CHECK_ERROR(cuMemcpyHtoDAsync(
        mesh_idx_data, mesh_data.indices.data, mesh_data.indices.byte_size, optix_ctx_->GetExecStream()));

    auto const attributes = mesh_data.attributes;

    std::size_t idx_el_size = 0;
    if (mesh_data.indices.type == mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT) {
        idx_el_size = sizeof(unsigned int);
    } else if (mesh_data.indices.type == mesh::MeshDataAccessCollection::ValueType::UNSIGNED_SHORT) {
        idx_el_size = sizeof(unsigned short);
    }

    auto num_indices = mesh_data.indices.byte_size / (idx_el_size);
    auto num_vertices = 0;

    std::vector<float> tmp_data;
    float* vert_ptr = nullptr;
    float* norm_ptr = nullptr;

    for (auto const& attr : attributes) {
        if (attr.semantic != mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION)
            continue;
        if (attr.component_type != mesh::MeshDataAccessCollection::ValueType::FLOAT && attr.component_cnt != 3)
            continue;

        num_vertices = attr.byte_size / (3 * sizeof(float));
        tmp_data.reserve(num_vertices * 3);

        vert_ptr = reinterpret_cast<float*>(attr.data);
        for (std::size_t idx = 0; idx < 3 * num_vertices; ++idx) {
            tmp_data.push_back(vert_ptr[idx]);
        }
    }

    for (auto const& attr : attributes) {
        if (attr.semantic == mesh::MeshDataAccessCollection::AttributeSemanticType::NORMAL) {
            norm_ptr = reinterpret_cast<float*>(attr.data);
        }
    }

    if (tmp_data.empty())
        return false;

    mesh_access_collection_ = std::make_shared<mesh::MeshDataAccessCollection>();

    CUdeviceptr mesh_pos_data;
    CUDA_CHECK_ERROR(cuMemAlloc(&mesh_pos_data, tmp_data.size() * sizeof(float)));
    CUDA_CHECK_ERROR(cuMemcpyHtoDAsync(
        mesh_pos_data, tmp_data.data(), tmp_data.size() * sizeof(float), optix_ctx_->GetExecStream()));

    OptixBuildInput build_input = {};
    unsigned int geo_flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    // unsigned int geo_flag = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    memset(&build_input, 0, sizeof(OptixBuildInput));
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    auto& tr_input = build_input.triangleArray;
    tr_input.indexBuffer = mesh_idx_data;
    tr_input.indexFormat = mesh_data.indices.type == mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT
                               ? OPTIX_INDICES_FORMAT_UNSIGNED_INT3
                               : OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
    tr_input.indexStrideInBytes = mesh_data.indices.type == mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT
                                      ? 3 * sizeof(unsigned int)
                                      : 3 * sizeof(unsigned short);
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

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    accelOptions.motionOptions.numKeys = 0;

    OptixAccelBufferSizes bufferSizes = {};
    OPTIX_CHECK_ERROR(
        optixAccelComputeMemoryUsage(optix_ctx_->GetOptiXContext(), &accelOptions, &build_input, 1, &bufferSizes));

    CUdeviceptr geo_temp;
    CUdeviceptr geo_buffer;
    // OptixTraversableHandle geo_handle;
    // CUDA_CHECK_ERROR(cuMemFree(geo_buffer));
    CUDA_CHECK_ERROR(cuMemAlloc(&geo_buffer, bufferSizes.outputSizeInBytes));
    CUDA_CHECK_ERROR(cuMemAlloc(&geo_temp, bufferSizes.tempSizeInBytes));

    OptixTraversableHandle geo_handle = 0;
    OPTIX_CHECK_ERROR(
        optixAccelBuild(optix_ctx_->GetOptiXContext(), optix_ctx_->GetExecStream(), &accelOptions, &build_input, 1,
            geo_temp, bufferSizes.tempSizeInBytes, geo_buffer, bufferSizes.outputSizeInBytes, &geo_handle, nullptr, 0));

    CUDA_CHECK_ERROR(cuMemFree(geo_temp));

    SBTRecord<device::ParticleMeshDistanceData> mesh_sbt_record;
    OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(mesh_module_, &mesh_sbt_record));
    mesh_sbt_record.data.index_buffer = (glm::uvec3*) mesh_idx_data;
    mesh_sbt_record.data.vertex_buffer = (glm::vec3*) mesh_pos_data;
    mesh_sbt_record.data.num_tris = num_vertices / 3;
    mesh_sbt_record.data.world = geo_handle;

    ///----------------------------------------------

    auto const rays = rays_from_particles(particles, frameID, frame_skip_slot_.Param<core::param::IntParam>()->Value());

    part_mesh_distances_.resize(rays.size());
    pmd_minmax_.resize(rays.size());
    positions_.resize(rays.size());
    // radius_.resize(rays.size());

    for (unsigned int plIdx = 0; plIdx < rays.size(); ++plIdx) {
        auto const& ray_vec = rays[plIdx];
        auto& part_mesh_dis = part_mesh_distances_[plIdx];

        CUdeviceptr ray_buffer = 0;
        CUDA_CHECK_ERROR(cuMemAlloc(&ray_buffer, ray_vec.size() * sizeof(RayH)));
        CUDA_CHECK_ERROR(
            cuMemcpyHtoDAsync(ray_buffer, ray_vec.data(), ray_vec.size() * sizeof(RayH), optix_ctx_->GetExecStream()));

        CUdeviceptr ray_distances = 0;
        CUDA_CHECK_ERROR(cuMemAlloc(&ray_distances, ray_vec.size() * sizeof(float)));
        CUDA_CHECK_ERROR(cuMemsetD32Async(ray_distances, 0, ray_vec.size(), optix_ctx_->GetExecStream()));

        // SBTRecord<device::ParticleMeshDistanceData> mesh_sbt_record;
        mesh_sbt_record.data.distances = (float*) ray_distances;
        mesh_sbt_record.data.ray_buffer = (void*) ray_buffer;
        mesh_sbt_record.data.num_rays = ray_vec.size();

        SBTRecord<device::ParticleMeshDistanceData> raygen_record;
        OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(raygen_module_, &raygen_record));
        raygen_record.data = mesh_sbt_record.data;
        SBTRecord<device::ParticleMeshDistanceData> miss_record;
        OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(miss_module_, &miss_record));
        miss_record.data = mesh_sbt_record.data;

        sbt_.SetSBT(&raygen_record, sizeof(raygen_record), nullptr, 0, &miss_record, sizeof(miss_record), 1,
            &mesh_sbt_record, sizeof(mesh_sbt_record), 1, nullptr, 0, 0, optix_ctx_->GetExecStream());

        glm::uvec2 launch_dim = glm::uvec2(std::ceilf(std::sqrtf(static_cast<float>(ray_vec.size()))));
        OPTIX_CHECK_ERROR(
            optixLaunch(pipeline_, optix_ctx_->GetExecStream(), 0, 0, sbt_, launch_dim.x, launch_dim.y, 1));

        part_mesh_dis.resize(ray_vec.size());

        CUDA_CHECK_ERROR(cuMemcpyDtoHAsync(
            part_mesh_dis.data(), ray_distances, part_mesh_dis.size() * sizeof(float), optix_ctx_->GetExecStream()));
        CUDA_CHECK_ERROR(cuMemFree(ray_buffer));
        CUDA_CHECK_ERROR(cuMemFree(ray_distances));

        auto const minmax_el = std::minmax_element(part_mesh_dis.begin(), part_mesh_dis.end());
        pmd_minmax_[plIdx] = std::make_pair(*minmax_el.first, *minmax_el.second);

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


bool megamol::optix_hpg::ParticleMeshDistance::get_data_cb(core::Call& c) {
    auto out_data = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_data == nullptr)
        return false;
    auto in_mesh = in_mesh_slot_.CallAs<mesh::CallMesh>();
    if (in_mesh == nullptr)
        return false;
    auto in_data = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    /*static bool not_init = true;
    if (not_init) {
        init();
        not_init = false;
    }*/

    if (!(*in_mesh)(1))
        return false;
    if (!(*in_data)(1))
        return false;

    auto meta = in_mesh->getMetaData();
    meta.m_frame_ID = out_data->FrameID();
    in_mesh->setMetaData(meta);
    if (!(*in_mesh)(1))
        return false;
    if (!(*in_mesh)(0))
        return false;
    meta = in_mesh->getMetaData();
    in_data->SetFrameID(out_data->FrameID());
    if (!(*in_data)(1))
        return false;
    if (!(*in_data)(0))
        return false;

    if (/*in_data->hasUpdate()*/ meta.m_frame_ID != frame_id_ /*|| meta.m_data_hash != _data_hash*/) {
        if (!assertData(*in_mesh, *in_data, meta.m_frame_ID))
            return false;
        frame_id_ = meta.m_frame_ID;
        //_data_hash = meta.m_data_hash;
    }

    out_data->SetParticleListCount(part_mesh_distances_.size());
    out_data->SetDataHash(out_data_hash_);
    out_data->SetFrameID(frame_id_);

    for (unsigned int plIdx = 0; plIdx < part_mesh_distances_.size(); ++plIdx) {
        auto const& in_parts = in_data->AccessParticles(plIdx);
        auto& out_parts = out_data->AccessParticles(plIdx);
        auto const& pos = positions_[plIdx];
        auto const& col = part_mesh_distances_[plIdx];
        auto const& minmax_el = pmd_minmax_[plIdx];

        out_parts.SetCount(pos.size() / 3);
        out_parts.SetGlobalRadius(in_parts.GetGlobalRadius());
        out_parts.SetVertexData(core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, pos.data());
        out_parts.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, col.data());
        out_parts.SetColourMapIndexValues(minmax_el.first, minmax_el.second);
    }

    return true;
}


bool megamol::optix_hpg::ParticleMeshDistance::get_extent_cb(core::Call& c) {
    auto out_data = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_data == nullptr)
        return false;
    auto in_mesh = in_mesh_slot_.CallAs<mesh::CallMesh>();
    if (in_mesh == nullptr)
        return false;
    auto in_data = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    auto in_meta = in_mesh->getMetaData();
    in_meta.m_frame_ID = out_data->FrameID();
    if (!(*in_mesh)(1))
        return false;
    in_meta = in_mesh->getMetaData(), in_data->SetFrameCount(in_meta.m_frame_cnt);
    in_data->SetFrameID(in_meta.m_frame_ID);
    if (!(*in_data)(1))
        return false;

    out_data->SetFrameCount(in_data->FrameCount());
    out_data->SetFrameID(in_data->FrameID());
    out_data->AccessBoundingBoxes() = in_data->AccessBoundingBoxes();

    return true;
}
