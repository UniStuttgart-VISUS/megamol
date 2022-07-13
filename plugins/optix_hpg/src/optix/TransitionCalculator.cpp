#include "TransitionCalculator.h"

#include "glm/glm.hpp"

#include "optix/utils_host.h"

#include "transitioncalculator_device.h"


namespace megamol::optix_hpg {
extern "C" const char embedded_transitioncalculator_programs[];
}


megamol::optix_hpg::TransitionCalculator::TransitionCalculator()
        : out_transitions_slot_("outTransitions", "")
        , in_mesh_slot_("inMesh", "")
        , in_paths_slot_("inPaths", "") {
    out_transitions_slot_.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &TransitionCalculator::get_data_cb);
    out_transitions_slot_.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &TransitionCalculator::get_extent_cb);
    MakeSlotAvailable(&out_transitions_slot_);

    in_mesh_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&in_mesh_slot_);

    in_paths_slot_.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&in_paths_slot_);
}


megamol::optix_hpg::TransitionCalculator::~TransitionCalculator() {
    this->Release();
}


bool megamol::optix_hpg::TransitionCalculator::create() {
    auto& cuda_res = frontend_resources.get<frontend_resources::CUDA_Context>();
    if (cuda_res.ctx_ != nullptr) {
        optix_ctx_ = std::make_unique<Context>(cuda_res);
    } else {
        return false;
    }
    return true;
}


void megamol::optix_hpg::TransitionCalculator::release() {}


bool megamol::optix_hpg::TransitionCalculator::init() {
    OptixBuiltinISOptions opts = {};
    opts.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
    opts.usesMotionBlur = false;
    OPTIX_CHECK_ERROR(optixBuiltinISModuleGet(optix_ctx_->GetOptiXContext(), &optix_ctx_->GetModuleCompileOptions(),
        &optix_ctx_->GetPipelineCompileOptions(), &opts, &builtin_triangle_intersector_));

    mesh_module_ = MMOptixModule(embedded_transitioncalculator_programs, optix_ctx_->GetOptiXContext(),
        &optix_ctx_->GetModuleCompileOptions(), &optix_ctx_->GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_HITGROUP, builtin_triangle_intersector_,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_INTERSECTION, "tc_intersect"},
            {MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_CLOSESTHIT, "tc_closesthit"}});

    raygen_module_ = MMOptixModule(embedded_transitioncalculator_programs, optix_ctx_->GetOptiXContext(),
        &optix_ctx_->GetModuleCompileOptions(), &optix_ctx_->GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_GENERIC, "tc_raygen_program"}});

    miss_module_ = MMOptixModule(embedded_transitioncalculator_programs, optix_ctx_->GetOptiXContext(),
        &optix_ctx_->GetModuleCompileOptions(), &optix_ctx_->GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_MISS,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_GENERIC, "tc_miss_program"}});

    std::array<OptixProgramGroup, 3> groups = {raygen_module_, miss_module_, mesh_module_};

    char log[2048];
    std::size_t log_size = 2048;

    OPTIX_CHECK_ERROR(optixPipelineCreate(optix_ctx_->GetOptiXContext(), &optix_ctx_->GetPipelineCompileOptions(),
        &optix_ctx_->GetPipelineLinkOptions(), groups.data(), 3, log, &log_size, &pipeline_));

    return true;
}


bool megamol::optix_hpg::TransitionCalculator::assertData(
    mesh::CallMesh& mesh, geocalls::MultiParticleDataCall& particles, unsigned int frameID) {
    // fetch data for current and subsequent timestep
    // create array of rays for the current particle configuration
    // upload mesh as geometry
    // trace rays with mesh and count inbound and outbound transistions (aka intersections with mesh)

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

    for (auto const& attr : attributes) {
        if (attr.semantic != mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION)
            continue;
        if (attr.component_type != mesh::MeshDataAccessCollection::ValueType::FLOAT && attr.component_cnt != 3)
            continue;

        num_vertices = attr.byte_size / (3 * sizeof(float));
        tmp_data.reserve(num_vertices * 3);

        auto vert_ptr = reinterpret_cast<float*>(attr.data);
        for (std::size_t idx = 0; idx < 3 * num_vertices; ++idx) {
            tmp_data.push_back(vert_ptr[idx]);
        }
    }

    if (tmp_data.empty())
        return false;

    CUdeviceptr mesh_pos_data;
    CUDA_CHECK_ERROR(cuMemAlloc(&mesh_pos_data, tmp_data.size() * sizeof(float)));
    CUDA_CHECK_ERROR(cuMemcpyHtoDAsync(
        mesh_pos_data, tmp_data.data(), tmp_data.size() * sizeof(float), optix_ctx_->GetExecStream()));

    OptixBuildInput build_input = {};
    unsigned int geo_flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
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

    SBTRecord<device::TransitionCalculatorData> mesh_sbt_record;
    OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(mesh_module_, &mesh_sbt_record));
    mesh_sbt_record.data.index_buffer = (glm::uvec3*)mesh_idx_data;
    mesh_sbt_record.data.vertex_buffer = (glm::vec3*)mesh_pos_data;

    ///----------------------------------------------

    for (auto el : ray_buffer_) {
        CUDA_CHECK_ERROR(cuMemFree(el));
    }

    bool got_0 = false;
    do {
        particles.SetFrameID(frameID, true);
        got_0 = particles(1);
        got_0 = got_0 && particles(0);
    } while (particles.FrameID() != frameID && !got_0);

    auto const plCount0 = particles.GetParticleListCount();

    std::vector<std::vector<std::pair<glm::vec3, glm::vec3>>> origins(plCount0);
    std::vector<std::vector<std::uint64_t>> idx_map(plCount0);

    for (unsigned int plIdx = 0; plIdx < plCount0; ++plIdx) {
        auto const& parts = particles.AccessParticles(plIdx);
        auto const pCount = parts.GetCount();

        auto& orgs = origins[plIdx];
        orgs.resize(pCount);

        auto& idc = idx_map[plIdx];
        idc.resize(pCount);

        auto xAcc = parts.GetParticleStore().GetXAcc();
        auto yAcc = parts.GetParticleStore().GetYAcc();
        auto zAcc = parts.GetParticleStore().GetZAcc();
        auto idAcc = parts.GetParticleStore().GetIDAcc();

        for (std::size_t pIdx = 0; pIdx < pCount; ++pIdx) {
            orgs[pIdx] = std::make_pair(glm::vec3(xAcc->Get_f(pIdx), yAcc->Get_f(pIdx), zAcc->Get_f(pIdx)),
                glm::vec3(std::numeric_limits<float>::lowest()));
            idc[pIdx] = idAcc->Get_u64(pIdx);
        }
    }

    bool got_1 = false;
    do {
        particles.SetFrameID(frameID + 1, true);
        got_1 = particles(1);
        got_1 = got_1 && particles(0);
    } while (particles.FrameID() != frameID + 1 && !got_1);

    auto const plCount1 = particles.GetParticleListCount();

    for (unsigned int plIdx = 0; plIdx < plCount1; ++plIdx) {
        auto const& parts = particles.AccessParticles(plIdx);
        auto const pCount = parts.GetCount();

        auto& orgs = origins[plIdx];
        auto const& idc = idx_map[plIdx];

        auto xAcc = parts.GetParticleStore().GetXAcc();
        auto yAcc = parts.GetParticleStore().GetYAcc();
        auto zAcc = parts.GetParticleStore().GetZAcc();
        auto idAcc = parts.GetParticleStore().GetIDAcc();

        for (std::size_t pIdx = 0; pIdx < pCount; ++pIdx) {
            auto const id = idAcc->Get_u64(pIdx);
            auto fit = std::find(idc.begin(), idc.end(), id);
            if (fit != idc.end()) {
                auto const idx = std::distance(idc.begin(), fit);
                orgs[idx].second = glm::vec3(xAcc->Get_f(pIdx), yAcc->Get_f(pIdx), zAcc->Get_f(pIdx));
            }
        }
    }


    std::vector<std::vector<RayH>> rays(plCount0);
    for (unsigned int plIdx = 0; plIdx < plCount0; ++plIdx) {
        auto const& orgs = origins[plIdx];

        auto& ray_vec = rays[plIdx];
        ray_vec.reserve(orgs.size());

        for (auto const& el : orgs) {
            auto const& origin = el.first;
            auto const& dest = el.second;

            if (dest.x < std::numeric_limits<float>::lowest() + std::numeric_limits<float>::epsilon() &&
                dest.y < std::numeric_limits<float>::lowest() + std::numeric_limits<float>::epsilon() &&
                dest.z < std::numeric_limits<float>::lowest() + std::numeric_limits<float>::epsilon())
                continue;

            auto const& dir = dest - origin;
            auto const tMax = glm::length(dir);
            ray_vec.emplace_back(origin, glm::normalize(dir), 0.f, tMax);
        }

        ray_buffer_.push_back(0);
        CUDA_CHECK_ERROR(cuMemAlloc(&ray_buffer_.back(), ray_vec.size() * sizeof(RayH)));
        CUDA_CHECK_ERROR(cuMemcpyHtoDAsync(
            ray_buffer_.back(), ray_vec.data(), ray_vec.size() * sizeof(RayH), optix_ctx_->GetExecStream()));

        CUdeviceptr mesh_inbound_ctr, mesh_outbound_ctr, ray_state;

        CUDA_CHECK_ERROR(cuMemAlloc(&mesh_inbound_ctr, (num_vertices / 3) * sizeof(std::uint32_t)));
        CUDA_CHECK_ERROR(cuMemAlloc(&mesh_outbound_ctr, (num_vertices / 3) * sizeof(std::uint32_t)));
        CUDA_CHECK_ERROR(cuMemAlloc(&ray_state, ray_vec.size()));

        CUDA_CHECK_ERROR(cuMemsetD32Async(mesh_inbound_ctr, 0, (num_vertices / 3), optix_ctx_->GetExecStream()));
        CUDA_CHECK_ERROR(cuMemsetD32Async(mesh_outbound_ctr, 0, (num_vertices / 3), optix_ctx_->GetExecStream()));
        CUDA_CHECK_ERROR(cuMemsetD8Async(ray_state, 0, ray_vec.size(), optix_ctx_->GetExecStream()));

        mesh_sbt_record.data.mesh_inbound_ctr_ptr = (std::uint32_t*)mesh_inbound_ctr;
        mesh_sbt_record.data.mesh_outbound_ctr_ptr = (std::uint32_t*)mesh_outbound_ctr;
        mesh_sbt_record.data.ray_state = (std::uint8_t*)ray_state;
        mesh_sbt_record.data.ray_buffer = (void*)ray_buffer_.back();
        mesh_sbt_record.data.num_rays = ray_vec.size();
        mesh_sbt_record.data.num_tris = num_vertices / 3;
        mesh_sbt_record.data.world = geo_handle;

        SBTRecord<device::TransitionCalculatorData> raygen_record;
        OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(raygen_module_, &raygen_record));
        raygen_record.data = mesh_sbt_record.data;
        SBTRecord<device::TransitionCalculatorData> miss_record;
        OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(miss_module_, &miss_record));
        miss_record.data = mesh_sbt_record.data;

        // launch
        sbt_.SetSBT(&raygen_record, sizeof(raygen_record), nullptr, 0, &miss_record, sizeof(miss_record), 1,
            &mesh_sbt_record, sizeof(mesh_sbt_record), 1, nullptr, 0, 0, optix_ctx_->GetExecStream());

        glm::uvec2 launch_dim = glm::uvec2(std::ceilf(std::sqrtf(static_cast<float>(ray_vec.size()))));
        OPTIX_CHECK_ERROR(
            optixLaunch(pipeline_, optix_ctx_->GetExecStream(), 0, 0, sbt_, launch_dim.x, launch_dim.y, 1));


        std::vector<std::uint32_t> mesh_inbound_vec(num_vertices / 3, 0);
        std::vector<std::uint32_t> mesh_outbound_vec(num_vertices / 3, 0);
        std::vector<std::uint8_t> ray_state_vec(ray_vec.size());

        CUDA_CHECK_ERROR(cuMemcpyDtoHAsync(mesh_inbound_vec.data(), mesh_inbound_ctr,
            (num_vertices / 3) * sizeof(std::uint32_t), optix_ctx_->GetExecStream()));
        CUDA_CHECK_ERROR(cuMemcpyDtoHAsync(mesh_outbound_vec.data(), mesh_outbound_ctr,
            (num_vertices / 3) * sizeof(std::uint32_t), optix_ctx_->GetExecStream()));
        CUDA_CHECK_ERROR(
            cuMemcpyDtoHAsync(ray_state_vec.data(), ray_state, ray_vec.size(), optix_ctx_->GetExecStream()));

        CUDA_CHECK_ERROR(cuMemFree(mesh_inbound_ctr));
        CUDA_CHECK_ERROR(cuMemFree(mesh_outbound_ctr));
        CUDA_CHECK_ERROR(cuMemFree(ray_state));
    }


    return true;
}


bool megamol::optix_hpg::TransitionCalculator::get_data_cb(core::Call& c) {
    auto out_geo = dynamic_cast<mesh::CallMesh*>(&c);
    if (out_geo == nullptr)
        return false;
    auto in_mesh = in_mesh_slot_.CallAs<mesh::CallMesh>();
    if (in_mesh == nullptr)
        return false;
    auto in_paths = in_paths_slot_.CallAs<geocalls::MultiParticleDataCall>();
    if (in_paths == nullptr)
        return false;

    static bool not_init = true;
    if (not_init) {
        init();
        not_init = false;
    }

    if (!(*in_mesh)(1))
        return false;
    if (!(*in_paths)(1))
        return false;

    auto meta = in_mesh->getMetaData();
    auto out_meta = out_geo->getMetaData();
    meta.m_frame_ID = out_meta.m_frame_ID;
    in_mesh->setMetaData(meta);
    if (!(*in_mesh)(1))
        return false;
    if (!(*in_mesh)(0))
        return false;
    meta = in_mesh->getMetaData();
    in_paths->SetFrameID(out_meta.m_frame_ID);
    if (!(*in_paths)(1))
        return false;
    if (!(*in_paths)(0))
        return false;

    if (/*in_data->hasUpdate()*/ meta.m_frame_ID != _frame_id /*|| meta.m_data_hash != _data_hash*/) {
        if (!assertData(*in_mesh, *in_paths, meta.m_frame_ID))
            return false;
        _frame_id = meta.m_frame_ID;
        //_data_hash = meta.m_data_hash;
    }

    /*program_groups_[0] = mesh_module_;
    program_groups_[1] = mesh_occlusion_module_;

    out_geo->set_handle(&_geo_handle);
    out_geo->set_program_groups(program_groups_.data());
    out_geo->set_num_programs(2);
    out_geo->set_record(sbt_records_.data());
    out_geo->set_record_stride(sizeof(SBTRecord<device::MeshGeoData>));
    out_geo->set_num_records(sbt_records_.size());*/
    out_geo->setMetaData(meta);
    out_geo->setData(in_mesh->getData(), in_mesh->version());

    return true;
}


bool megamol::optix_hpg::TransitionCalculator::get_extent_cb(core::Call& c) {
    auto out_geo = dynamic_cast<mesh::CallMesh*>(&c);
    if (out_geo == nullptr)
        return false;
    auto in_data = in_mesh_slot_.CallAs<mesh::CallMesh>();
    if (in_data == nullptr)
        return false;

    if ((*in_data)(1) && (*in_data)(0)) {
        auto const meta = in_data->getMetaData();
        out_geo->setMetaData(meta);
    }

    return true;
}
