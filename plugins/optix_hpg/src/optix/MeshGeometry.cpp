#include "MeshGeometry.h"

#include "optix/CallGeometry.h"


namespace megamol::optix_hpg {
extern "C" const char embedded_mesh_programs[];
}


megamol::optix_hpg::MeshGeometry::MeshGeometry() : _out_geo_slot("outGeo", ""), _in_data_slot("inData", "") {
    _out_geo_slot.SetCallback(CallGeometry::ClassName(), CallGeometry::FunctionName(0), &MeshGeometry::get_data_cb);
    _out_geo_slot.SetCallback(CallGeometry::ClassName(), CallGeometry::FunctionName(1), &MeshGeometry::get_extents_cb);
    MakeSlotAvailable(&_out_geo_slot);

    _in_data_slot.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&_in_data_slot);
}


megamol::optix_hpg::MeshGeometry::~MeshGeometry() {
    this->Release();
}


bool megamol::optix_hpg::MeshGeometry::create() {
    return true;
}


void megamol::optix_hpg::MeshGeometry::release() {}


void megamol::optix_hpg::MeshGeometry::init(Context const& ctx) {
    OptixBuiltinISOptions opts = {};
    opts.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
    opts.usesMotionBlur = false;
    OPTIX_CHECK_ERROR(optixBuiltinISModuleGet(ctx.GetOptiXContext(), &ctx.GetModuleCompileOptions(),
        &ctx.GetPipelineCompileOptions(), &opts, &triangle_intersector_));

    mesh_module_ = MMOptixModule(embedded_mesh_programs, ctx.GetOptiXContext(), &ctx.GetModuleCompileOptions(),
        &ctx.GetPipelineCompileOptions(), MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        triangle_intersector_,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_INTERSECTION, "mesh_intersect"},
            {MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_CLOSESTHIT, "mesh_closesthit"}});
    mesh_occlusion_module_ = MMOptixModule(embedded_mesh_programs, ctx.GetOptiXContext(),
        &ctx.GetModuleCompileOptions(), &ctx.GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_HITGROUP, triangle_intersector_,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_INTERSECTION, "mesh_intersect"},
            {MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_CLOSESTHIT, "mesh_closesthit_occlusion"}});

    ++program_version;
}


bool megamol::optix_hpg::MeshGeometry::assertData(mesh::CallMesh& call, Context const& ctx) {
    auto mesh_collection = call.getData();

    auto const& meshes = mesh_collection->accessMeshes();

    for (auto const& el : mesh_idx_data_) {
        CUDA_CHECK_ERROR(cuMemFree(el));
    }
    mesh_idx_data_.clear();
    for (auto const& el : mesh_pos_data_) {
        CUDA_CHECK_ERROR(cuMemFree(el));
    }
    mesh_pos_data_.clear();
    sbt_records_.clear();

    std::vector<OptixBuildInput> build_inputs;

    for (auto const& mesh : meshes) {
        auto const& mesh_data = mesh.second;
        if (mesh_data.primitive_type != mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES)
            continue;

        if (mesh_data.indices.type != mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT &&
            mesh_data.indices.type != mesh::MeshDataAccessCollection::ValueType::UNSIGNED_SHORT)
            continue;

        mesh_idx_data_.push_back(0);
        CUDA_CHECK_ERROR(cuMemAlloc(&mesh_idx_data_.back(), mesh_data.indices.byte_size));
        CUDA_CHECK_ERROR(cuMemcpyHtoDAsync(
            mesh_idx_data_.back(), mesh_data.indices.data, mesh_data.indices.byte_size, ctx.GetExecStream()));

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

        mesh_pos_data_.push_back(0);
        CUDA_CHECK_ERROR(cuMemAlloc(&mesh_pos_data_.back(), tmp_data.size() * sizeof(float)));
        CUDA_CHECK_ERROR(cuMemcpyHtoDAsync(
            mesh_pos_data_.back(), tmp_data.data(), tmp_data.size() * sizeof(float), ctx.GetExecStream()));

        build_inputs.emplace_back();
        unsigned int geo_flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        auto& build_input = build_inputs.back();
        memset(&build_input, 0, sizeof(OptixBuildInput));
        build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        auto& tr_input = build_input.triangleArray;
        tr_input.indexBuffer = mesh_idx_data_.back();
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
        tr_input.vertexBuffers = &mesh_pos_data_.back();
        tr_input.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        tr_input.vertexStrideInBytes = 3 * sizeof(float);
        tr_input.flags = &geo_flag;
        tr_input.preTransform = 0;
        tr_input.sbtIndexOffsetBuffer = NULL;
        tr_input.sbtIndexOffsetSizeInBytes = 0;
        tr_input.sbtIndexOffsetStrideInBytes = 0;

        SBTRecord<device::MeshGeoData> sbt_record;
        OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(mesh_module_, &sbt_record));
        sbt_record.data.index_buffer = (glm::uvec3*)mesh_idx_data_.back();
        sbt_record.data.vertex_buffer = (glm::vec3*)mesh_pos_data_.back();

        sbt_records_.push_back(sbt_record);

        // occlusion stuff
        SBTRecord<device::MeshGeoData> sbt_record_occlusion;
        OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(mesh_occlusion_module_, &sbt_record_occlusion));
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

    return true;
}


bool megamol::optix_hpg::MeshGeometry::get_data_cb(core::Call& c) {
    auto out_geo = dynamic_cast<CallGeometry*>(&c);
    if (out_geo == nullptr)
        return false;
    auto in_data = _in_data_slot.CallAs<mesh::CallMesh>();
    if (in_data == nullptr)
        return false;

    auto const ctx = out_geo->get_ctx();

    static bool not_init = true;
    if (not_init) {
        init(*ctx);
        not_init = false;
    }

    if (!(*in_data)(1))
        return false;
    auto meta = in_data->getMetaData();
    meta.m_frame_ID = out_geo->FrameID();
    in_data->setMetaData(meta);
    if (!(*in_data)(1))
        return false;
    if (!(*in_data)(0))
        return false;
    meta = in_data->getMetaData();

    if (in_data->hasUpdate() || meta.m_frame_ID != _frame_id /*|| meta.m_data_hash != _data_hash*/) {
        if (!assertData(*in_data, *ctx))
            return false;
        _frame_id = meta.m_frame_ID;
        //_data_hash = meta.m_data_hash;
    }

    program_groups_[0] = mesh_module_;
    program_groups_[1] = mesh_occlusion_module_;

    out_geo->set_handle(&_geo_handle);
    out_geo->set_program_groups(program_groups_.data(), program_groups_.size(), program_version);
    out_geo->set_record(sbt_records_.data(), sbt_records_.size(), sizeof(SBTRecord<device::MeshGeoData>), sbt_version);

    return true;
}


bool megamol::optix_hpg::MeshGeometry::get_extents_cb(core::Call& c) {
    auto out_geo = dynamic_cast<CallGeometry*>(&c);
    if (out_geo == nullptr)
        return false;
    auto in_data = _in_data_slot.CallAs<mesh::CallMesh>();
    if (in_data == nullptr)
        return false;

    if ((*in_data)(1) && (*in_data)(0)) {
        auto const meta = in_data->getMetaData();
        out_geo->SetFrameCount(meta.m_frame_cnt);
        out_geo->AccessBoundingBoxes().SetObjectSpaceBBox(meta.m_bboxs.BoundingBox());
        out_geo->AccessBoundingBoxes().SetObjectSpaceClipBox(meta.m_bboxs.ClipBox());
    }

    return true;
}
