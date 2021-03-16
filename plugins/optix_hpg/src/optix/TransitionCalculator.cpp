#include "stdafx.h"
#include "TransitionCalculator.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/view/CallGetTransferFunction.h"

#include "glm/glm.hpp"

#include "optix/utils_host.h"

#include "transitioncalculator_device.h"


namespace megamol::optix_hpg {
extern "C" const char embedded_transitioncalculator_programs[];
}


megamol::optix_hpg::TransitionCalculator::TransitionCalculator()
        : out_transitions_slot_("outTransitions", "")
        , in_mesh_slot_("inMesh", "")
        , in_paths_slot_("inPaths", "")
        , in_tf_slot_("inTF", "")
        , output_type_slot_("output type", "")
        , frame_count_slot_("frame count", "")
        , frame_skip_slot_("frame skip", "") {
    out_transitions_slot_.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &TransitionCalculator::get_data_cb);
    out_transitions_slot_.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &TransitionCalculator::get_extent_cb);
    MakeSlotAvailable(&out_transitions_slot_);

    in_mesh_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&in_mesh_slot_);

    in_paths_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&in_paths_slot_);

    in_tf_slot_.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    MakeSlotAvailable(&in_tf_slot_);

    using output_type_ut = std::underlying_type_t<output_type>;
    auto ep = new core::param::EnumParam(static_cast<output_type_ut>(output_type::outbound));
    ep->SetTypePair(static_cast<output_type_ut>(output_type::inbound), "inbound");
    ep->SetTypePair(static_cast<output_type_ut>(output_type::outbound), "outbound");
    output_type_slot_ << ep;
    MakeSlotAvailable(&output_type_slot_);

    frame_count_slot_ << new core::param::IntParam(10, 1);
    MakeSlotAvailable(&frame_count_slot_);

    frame_skip_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&frame_skip_slot_);
}


megamol::optix_hpg::TransitionCalculator::~TransitionCalculator() {
    this->Release();
}


bool megamol::optix_hpg::TransitionCalculator::create() {
    auto const fit = std::find_if(this->frontend_resources.begin(), this->frontend_resources.end(),
        [](auto const& el) { return el.getIdentifier() == frontend_resources::CUDA_Context_Req_Name; });

    if (fit == this->frontend_resources.end())
        return false;

    optix_ctx_ = std::make_unique<Context>(fit->getResource<frontend_resources::CUDA_Context>());

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
        OPTIX_PROGRAM_GROUP_KIND_HITGROUP, builtin_triangle_intersector_, {"tc_intersect", "tc_closesthit"});

    raygen_module_ = MMOptixModule(embedded_transitioncalculator_programs, optix_ctx_->GetOptiXContext(),
        &optix_ctx_->GetModuleCompileOptions(), &optix_ctx_->GetPipelineCompileOptions(),
        OPTIX_PROGRAM_GROUP_KIND_RAYGEN, {"tc_raygen_program"});

    miss_module_ = MMOptixModule(embedded_transitioncalculator_programs, optix_ctx_->GetOptiXContext(),
        &optix_ctx_->GetModuleCompileOptions(), &optix_ctx_->GetPipelineCompileOptions(), OPTIX_PROGRAM_GROUP_KIND_MISS,
        {"tc_miss_program"});

    std::array<OptixProgramGroup, 3> groups = {raygen_module_, miss_module_, mesh_module_};

    char log[2048];
    std::size_t log_size = 2048;

    OPTIX_CHECK_ERROR(optixPipelineCreate(optix_ctx_->GetOptiXContext(), &optix_ctx_->GetPipelineCompileOptions(),
        &optix_ctx_->GetPipelineLinkOptions(), groups.data(), 3, log, &log_size, &pipeline_));

    return true;
}


inline float tf_lerp(float a, float b, float inter) {
    return a * (1.0f - inter) + b * inter;
}


inline glm::vec4 sample_tf(float const* tf, unsigned int tf_size, int base, float rest) {
    if (base < 0 || tf_size == 0)
        return glm::vec4(0);
    auto const last_el = tf_size - 1;
    if (base >= last_el)
        return glm::vec4(tf[last_el * 4], tf[last_el * 4 + 1], tf[last_el * 4 + 2], tf[last_el * 4 + 3]);

    auto const a = base;
    auto const b = base + 1;

    return glm::vec4(tf_lerp(tf[a * 4], tf[b * 4], rest), tf_lerp(tf[a * 4 + 1], tf[b * 4 + 1], rest),
        tf_lerp(tf[a * 4 + 2], tf[b * 4 + 2], rest), tf_lerp(tf[a * 4 + 3], tf[b * 4 + 3], rest));
}


bool megamol::optix_hpg::TransitionCalculator::assertData(mesh::CallMesh& mesh,
    core::moldyn::MultiParticleDataCall& particles, core::view::CallGetTransferFunction& tf, unsigned int frameID) {
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
    float* vert_ptr = nullptr;

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

    if (tmp_data.empty())
        return false;

    mesh_access_collection_ = std::make_shared<mesh::MeshDataAccessCollection>();

    CUdeviceptr mesh_pos_data;
    CUDA_CHECK_ERROR(cuMemAlloc(&mesh_pos_data, tmp_data.size() * sizeof(float)));
    CUDA_CHECK_ERROR(cuMemcpyHtoDAsync(
        mesh_pos_data, tmp_data.data(), tmp_data.size() * sizeof(float), optix_ctx_->GetExecStream()));

    OptixBuildInput build_input = {};
    unsigned int geo_flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    //unsigned int geo_flag = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
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
    mesh_sbt_record.data.index_buffer = (glm::uvec3*) mesh_idx_data;
    mesh_sbt_record.data.vertex_buffer = (glm::vec3*) mesh_pos_data;

    ///----------------------------------------------

    CUdeviceptr mesh_inbound_ctr, mesh_outbound_ctr, ray_state;

    for (auto el : ray_buffer_) {
        CUDA_CHECK_ERROR(cuMemFree(el));
    }
    auto const plCount0 = particles.GetParticleListCount();
    std::vector<std::vector<RayH>> rays(plCount0);
    std::vector<std::vector<std::pair<glm::vec3, glm::vec3>>> origins(plCount0);
    std::vector<std::vector<std::uint64_t>> idx_map(plCount0);

    CUDA_CHECK_ERROR(cuMemAlloc(&mesh_inbound_ctr, (num_vertices / 3) * sizeof(std::uint32_t)));
    CUDA_CHECK_ERROR(cuMemAlloc(&mesh_outbound_ctr, (num_vertices / 3) * sizeof(std::uint32_t)));


    CUDA_CHECK_ERROR(cuMemsetD32Async(mesh_inbound_ctr, 0, (num_vertices / 3), optix_ctx_->GetExecStream()));
    CUDA_CHECK_ERROR(cuMemsetD32Async(mesh_outbound_ctr, 0, (num_vertices / 3), optix_ctx_->GetExecStream()));

    auto const frame_count = frame_count_slot_.Param<core::param::IntParam>()->Value();
    auto const frame_skip = frame_skip_slot_.Param<core::param::IntParam>()->Value();

    for (int fid = frameID; fid < frameID + frame_count; fid += frame_skip) {

        bool got_0 = false;
        do {
            particles.SetFrameID(fid, true);
            got_0 = particles(1);
            got_0 = got_0 && particles(0);
        } while (particles.FrameID() != fid && !got_0);


        /*std::vector<std::vector<std::pair<glm::vec3, glm::vec3>>> origins(plCount0);
        std::vector<std::vector<std::uint64_t>> idx_map(plCount0);*/

        for (unsigned int plIdx = 0; plIdx < plCount0; ++plIdx) {
            auto const& parts = particles.AccessParticles(plIdx);
            auto const pCount = parts.GetCount();

            auto& orgs = origins[plIdx];
            auto& idc = idx_map[plIdx];
            if (fid == frameID) {
                orgs.resize(pCount);
                /*std::fill(orgs.begin(), orgs.end(),
                    std::make_pair<glm::vec3, glm::vec3>(glm::vec3(std::numeric_limits<float>::lowest()),
                        glm::vec3(std::numeric_limits<float>::lowest())));*/
                idc.resize(pCount);
            }
            std::fill(orgs.begin(), orgs.end(),
                std::make_pair<glm::vec3, glm::vec3>(
                    glm::vec3(std::numeric_limits<float>::lowest()), glm::vec3(std::numeric_limits<float>::lowest())));

            auto xAcc = parts.GetParticleStore().GetXAcc();
            auto yAcc = parts.GetParticleStore().GetYAcc();
            auto zAcc = parts.GetParticleStore().GetZAcc();
            auto idAcc = parts.GetParticleStore().GetIDAcc();

            if (fid == frameID) {
                for (std::size_t pIdx = 0; pIdx < pCount; ++pIdx) {
                    orgs[pIdx] = std::make_pair(glm::vec3(xAcc->Get_f(pIdx), yAcc->Get_f(pIdx), zAcc->Get_f(pIdx)),
                        glm::vec3(std::numeric_limits<float>::lowest()));
                    idc[pIdx] = idAcc->Get_u64(pIdx);
                }
            } else {
                for (std::size_t pIdx = 0; pIdx < pCount; ++pIdx) {
                    auto const id = idAcc->Get_u64(pIdx);
                    auto fit = std::find(idc.begin(), idc.end(), id);
                    if (fit != idc.end()) {
                        auto const idx = std::distance(idc.begin(), fit);
                        orgs[idx] = std::make_pair(glm::vec3(xAcc->Get_f(pIdx), yAcc->Get_f(pIdx), zAcc->Get_f(pIdx)),
                            glm::vec3(std::numeric_limits<float>::lowest()));
                        // idc[pIdx] = idAcc->Get_u64(pIdx);
                    }

                    /*orgs[pIdx] = std::make_pair(glm::vec3(xAcc->Get_f(pIdx), yAcc->Get_f(pIdx), zAcc->Get_f(pIdx)),
                        glm::vec3(std::numeric_limits<float>::lowest()));
                    idc[pIdx] = idAcc->Get_u64(pIdx);*/
                }
            }
        }

        bool got_1 = false;
        do {
            particles.SetFrameID(fid + frame_skip, true);
            got_1 = particles(1);
            got_1 = got_1 && particles(0);
        } while (particles.FrameID() != fid + frame_skip && !got_1);

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


        positions_.resize(plCount0);
        colors_.resize(plCount0);
        indices_.resize(plCount0);
        for (unsigned int plIdx = 0; plIdx < plCount0; ++plIdx) {
            auto const& orgs = origins[plIdx];

            auto& ray_vec = rays[plIdx];
            ray_vec.clear();
            ray_vec.reserve(orgs.size());


            for (auto const& el : orgs) {
                auto const& origin = el.first;
                auto const& dest = el.second;

                if (origin.x <= std::numeric_limits<float>::lowest() + std::numeric_limits<float>::epsilon() ||
                    origin.y <= std::numeric_limits<float>::lowest() + std::numeric_limits<float>::epsilon() ||
                    origin.z <= std::numeric_limits<float>::lowest() + std::numeric_limits<float>::epsilon() ||
                    dest.x <= std::numeric_limits<float>::lowest() + std::numeric_limits<float>::epsilon() ||
                    dest.y <= std::numeric_limits<float>::lowest() + std::numeric_limits<float>::epsilon() ||
                    dest.z <= std::numeric_limits<float>::lowest() + std::numeric_limits<float>::epsilon())
                    continue;

                auto const& dir = dest - origin;
                auto const tMax = glm::length(dir);
                ray_vec.emplace_back(origin, glm::normalize(dir), 0.f, tMax);
            }

            ray_buffer_.push_back(0);
            CUDA_CHECK_ERROR(cuMemAlloc(&ray_buffer_.back(), ray_vec.size() * sizeof(RayH)));
            CUDA_CHECK_ERROR(cuMemcpyHtoDAsync(
                ray_buffer_.back(), ray_vec.data(), ray_vec.size() * sizeof(RayH), optix_ctx_->GetExecStream()));

            CUDA_CHECK_ERROR(cuMemAlloc(&ray_state, ray_vec.size()));
            CUDA_CHECK_ERROR(cuMemsetD8Async(ray_state, 0, ray_vec.size(), optix_ctx_->GetExecStream()));


            mesh_sbt_record.data.mesh_inbound_ctr_ptr = (std::uint32_t*) mesh_inbound_ctr;
            mesh_sbt_record.data.mesh_outbound_ctr_ptr = (std::uint32_t*) mesh_outbound_ctr;
            mesh_sbt_record.data.ray_state = (std::uint8_t*) ray_state;
            mesh_sbt_record.data.ray_buffer = (void*) ray_buffer_.back();
            mesh_sbt_record.data.num_rays = ray_vec.size();
            mesh_sbt_record.data.num_tris = num_vertices / 3;
            mesh_sbt_record.data.world = geo_handle;

            SBTRecord<device::TransitionCalculatorData> raygen_record;
            OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(raygen_module_, &raygen_record));
            raygen_record.data = mesh_sbt_record.data;
            SBTRecord<device::TransitionCalculatorData> miss_record;
            OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(miss_module_, &miss_record));
            miss_record.data = mesh_sbt_record.data;

            core::utility::log::Log::DefaultLog.WriteInfo("[TransitionCalculator] Starting computation");
            // launch
            sbt_.SetSBT(&raygen_record, sizeof(raygen_record), nullptr, 0, &miss_record, sizeof(miss_record), 1,
                &mesh_sbt_record, sizeof(mesh_sbt_record), 1, nullptr, 0, 0, optix_ctx_->GetExecStream());

            glm::uvec2 launch_dim = glm::uvec2(std::ceilf(std::sqrtf(static_cast<float>(ray_vec.size()))));
            OPTIX_CHECK_ERROR(
                optixLaunch(pipeline_, optix_ctx_->GetExecStream(), 0, 0, sbt_, launch_dim.x, launch_dim.y, 1));
        }
    }

    for (unsigned int plIdx = 0; plIdx < plCount0; ++plIdx) {
        auto& positions = positions_[plIdx];
        auto& colors = colors_[plIdx];
        auto& indices = indices_[plIdx];

        auto& ray_vec = rays[plIdx];

        /*auto file = std::ofstream("bla1.txt");
        for (size_t i = 0; i < ray_vec.size(); ++i) {
            file << std::to_string(ray_vec[i].origin.x) << std::to_string(ray_vec[i].origin.y)
                 << std::to_string(ray_vec[i].origin.z) << std::to_string(ray_vec[i].direction.x)
                 << std::to_string(ray_vec[i].direction.y) << std::to_string(ray_vec[i].direction.z)
                 << std::to_string(ray_vec[i].tMin) << std::to_string(ray_vec[i].tMax) << '\n';
        }
        file.close();*/

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

        auto const num_rel_trans =
            std::count_if(ray_state_vec.begin(), ray_state_vec.end(), [](auto el) { return el == 3; });

        int best_idx = -1;
        auto best_fit = std::find(ray_state_vec.begin(), ray_state_vec.end(), 3);
        if (best_fit != ray_state_vec.end()) {
            best_idx = std::distance(ray_state_vec.begin(), best_fit);
            best_idx = idx_map[plIdx][best_idx];
        }

        core::utility::log::Log::DefaultLog.WriteInfo(
            "[TransitionCalculator] Ending computation with %d relevant transitions from %d particles at base %d beginning at %d", num_rel_trans, ray_vec.size(), frameID, best_idx);

        auto mesh_indices = reinterpret_cast<glm::u32vec3 const*>(mesh_data.indices.data);
        auto vert_glm_ptr = reinterpret_cast<glm::vec3 const*>(vert_ptr);

        using output_type_ut = std::underlying_type_t<output_type>;
        auto const type = static_cast<output_type>(output_type_slot_.Param<core::param::EnumParam>()->Value());
        auto data_ptr = &mesh_outbound_vec;
        switch (type) {
        case output_type::inbound: {
            data_ptr = &mesh_inbound_vec;
            /*colors.clear();
            colors.reserve(mesh_inbound_vec.size() * 3);
            positions.clear();
            positions.reserve(mesh_inbound_vec.size() * 3);
            indices.clear();
            indices.reserve(num_vertices / 3);
            auto const minmax = std::minmax_element(mesh_inbound_vec.begin(), mesh_inbound_vec.end());
            tf.SetRange({static_cast<float>(*minmax.first), static_cast<float>(*minmax.second)});
            tf();
            auto const range = tf.Range();

            auto const color_tf = tf.GetTextureData();
            auto const color_tf_size = tf.TextureSize();

            for (std::size_t i = 0; i < mesh_inbound_vec.size(); ++i) {
                auto const val_a = (static_cast<float>(mesh_inbound_vec[i]) - range[0]) / (range[1] - range[0]);
                std::remove_const_t<decltype(val_a)> main_a = 0;
                auto rest_a = std::modf(val_a, &main_a);
                rest_a = static_cast<int>(main_a) >= 0 && static_cast<int>(main_a) < color_tf_size ? rest_a : 0.0f;

                main_a = std::clamp<float>(static_cast<int>(main_a), 0, color_tf_size - 1);

                auto const col = sample_tf(color_tf, color_tf_size, static_cast<int>(main_a), rest_a);
                colors.push_back(col);
                colors.push_back(col);
                colors.push_back(col);

                auto const idx = mesh_indices[i];
                positions.push_back(vert_glm_ptr[idx.x]);
                positions.push_back(vert_glm_ptr[idx.y]);
                positions.push_back(vert_glm_ptr[idx.z]);

                indices.push_back(idx.x);
                indices.push_back(idx.y);
                indices.push_back(idx.z);
            }*/
        } break;
        case output_type::outbound: {
            data_ptr = &mesh_outbound_vec;
            // colors.clear();
            // colors.reserve(mesh_outbound_vec.size() * 3);
            // positions.clear();
            // positions.reserve(mesh_outbound_vec.size() * 3);
            // auto const minmax = std::minmax_element(mesh_outbound_vec.begin(), mesh_outbound_vec.end());
            // tf.SetRange({static_cast<float>(*minmax.first), static_cast<float>(*minmax.second)});
            // tf();
            ////auto const range = tf.Range();
            // auto const range =
            //    std::array<float, 2>({static_cast<float>(*minmax.first), static_cast<float>(*minmax.second)});

            // auto const color_tf = tf.GetTextureData();
            // auto const color_tf_size = tf.TextureSize();

            // for (std::size_t i = 0; i < mesh_outbound_vec.size(); ++i) {
            //    auto const val_a = (static_cast<float>(mesh_outbound_vec[i]) - range[0]) / (range[1] - range[0]) *
            //                       static_cast<float>(color_tf_size);
            //    std::remove_const_t<decltype(val_a)> main_a = 0;
            //    auto rest_a = std::modf(val_a, &main_a);
            //    rest_a = static_cast<int>(main_a) >= 0 && static_cast<int>(main_a) < color_tf_size ? rest_a : 0.0f;

            //    main_a = std::clamp<float>(static_cast<int>(main_a), 0, color_tf_size - 1);

            //    auto const col = sample_tf(color_tf, color_tf_size, static_cast<int>(main_a), rest_a);
            //    // auto const col = glm::vec4(1.0f);
            //    colors.push_back(col);
            //    colors.push_back(col);
            //    colors.push_back(col);

            //    auto const idx = mesh_indices[i];
            //    positions.push_back(vert_glm_ptr[idx.x]);
            //    positions.push_back(vert_glm_ptr[idx.y]);
            //    positions.push_back(vert_glm_ptr[idx.z]);

            //    indices.push_back(idx.x);
            //    indices.push_back(idx.y);
            //    indices.push_back(idx.z);
            //}
        } break;
        }
        colors.clear();
        colors.reserve(data_ptr->size() * 3);
        positions.clear();
        positions.reserve(data_ptr->size() * 3);
        indices.clear();
        indices.reserve(num_vertices / 3);
        auto const minmax = std::minmax_element(data_ptr->begin(), data_ptr->end());
        tf.SetRange({static_cast<float>(*minmax.first), static_cast<float>(*minmax.second)});
        tf();
        // auto const range = tf.Range();
        auto const range =
            std::array<float, 2>({static_cast<float>(*minmax.first), static_cast<float>(*minmax.second)});

        auto const color_tf = tf.GetTextureData();
        auto const color_tf_size = tf.TextureSize();

        for (std::size_t i = 0; i < data_ptr->size(); ++i) {
            auto const val_a = (static_cast<float>((*data_ptr)[i]) - range[0]) / (range[1] - range[0]) *
                               static_cast<float>(color_tf_size);
            std::remove_const_t<decltype(val_a)> main_a = 0;
            auto rest_a = std::modf(val_a, &main_a);
            rest_a = static_cast<int>(main_a) >= 0 && static_cast<int>(main_a) < color_tf_size ? rest_a : 0.0f;

            main_a = std::clamp<float>(static_cast<int>(main_a), 0, color_tf_size - 1);

            auto const col = sample_tf(color_tf, color_tf_size, static_cast<int>(main_a), rest_a);
            colors.push_back(col);
            colors.push_back(col);
            colors.push_back(col);

            auto const idx = mesh_indices[i];
            positions.push_back(vert_glm_ptr[idx.x]);
            positions.push_back(vert_glm_ptr[idx.y]);
            positions.push_back(vert_glm_ptr[idx.z]);

            indices.push_back(idx.x);
            indices.push_back(idx.y);
            indices.push_back(idx.z);
        }

        // std::iota(indices.begin(), indices.end(), 0);

        std::vector<mesh::MeshDataAccessCollection::VertexAttribute> mesh_attributes;
        mesh::MeshDataAccessCollection::IndexData cmd_indices;

        cmd_indices.byte_size = indices.size() * sizeof(uint32_t);
        cmd_indices.data = reinterpret_cast<uint8_t*>(indices.data());
        cmd_indices.type = mesh::MeshDataAccessCollection::UNSIGNED_INT;

        /*mesh_attributes.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
            reinterpret_cast<uint8_t*>(normals.data()), normals.size() * sizeof(float), 3,
            mesh::MeshDataAccessCollection::FLOAT, sizeof(float) * 3, 0, mesh::MeshDataAccessCollection::NORMAL});*/
        mesh_attributes.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
            reinterpret_cast<uint8_t*>(colors.data()), colors.size() * 4 * sizeof(float), 4,
            mesh::MeshDataAccessCollection::FLOAT, sizeof(float) * 4, 0, mesh::MeshDataAccessCollection::COLOR});
        mesh_attributes.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
            reinterpret_cast<uint8_t*>(positions.data()), positions.size() * 3 * sizeof(float), 3,
            mesh::MeshDataAccessCollection::FLOAT, sizeof(float) * 3, 0, mesh::MeshDataAccessCollection::POSITION});

        auto identifier = std::string("particle_surface_") + std::to_string(plIdx);
        mesh_access_collection_->addMesh(identifier, mesh_attributes, cmd_indices);
    }

    ++out_data_hash_;

    return true;
}


bool megamol::optix_hpg::TransitionCalculator::get_data_cb(core::Call& c) {
    auto out_geo = dynamic_cast<mesh::CallMesh*>(&c);
    if (out_geo == nullptr)
        return false;
    auto in_mesh = in_mesh_slot_.CallAs<mesh::CallMesh>();
    if (in_mesh == nullptr)
        return false;
    auto in_paths = in_paths_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_paths == nullptr)
        return false;
    auto in_tf = in_tf_slot_.CallAs<core::view::CallGetTransferFunction>();
    if (in_tf == nullptr)
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
        if (!assertData(*in_mesh, *in_paths, *in_tf, meta.m_frame_ID))
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
    // out_geo->setData(in_mesh->getData(), in_mesh->version());
    out_geo->setData(mesh_access_collection_, out_data_hash_);

    return true;
}


bool megamol::optix_hpg::TransitionCalculator::get_extent_cb(core::Call& c) {
    auto out_geo = dynamic_cast<mesh::CallMesh*>(&c);
    if (out_geo == nullptr)
        return false;
    auto in_data = in_mesh_slot_.CallAs<mesh::CallMesh>();
    if (in_data == nullptr)
        return false;
    auto in_paths = in_paths_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_paths == nullptr)
        return false;

    auto out_meta = out_geo->getMetaData();
    auto in_meta = in_data->getMetaData();
    in_meta = out_meta;
    in_data->setMetaData(in_meta);
    if (!(*in_data)(1))
        return false;
    in_paths->SetFrameCount(in_meta.m_frame_cnt);
    in_paths->SetFrameID(in_meta.m_frame_ID);
    if (!(*in_paths)(1))
        return false;

    in_meta = in_data->getMetaData();
    out_meta = in_meta;
    out_geo->setMetaData(out_meta);

    //if ((*in_data)(1) && (*in_data)(0)) {
    //    auto const meta = in_data->getMetaData();
    //    out_geo->setMetaData(meta);
    //}

    return true;
}
