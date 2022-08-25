/*
 * DualMeshProbeSampling.cpp
 * Copyright (C) 2022 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "DualMeshProbeSampling.h"
#include "mmadios/CallADIOSData.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "probe/CallKDTree.h"
#include "probe/ProbeCalls.h"


namespace megamol::probe {

DualMeshProbeSampling::DualMeshProbeSampling()
        : Module()
        , _version(0)
        , _old_datahash(0)
        , _probe_lhs_slot("deployProbe", "")
        , _probe_rhs_slot("getProbe", "")
        , _adios_rhs_slot("getData", "")
        , _full_tree_rhs_slot("getTree", "")
        , _parameter_to_sample_slot("ParameterToSample", "")
        , _num_samples_per_probe_slot(
              "NumSamplesPerProbe", "Note: Tighter sample placement leads to reduced sampling radius.")
        , _vec_param_to_samplex_x("ParameterToSampleX", "")
        , _vec_param_to_samplex_y("ParameterToSampleY", "")
        , _vec_param_to_samplex_z("ParameterToSampleZ", "")
        , _vec_param_to_samplex_w("ParameterToSampleW", "")
        , _mesh_rhs_slot("getMesh", "Get the surface mesh")
        , _debug_lhs_slot("getDMvertices", "") {

    _probe_lhs_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(0), &DualMeshProbeSampling::getData);
    _probe_lhs_slot.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(1), &DualMeshProbeSampling::getMetaData);
    MakeSlotAvailable(&_probe_lhs_slot);

    _debug_lhs_slot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(0), &DualMeshProbeSampling::getParticleData);
    _debug_lhs_slot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(1), &DualMeshProbeSampling::getParticleMetaData);
    MakeSlotAvailable(&_debug_lhs_slot);

    _probe_rhs_slot.SetCompatibleCall<CallProbesDescription>();
    MakeSlotAvailable(&_probe_rhs_slot);
    _probe_rhs_slot.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);

    _mesh_rhs_slot.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&_mesh_rhs_slot);
    _mesh_rhs_slot.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);

    _adios_rhs_slot.SetCompatibleCall<adios::CallADIOSDataDescription>();
    MakeSlotAvailable(&_adios_rhs_slot);
    _adios_rhs_slot.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);

    _full_tree_rhs_slot.SetCompatibleCall<CallKDTreeDescription>();
    MakeSlotAvailable(&_full_tree_rhs_slot);
    _full_tree_rhs_slot.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);

    core::param::FlexEnumParam* paramEnum = new core::param::FlexEnumParam("undef");
    _parameter_to_sample_slot << paramEnum;
    _parameter_to_sample_slot.SetUpdateCallback(&DualMeshProbeSampling::paramChanged);
    MakeSlotAvailable(&_parameter_to_sample_slot);

    _num_samples_per_probe_slot << new core::param::IntParam(10);
    _num_samples_per_probe_slot.SetUpdateCallback(&DualMeshProbeSampling::paramChanged);
    MakeSlotAvailable(&_num_samples_per_probe_slot);

    core::param::FlexEnumParam* paramEnum_1 = new core::param::FlexEnumParam("undef");
    _vec_param_to_samplex_x << paramEnum_1;
    _vec_param_to_samplex_x.SetUpdateCallback(&DualMeshProbeSampling::paramChanged);
    MakeSlotAvailable(&_vec_param_to_samplex_x);

    core::param::FlexEnumParam* paramEnum_2 = new core::param::FlexEnumParam("undef");
    _vec_param_to_samplex_y << paramEnum_2;
    _vec_param_to_samplex_y.SetUpdateCallback(&DualMeshProbeSampling::paramChanged);
    MakeSlotAvailable(&_vec_param_to_samplex_y);

    core::param::FlexEnumParam* paramEnum_3 = new core::param::FlexEnumParam("undef");
    _vec_param_to_samplex_z << paramEnum_3;
    _vec_param_to_samplex_z.SetUpdateCallback(&DualMeshProbeSampling::paramChanged);
    MakeSlotAvailable(&_vec_param_to_samplex_z);

    core::param::FlexEnumParam* paramEnum_4 = new core::param::FlexEnumParam("undef");
    _vec_param_to_samplex_w << paramEnum_4;
    _vec_param_to_samplex_w.SetUpdateCallback(&DualMeshProbeSampling::paramChanged);
    MakeSlotAvailable(&_vec_param_to_samplex_w);
}

DualMeshProbeSampling::~DualMeshProbeSampling() {
    Release();
}

bool DualMeshProbeSampling::create() {
    return true;
}

void DualMeshProbeSampling::release() {}

bool DualMeshProbeSampling::getData(core::Call& call) {

    bool something_has_changed = false;
    auto cp = dynamic_cast<CallProbes*>(&call);
    if (cp == nullptr)
        return false;

    // query adios data
    auto cd = _adios_rhs_slot.CallAs<adios::CallADIOSData>();
    auto ct = _full_tree_rhs_slot.CallAs<CallKDTree>();
    auto cprobes = _probe_rhs_slot.CallAs<CallProbes>();
    auto csm = _mesh_rhs_slot.CallAs<mesh::CallMesh>();


    std::vector<std::string> toInq;
    std::string var_str =
        std::string(_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->ValueString());

    std::string x_var_str =
        std::string(_vec_param_to_samplex_x.Param<core::param::FlexEnumParam>()->ValueString());
    std::string y_var_str =
        std::string(_vec_param_to_samplex_y.Param<core::param::FlexEnumParam>()->ValueString());
    std::string z_var_str =
        std::string(_vec_param_to_samplex_z.Param<core::param::FlexEnumParam>()->ValueString());
    std::string w_var_str =
        std::string(_vec_param_to_samplex_w.Param<core::param::FlexEnumParam>()->ValueString());

    core::Spatial3DMetaData meta_data = cp->getMetaData();
    core::Spatial3DMetaData tree_meta_data;
    core::Spatial3DMetaData probes_meta_data;

    // only for particle data
    if (cd != nullptr && ct != nullptr && csm != nullptr) {

        toInq.clear();
        if (var_str != "undef") {
            toInq.emplace_back(
                std::string(_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->ValueString()));
        } else {
            toInq.emplace_back(
                std::string(_vec_param_to_samplex_x.Param<core::param::FlexEnumParam>()->ValueString()));
            toInq.emplace_back(
                std::string(_vec_param_to_samplex_y.Param<core::param::FlexEnumParam>()->ValueString()));
            toInq.emplace_back(
                std::string(_vec_param_to_samplex_z.Param<core::param::FlexEnumParam>()->ValueString()));
            toInq.emplace_back(
                std::string(_vec_param_to_samplex_w.Param<core::param::FlexEnumParam>()->ValueString()));
        }

        // get data from adios
        for (auto var : toInq) {
            if (!cd->inquireVar(var))
                return false;
        }

        if (cd->getDataHash() != _old_datahash || _trigger_recalc) {
            if (!(*cd)(0))
                return false;
        }

        // query kd tree data
        if (!(*ct)(0))
            return false;

        // query dual mesh
        if (!(*csm)(0))
            return false;


        tree_meta_data = ct->getMetaData();

        something_has_changed = something_has_changed || (cd->getDataHash() != _old_datahash) || ct->hasUpdate();
    } else {
        return false;
    }

    // query probe data
    if (cprobes == nullptr)
        return false;
    if (!(*cprobes)(0))
        return false;

    something_has_changed = something_has_changed || cprobes->hasUpdate() || _trigger_recalc;

    probes_meta_data = cprobes->getMetaData();
    _probes = cprobes->getData();

    if (something_has_changed) {
        ++_version;

        auto surface_mesh = csm->getData();
        if (!calcDualMesh(surface_mesh)) {
            log.WriteError("[DualMeshProbeSampling] Error during dual mesh calculation.");
            return false;
        }

        if (var_str != "undef") {
            if (cd == nullptr || ct == nullptr) {
                log.WriteError(
                    "[DualMeshProbeSampling] Scalar mode selected but no particle data connected.");
                return false;
            }
            // scalar sampling
            auto tree = ct->getData();
            if (cd->getData(var_str)->getType() == "double") {
                std::vector<double> data = cd->getData(var_str)->GetAsDouble();
                doScalarDistributionSampling(tree, data);
            } else if (cd->getData(var_str)->getType() == "float") {
                std::vector<float> data = cd->getData(var_str)->GetAsFloat();
                doScalarDistributionSampling(tree, data);
            }
        } else {
            if (cd == nullptr || ct == nullptr) {
                log.WriteError(
                    "[DualMeshProbeSampling] Vector mode selected but no particle data connected.");
                return false;
            }
            // vector sampling
            auto tree = ct->getData();
            if (cd->getData(x_var_str)->getType() == "double" && cd->getData(y_var_str)->getType() == "double" &&
                cd->getData(z_var_str)->getType() == "double" && cd->getData(w_var_str)->getType() == "double") {
                std::vector<double> data_x = cd->getData(x_var_str)->GetAsDouble();
                std::vector<double> data_y = cd->getData(y_var_str)->GetAsDouble();
                std::vector<double> data_z = cd->getData(z_var_str)->GetAsDouble();
                std::vector<double> data_w = cd->getData(w_var_str)->GetAsDouble();

                doVectorSamling(tree, data_x, data_y, data_z, data_w);

            } else if (cd->getData(x_var_str)->getType() == "float" && cd->getData(y_var_str)->getType() == "float" &&
                       cd->getData(z_var_str)->getType() == "float" && cd->getData(w_var_str)->getType() == "float") {
                std::vector<float> data_x = cd->getData(x_var_str)->GetAsFloat();
                std::vector<float> data_y = cd->getData(y_var_str)->GetAsFloat();
                std::vector<float> data_z = cd->getData(z_var_str)->GetAsFloat();
                std::vector<float> data_w = cd->getData(w_var_str)->GetAsFloat();

                doVectorSamling(tree, data_x, data_y, data_z, data_w);

            }
        }
    }

    // put data into probes

    if (cd != nullptr) {
        _old_datahash = cd->getDataHash();
        meta_data.m_bboxs = tree_meta_data.m_bboxs;
    }

    cp->setMetaData(meta_data);
    cp->setData(_probes, _version);
    _trigger_recalc = false;


    return true;
}

bool DualMeshProbeSampling::getMetaData(core::Call& call) {

    auto cp = dynamic_cast<CallProbes*>(&call);
    if (cp == nullptr)
        return false;

    auto cd = _adios_rhs_slot.CallAs<adios::CallADIOSData>();
    auto ct = _full_tree_rhs_slot.CallAs<CallKDTree>();
    auto cprobes = _probe_rhs_slot.CallAs<CallProbes>();
    if (cprobes == nullptr) {
        log.WriteError("[DualMeshProbeSampling] Connecting probes is required.");
        return false;
    }

    auto cdm = _mesh_rhs_slot.CallAs<mesh::CallMesh>();
    if (cdm == nullptr) {
        log.WriteError("[DualMeshProbeSampling] Connecting a dual mesh is required.");
        return false;
    }

    auto meta_data = cp->getMetaData();
    // if (cd->getDataHash() == _old_datahash && meta_data.m_frame_ID == cd->getFrameIDtoLoad() && !_trigger_recalc)
    //    return true;

    if (cd != nullptr && ct != nullptr && cdm != nullptr) {
        cd->setFrameIDtoLoad(meta_data.m_frame_ID);
        auto dm_meta_data = cdm->getMetaData();
        dm_meta_data.m_frame_ID = meta_data.m_frame_ID;
        cdm->setMetaData(dm_meta_data);
        if (!(*cd)(1))
            return false;
        if (!(*ct)(1))
            return false;
        if (!(*cdm)(1))
            return false;
        meta_data.m_frame_cnt = cd->getFrameCount();

        // get adios meta data
        auto vars = cd->getAvailableVars();
        for (auto var : vars) {
            _parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->AddValue(var);
            _vec_param_to_samplex_x.Param<core::param::FlexEnumParam>()->AddValue(var);
            _vec_param_to_samplex_y.Param<core::param::FlexEnumParam>()->AddValue(var);
            _vec_param_to_samplex_z.Param<core::param::FlexEnumParam>()->AddValue(var);
            _vec_param_to_samplex_w.Param<core::param::FlexEnumParam>()->AddValue(var);
        }
    } else {
        log.WriteError("[DualMeshProbeSampling] Connecting adios data and kd tree modules is required.");
        return false;
    }

    auto probes_meta_data = cprobes->getMetaData();
    probes_meta_data.m_frame_ID = meta_data.m_frame_ID;
    cprobes->setMetaData(probes_meta_data);
    if (!(*cprobes)(1))
        return false;

    // put metadata in mesh call
    cp->setMetaData(meta_data);

    return true;
}

bool DualMeshProbeSampling::getParticleMetaData(core::Call& call) {

    auto cp = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
    if (cp == nullptr)
        return false;

    auto cd = _adios_rhs_slot.CallAs<adios::CallADIOSData>();
    auto ct = _full_tree_rhs_slot.CallAs<CallKDTree>();
    auto cprobes = _probe_rhs_slot.CallAs<CallProbes>();
    if (cprobes == nullptr) {
        log.WriteError("[DualMeshProbeSampling] Connecting probes is required.");
        return false;
    }

    auto cdm = _mesh_rhs_slot.CallAs<mesh::CallMesh>();
    if (cdm == nullptr) {
        log.WriteError("[DualMeshProbeSampling] Connecting a dual mesh is required.");
        return false;
    }

    auto current_frame = cp->FrameID();
    // if (cd->getDataHash() == _old_datahash && meta_data.m_frame_ID == cd->getFrameIDtoLoad() && !_trigger_recalc)
    //    return true;

    if (cd != nullptr && ct != nullptr && cdm != nullptr) {
        cd->setFrameIDtoLoad(current_frame);
        auto dm_meta_data = cdm->getMetaData();
        dm_meta_data.m_frame_ID = current_frame;
        cdm->setMetaData(dm_meta_data);
        if (!(*cd)(1))
            return false;
        if (!(*ct)(1))
            return false;
        if (!(*cdm)(1))
            return false;
        cp->SetFrameCount(cd->getFrameCount());

        // get adios meta data
        auto vars = cd->getAvailableVars();
        for (auto var : vars) {
            _parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->AddValue(var);
            _vec_param_to_samplex_x.Param<core::param::FlexEnumParam>()->AddValue(var);
            _vec_param_to_samplex_y.Param<core::param::FlexEnumParam>()->AddValue(var);
            _vec_param_to_samplex_z.Param<core::param::FlexEnumParam>()->AddValue(var);
            _vec_param_to_samplex_w.Param<core::param::FlexEnumParam>()->AddValue(var);
        }
    } else {
        log.WriteError(
            "[DualMeshProbeSampling] Connecting adios data and kd tree modules is required.");
        return false;
    }

    auto probes_meta_data = cprobes->getMetaData();
    probes_meta_data.m_frame_ID = current_frame;
    cprobes->setMetaData(probes_meta_data);
    if (!(*cprobes)(1))
        return false;

    return true;
}

bool DualMeshProbeSampling::getParticleData(core::Call& call) {

    bool something_has_changed = false;
    auto cp = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
    if (cp == nullptr)
        return false;

    // query adios data
    auto cd = _adios_rhs_slot.CallAs<adios::CallADIOSData>();
    auto ct = _full_tree_rhs_slot.CallAs<CallKDTree>();
    auto cprobes = _probe_rhs_slot.CallAs<CallProbes>();
    auto csm = _mesh_rhs_slot.CallAs<mesh::CallMesh>();


    std::vector<std::string> toInq;
    std::string var_str = std::string(_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->ValueString());

    std::string x_var_str = std::string(_vec_param_to_samplex_x.Param<core::param::FlexEnumParam>()->ValueString());
    std::string y_var_str = std::string(_vec_param_to_samplex_y.Param<core::param::FlexEnumParam>()->ValueString());
    std::string z_var_str = std::string(_vec_param_to_samplex_z.Param<core::param::FlexEnumParam>()->ValueString());
    std::string w_var_str = std::string(_vec_param_to_samplex_w.Param<core::param::FlexEnumParam>()->ValueString());

    core::Spatial3DMetaData tree_meta_data;
    core::Spatial3DMetaData probes_meta_data;

    // only for particle data
    if (cd != nullptr && ct != nullptr && csm != nullptr) {

        toInq.clear();
        if (var_str != "undef") {
            toInq.emplace_back(
                std::string(_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->ValueString()));
        } else {
            toInq.emplace_back(std::string(_vec_param_to_samplex_x.Param<core::param::FlexEnumParam>()->ValueString()));
            toInq.emplace_back(std::string(_vec_param_to_samplex_y.Param<core::param::FlexEnumParam>()->ValueString()));
            toInq.emplace_back(std::string(_vec_param_to_samplex_z.Param<core::param::FlexEnumParam>()->ValueString()));
            toInq.emplace_back(std::string(_vec_param_to_samplex_w.Param<core::param::FlexEnumParam>()->ValueString()));
        }

        // get data from adios
        for (auto var : toInq) {
            if (!cd->inquireVar(var))
                return false;
        }

        if (cd->getDataHash() != _old_datahash || _trigger_recalc) {
            if (!(*cd)(0))
                return false;
        }

        // query kd tree data
        if (!(*ct)(0))
            return false;

        // query dual mesh
        if (!(*csm)(0))
            return false;


        tree_meta_data = ct->getMetaData();

        something_has_changed = something_has_changed || (cd->getDataHash() != _old_datahash) || ct->hasUpdate();
    } else {
        return false;
    }

    // query probe data
    if (cprobes == nullptr)
        return false;
    if (!(*cprobes)(0))
        return false;

    something_has_changed = something_has_changed || cprobes->hasUpdate() || _trigger_recalc;

    probes_meta_data = cprobes->getMetaData();
    _probes = cprobes->getData();

    if (something_has_changed) {
        ++_version;

        auto surface_mesh = csm->getData();
        calcDualMesh(surface_mesh);
    }

    // put data into probes

    if (cd != nullptr) {
        _old_datahash = cd->getDataHash();
        cp->AccessBoundingBoxes().SetObjectSpaceBBox(tree_meta_data.m_bboxs.BoundingBox());
    }

    cp->SetParticleListCount(_dual_mesh_vertices.size());
    cp->SetDataHash(_version);
    for (int i = 0; i < _dual_mesh_vertices.size(); i++) {
       cp->AccessParticles(i).SetCount(_dual_mesh_vertices[i].size());
       cp->AccessParticles(i).SetVertexData(
          geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, _dual_mesh_vertices[i].data(), sizeof(std::array<float,3>));
    }

    _trigger_recalc = false;


    return true;
}

bool DualMeshProbeSampling::paramChanged(core::param::ParamSlot& p) {

    _trigger_recalc = true;
    return true;
}

bool DualMeshProbeSampling::createMeshTree(std::shared_ptr<mesh::MeshDataAccessCollection> mesh) {

    if (mesh->accessMeshes().size() > 1) {
        log.WriteError(
            "[DualMeshProbeSampling] Dual Mesh accessor too large. Please use a MeshSelector in front.");
        return false;
    }

    auto the_mesh = mesh->accessMeshes().begin();

    _mesh_vertex_data.clear();
    for (auto& attr : the_mesh->second.attributes) {
        if (attr.semantic == mesh::MeshDataAccessCollection::POSITION) {
            int const count = attr.byte_size / attr.stride;
            _mesh_vertex_data.resize(count);
            auto data = reinterpret_cast<float*>(attr.data);
            for (int i = 0; i < count; ++i) {
                std::array<float, 3> const vertex = {data[3 * i + 0], data[3 * i + 1], data[3 * i + 2]};
                _mesh_vertex_data[i] = vertex;
            }
        }
    }


    _mesh_dataKD = std::make_shared<const data2KD>(_mesh_vertex_data);
    _mesh_tree = std::make_shared<my_kd_tree_t>(
        3 /*dim*/, *_mesh_dataKD, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    _mesh_tree->buildIndex();
    return true;
}

bool DualMeshProbeSampling::calcDualMesh(std::shared_ptr<mesh::MeshDataAccessCollection> mesh) {

    auto cprobes = _probe_rhs_slot.CallAs<CallProbes>();
    auto const num_probes = cprobes->getData()->getProbeCount();


    // should only have one mesh
    if (mesh->accessMeshes().size() != 1) {
        log.WriteError(
            "[DualMeshProbeSampling] One mesh expected, got %d", mesh->accessMeshes().size());
        return false;
    }
    auto const current_mesh = mesh->accessMeshes().begin();


    mesh::MeshDataAccessCollection::VertexAttribute pos_attr;
    for (auto& attr : current_mesh->second.attributes) {
        if (attr.semantic == mesh::MeshDataAccessCollection::POSITION) {
            pos_attr = attr;
        }
    }

    size_t float_count = float_count = (pos_attr.byte_size / pos_attr.stride);
    size_t vertex_count = float_count / pos_attr.component_cnt;
    auto vertex_accessor = reinterpret_cast<float*>(pos_attr.data);


    size_t num_indices = current_mesh->second.indices.byte_size /
                         mesh::MeshDataAccessCollection::getByteSize(current_mesh->second.indices.type);
    std::vector<unsigned int> mesh_indices(reinterpret_cast<unsigned int*>(current_mesh->second.indices.data),
        reinterpret_cast<unsigned int*>(current_mesh->second.indices.data) + num_indices);
    assert(*std::max_element(mesh_indices.begin(),mesh_indices.end()) >= vertex_count);

    _dual_mesh_vertices.resize(num_probes);
    for (int i = 0; i < num_probes; ++i) {
        auto const current_probe = cprobes->getData()->getProbe<BaseProbe>(i);
        // should only have one ID
        if (current_probe.m_vert_ids.size() != 1) {
            log.WriteError(
                "[DualMeshProbeSampling] One vertex id expected, got %d", current_probe.m_vert_ids.size());
            return false;
        }

        int const current_vert_id = current_probe.m_vert_ids[0];

        std::vector<unsigned int> triangles;
        triangles.reserve(5);
        auto b = mesh_indices.begin(), end = mesh_indices.end();
        while (b != end) {
            b = std::find(b, end, current_vert_id);
            if (b != end) {
                unsigned int const diff = b++ - mesh_indices.begin();
                unsigned int const mod_result = diff % 3;
                triangles.emplace_back(diff - mod_result);
            }
        }
        // calc centers of triangles
        std::vector<std::array<float,3>> unsorted_dual_mesh_vertices;
        std::vector<float> dual_mesh_angles;
        for (auto const& triangle : triangles) {
            auto const t0 = 3 * mesh_indices[triangle];
            auto const t1 = 3 * mesh_indices[triangle+1];
            auto const t2 = 3 * mesh_indices[triangle+2];
            glm::vec3 const v0 = {vertex_accessor[t0 + 0],
                vertex_accessor[t0 + 1],
                vertex_accessor[t0 + 2]};
            glm::vec3 const v1 = {vertex_accessor[t1 + 0],
                vertex_accessor[t1 + 1],
                vertex_accessor[t1 + 2]};
            glm::vec3 const v2 = {vertex_accessor[t2 + 0],
                vertex_accessor[t2 + 1],
                vertex_accessor[t2 + 2]};
            glm::vec3 const center = (v0 + v1 + v2)/3.0f;
            unsorted_dual_mesh_vertices
                .emplace_back(std::array<float, 3>{center.x, center.y, center.z});
            if (unsorted_dual_mesh_vertices.size() > 1) {
                auto to_dm0 = to_vec3(unsorted_dual_mesh_vertices[0]) - to_vec3(current_probe.m_position);
                auto to_dm1 = to_vec3(unsorted_dual_mesh_vertices[1]) - to_vec3(current_probe.m_position);
                auto to_dmx = to_vec3(unsorted_dual_mesh_vertices.back()) - to_vec3(current_probe.m_position);
                auto face_n = -1.0f*to_vec3(current_probe.m_direction);
                auto n = glm::normalize(glm::cross(to_dm0, to_dmx));
                auto tangent = glm::normalize(glm::cross(face_n, to_dm0));

                auto angle = std::atan2(glm::dot(n, glm::cross(to_dm0, to_dmx)),
                    glm::dot(to_dm0, to_dmx));
                if ( glm::dot(to_dmx, tangent) < 0.0f) {
                    angle = 2.0f*3.1415926535f - angle;
                }
                dual_mesh_angles.emplace_back(angle);
            }
        }

        std::vector<int> index_permutation(dual_mesh_angles.size(),0);
        for (int j = 0; j < index_permutation.size(); j++) {
            index_permutation[j] = j;
        }
        std::sort(index_permutation.begin(), index_permutation.end(),
            [&](const int& a, const int& b) { return (dual_mesh_angles[a] < dual_mesh_angles[b]); });

        _dual_mesh_vertices[i].resize(unsorted_dual_mesh_vertices.size());
        _dual_mesh_vertices[i][0] = unsorted_dual_mesh_vertices[0];
        for (int k = 0; k < index_permutation.size(); k++) {
            _dual_mesh_vertices[i][k + 1] = unsorted_dual_mesh_vertices[index_permutation[k] + 1];
        }

        //_dual_mesh_vertices[i] = unsorted_dual_mesh_vertices;

    }
    
    return true;
}

} // namespace megamol::probe
