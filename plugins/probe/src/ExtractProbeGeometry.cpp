/*
 * ExtractProbeGeometry.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ExtractProbeGeometry.h"

#include "mesh/MeshCalls.h"
#include "probe/ProbeCalls.h"

megamol::probe::ExtractProbeGeometry::ExtractProbeGeometry()
        : Module()
        , _version(0)
        , _line(nullptr)
        , m_mesh_slot("deployMesh", "")
        , m_probe_slot("getProbes", "") {

    this->m_mesh_slot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &ExtractProbeGeometry::getData);
    this->m_mesh_slot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &ExtractProbeGeometry::getMetaData);
    this->MakeSlotAvailable(&this->m_mesh_slot);

    this->m_probe_slot.SetCompatibleCall<CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probe_slot);
}

megamol::probe::ExtractProbeGeometry::~ExtractProbeGeometry() {
    this->Release();
}

bool megamol::probe::ExtractProbeGeometry::create() {
    return true;
}

void megamol::probe::ExtractProbeGeometry::release() {}

std::shared_ptr<megamol::mesh::MeshDataAccessCollection> megamol::probe::ExtractProbeGeometry::convertToLine(
    core::Call& call) {

    auto* cm = dynamic_cast<mesh::CallMesh*>(&call);
    std::shared_ptr<mesh::MeshDataAccessCollection> line = std::make_shared<mesh::MeshDataAccessCollection>();

    auto probe_count = this->_probes->getProbeCount();

    _index_data = {0};
    _line_indices.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
    _line_indices.byte_size = _index_data.size() * sizeof(uint32_t);
    _line_indices.data = reinterpret_cast<uint8_t*>(_index_data.data());

    _line_attribs.clear();
    _vertex_data.clear();

    _line_attribs.resize(probe_count);
    _vertex_data.resize(2 * probe_count);

    //#pragma omp parallel for
    for (auto i = 0; i < probe_count; i++) {
        auto generic_probe = _probes->getGenericProbe(i);

        std::array<float, 3> direction;
        std::array<float, 3> position;
        float begin;
        float end;

        auto visitor = [&direction, &position, &begin, &end](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, probe::FloatProbe>) {
                direction = arg.m_direction;
                position = arg.m_position;
                begin = arg.m_begin;
                end = arg.m_end;
            } else if constexpr (std::is_same_v<T, probe::IntProbe>) {
                direction = arg.m_direction;
                position = arg.m_position;
                begin = arg.m_begin;
                end = arg.m_end;
            } else if constexpr (std::is_same_v<T, probe::Vec4Probe>) {
                direction = arg.m_direction;
                position = arg.m_position;
                begin = arg.m_begin;
                end = arg.m_end;
            } else if constexpr (std::is_same_v<T, probe::FloatDistributionProbe>) {
                direction = arg.m_direction;
                position = arg.m_position;
                begin = arg.m_begin;
                end = arg.m_end;
            } else if constexpr (std::is_same_v<T, probe::BaseProbe>) {
                direction = arg.m_direction;
                position = arg.m_position;
                begin = arg.m_begin;
                end = arg.m_end;
            } else {
                // unknown probe type, throw error? do nothing?
            }
        };

        std::visit(visitor, generic_probe);

        std::array<float, 3> vert1, vert2;

        vert1[0] = position[0] + direction[0] * begin;
        vert1[1] = position[1] + direction[1] * begin;
        vert1[2] = position[2] + direction[2] * begin;

        vert2[0] = position[0] + direction[0] * end;
        vert2[1] = position[1] + direction[1] * end;
        vert2[2] = position[2] + direction[2] * end;

        _vertex_data[2 * i + 0] = vert1;
        _vertex_data[2 * i + 1] = vert2;

        _line_attribs[i].resize(1);
        _line_attribs[i][0].semantic = mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION;
        _line_attribs[i][0].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
        _line_attribs[i][0].byte_size = 2 * sizeof(std::array<float, 3>);
        _line_attribs[i][0].component_cnt = 3;
        _line_attribs[i][0].stride = 3 * sizeof(float);
        _line_attribs[i][0].data = reinterpret_cast<uint8_t*>(_vertex_data[2 * i + 0].data());

        // put data in line
        std::string identifier = std::string(FullName()) + "_pl_" + std::to_string(i);
        line->addMesh(identifier, _line_attribs[i], _line_indices, mesh::MeshDataAccessCollection::LINE_STRIP);
    }

    _line = line;
    return line;
}

bool megamol::probe::ExtractProbeGeometry::getData(core::Call& call) {

    auto* cm = dynamic_cast<mesh::CallMesh*>(&call);
    auto* cp = this->m_probe_slot.CallAs<CallProbes>();

    if (cp == nullptr)
        return false;
    if (!(*cp)(0))
        return false;

    auto mesh_meta_data = cm->getMetaData();
    auto probe_meta_data = cp->getMetaData();

    if (cp->hasUpdate()) {
        ++_version;
        _probes = cp->getData();

        // here something really happens
        this->convertToLine(call);
    }

    if (cm->version() < _version) {
        mesh_meta_data.m_bboxs = probe_meta_data.m_bboxs;
        cm->setMetaData(mesh_meta_data);
        cm->setData(_line, _version);
    }

    return true;
}

bool megamol::probe::ExtractProbeGeometry::getMetaData(core::Call& call) {

    auto* cm = dynamic_cast<mesh::CallMesh*>(&call);
    auto* cp = this->m_probe_slot.CallAs<CallProbes>();

    if (cp == nullptr)
        return false;

    // set frame id before callback
    auto mesh_meta_data = cm->getMetaData();
    auto probe_meta_data = cp->getMetaData();

    probe_meta_data.m_frame_ID = mesh_meta_data.m_frame_ID;
    cp->setMetaData(probe_meta_data);

    if (!(*cp)(1))
        return false;

    probe_meta_data = cp->getMetaData();
    mesh_meta_data.m_frame_cnt = probe_meta_data.m_frame_cnt;
    mesh_meta_data.m_bboxs = probe_meta_data.m_bboxs; // normally not available here

    cm->setMetaData(mesh_meta_data);

    return true;
}
