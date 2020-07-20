/*
 * ExtractCenterline.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ExtractCenterline.h"
#include <limits>
#include "CallKDTree.h"
#include "adios_plugin/CallADIOSData.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "normal_3d_omp.h"
#include <atomic>


namespace megamol {
namespace probe {


ExtractCenterline::ExtractCenterline()
    : Module()
    , _getDataCall("getData", "")
    , _deployLineCall("deployCenterline", "") {

    this->_deployLineCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &ExtractCenterline::getData);
    this->_deployLineCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &ExtractCenterline::getMetaData);
    this->MakeSlotAvailable(&this->_deployLineCall);


    this->_getDataCall.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->_getDataCall);
}

ExtractCenterline::~ExtractCenterline() { this->Release(); }

bool ExtractCenterline::create() { return true; }

void ExtractCenterline::release() {}

bool ExtractCenterline::InterfaceIsDirty() { return false; }

bool ExtractCenterline::extractCenterLine(float* vertices, uint32_t num_vertices, uint32_t num_components) {

    // We need a valid bounding box
    if (!this->_bbox.IsBoundingBoxValid()) {
        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::min();
        float min_y = std::numeric_limits<float>::max();
        float max_y = std::numeric_limits<float>::min();
        float min_z = std::numeric_limits<float>::max();
        float max_z = std::numeric_limits<float>::min();
        for (uint32_t i = 0; i < num_vertices; i++) {
            min_x = std::min(min_x, vertices[num_components * i + 0]);
            max_x = std::max(max_x, vertices[num_components * i + 0]);
            min_y = std::min(min_y, vertices[num_components * i + 1]);
            max_y = std::max(max_y, vertices[num_components * i + 1]);
            min_z = std::min(min_z, vertices[num_components * i + 2]);
            max_z = std::max(max_z, vertices[num_components * i + 2]);
        }
        this->_bbox.SetBoundingBox(min_x, min_y, max_z, max_x, max_y, min_z);
    }

    std::array<float, 3> whd = {this->_bbox.BoundingBox().Width(), this->_bbox.BoundingBox().Height(),
        this->_bbox.BoundingBox().Depth()};
    const auto longest_edge_index = std::distance(whd.begin(), std::max_element(whd.begin(), whd.end()));

    const uint32_t num_steps = 20;
    const auto step_size = whd[longest_edge_index] / (num_steps + 3); // without begin and end
    const auto step_epsilon = step_size / 2;
    float offset = 0.0f;
    if (longest_edge_index == 0) {
        offset = std::min(this->_bbox.BoundingBox().GetLeft(), this->_bbox.BoundingBox().GetRight());
    } else if (longest_edge_index == 1) {
        offset = std::min(this->_bbox.BoundingBox().GetBottom(), this->_bbox.BoundingBox().GetTop());
    } else if (longest_edge_index == 2) {
        offset = std::min(this->_bbox.BoundingBox().GetFront(), this->_bbox.BoundingBox().GetBack());
    }

    _cl_indices_per_slice.clear();
    _centerline.clear();
    _centerline.resize(num_steps);
    _cl_indices_per_slice.resize(num_steps);


    for (uint32_t i = 0; i < _centerline.size(); i++) {
        const auto slice = offset + (i + 2) * step_size;
        const auto slice_min = slice - step_epsilon;
        const auto slice_max = slice + step_epsilon;
        float slice_dim1_mean = 0.0f;
        float slice_dim2_mean = 0.0f;
        for (uint32_t n = 0; n < num_vertices; n++) {
            if (longest_edge_index == 0) {
                if (vertices[num_components * n + 0] >= slice_min && vertices[num_components * n + 0] < slice_max) {
                    _cl_indices_per_slice[i].emplace_back(n);
                    slice_dim1_mean += vertices[num_components * n + 1];
                    slice_dim2_mean += vertices[num_components * n + 2];
                    // std::array<float, 2> tmp_slice_point = {point_cloud.points[n].y, point_cloud.points[n].z};
                    // slice_vertices.emplace_back(tmp_slice_point);
                }
            } else if (longest_edge_index == 1) {
                if (vertices[num_components * n + 1] >= slice_min && vertices[num_components * n + 1] < slice_max) {
                    _cl_indices_per_slice[i].emplace_back(n);
                    slice_dim1_mean += vertices[num_components * n + 0];
                    slice_dim2_mean += vertices[num_components * n + 2];
                }
            } else if (longest_edge_index == 2) {
                if (vertices[num_components * n + 2] >= slice_min && vertices[num_components * n + 2] < slice_max) {
                    _cl_indices_per_slice[i].emplace_back(n);
                    slice_dim1_mean += vertices[num_components * n + 0];
                    slice_dim2_mean += vertices[num_components * n + 1];
                }
            }
        }
        slice_dim1_mean /= _cl_indices_per_slice[i].size();
        slice_dim2_mean /= _cl_indices_per_slice[i].size();
        if (longest_edge_index == 0) {
            _centerline[i] = {slice, slice_dim1_mean, slice_dim2_mean, 1.0f};
        } else if (longest_edge_index == 1) {
            _centerline[i] = {slice_dim1_mean, slice, slice_dim2_mean, 1.0f};
        } else if (longest_edge_index == 2) {
            _centerline[i] = {slice_dim1_mean, slice_dim2_mean, slice, 1.0f};
        }
    }

    return true;
}

bool ExtractCenterline::getMetaData(core::Call& call) {

    auto cl = dynamic_cast<mesh::CallMesh*>(&call);
    if (cl == nullptr) return false;

    auto cm = this->_getDataCall.CallAs<mesh::CallMesh>();
    if (cm == nullptr) return false;

    auto line_meta_data = cl->getMetaData();
    auto mesh_meta_data = cm->getMetaData();

    // get metadata from adios
    mesh_meta_data.m_frame_ID = line_meta_data.m_frame_ID;
    cm->setMetaData(mesh_meta_data);
    if (!(*cm)(1)) return false;
    mesh_meta_data = cm->getMetaData();

    // put metadata in line call
    _bbox = mesh_meta_data.m_bboxs;
    line_meta_data.m_bboxs = mesh_meta_data.m_bboxs;
    line_meta_data.m_frame_cnt = mesh_meta_data.m_frame_cnt;
    cl->setMetaData(line_meta_data);

    return true;
}

bool ExtractCenterline::getData(core::Call& call) {
    auto cl = dynamic_cast<mesh::CallMesh*>(&call);
    if (cl == nullptr) return false;

    auto cm = this->_getDataCall.CallAs<mesh::CallMesh>();
    if (cm == nullptr) return false;
    if (!(*cm)(0)) return false;

    auto meta_data = cm->getMetaData();
    

    if (cm->hasUpdate())
    {
        ++_version;

        auto data = cm->getData();

        if (data->accessMesh().size() > 1 || data->accessMesh().empty()) {
            vislib::sys::Log::DefaultLog.WriteError("[ExtractCenterline] Cannot handle mesh");
            return false;
        }

        float* vertices = nullptr;
        uint32_t num_vertices = 0;
        uint32_t num_components = 0;
        for (auto& attrib : data->accessMesh()[0].attributes) {
            if (attrib.semantic == mesh::MeshDataAccessCollection::POSITION) {
                if (attrib.component_type != mesh::MeshDataAccessCollection::FLOAT) {
                    vislib::sys::Log::DefaultLog.WriteError("[ExtractCenterline] Cannot handle data type");
                    return false;
                }
                vertices = reinterpret_cast<float*>(attrib.data);
                num_vertices =
                    attrib.byte_size / (mesh::MeshDataAccessCollection::getByteSize(attrib.component_type) * attrib.component_cnt);
                num_components = attrib.component_cnt;
            }
        }

        assert(vertices != nullptr || num_vertices != 0 || num_components != 0);
            
        this->extractCenterLine(vertices, num_vertices, num_components);

        _line_attribs.resize(1);
        _line_attribs[0].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
        _line_attribs[0].byte_size = _centerline.size() * sizeof(std::array<float, 4>);
        _line_attribs[0].component_cnt = 3;
        _line_attribs[0].stride = sizeof(std::array<float, 4>);
        _line_attribs[0].data = reinterpret_cast<uint8_t*>(_centerline.data());
        _line_attribs[0].semantic = mesh::MeshDataAccessCollection::POSITION;

        _cl_indices.resize(_centerline.size()-1);
        std::generate(_cl_indices.begin(), _cl_indices.end(), [n = 0]() mutable { return n++; });

        _line_indices.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
        _line_indices.byte_size = _cl_indices.size() * sizeof(uint32_t);
        _line_indices.data = reinterpret_cast<uint8_t*>(_cl_indices.data());
    }

    // put data in line
    mesh::MeshDataAccessCollection line;
    line.addMesh(_line_attribs, _line_indices, mesh::MeshDataAccessCollection::LINES);
    cl->setData(std::make_shared<mesh::MeshDataAccessCollection>(std::move(line)),_version);

    auto line_meta_data = cl->getMetaData();
    line_meta_data.m_bboxs = this->_bbox;
    cl->setMetaData(line_meta_data);

    return true;
}

} // namespace probe
} // namespace megamol
