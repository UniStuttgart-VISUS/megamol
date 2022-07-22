/*
 * MeshSelector.cpp
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "MeshSelector.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FlexEnumParam.h"


namespace megamol {
namespace probe {

MeshSelector::MeshSelector()
        : Module()
        , _getDataSlot("getDataSlot", "")
        , _deployMeshSlot("deployMeshSlot", "")
        , _meshNumberSlot("meshNumber", "")
        , _splitMeshSlot("splitMesh", "") {

    this->_meshNumberSlot << new core::param::FlexEnumParam("0");
    this->_meshNumberSlot.SetUpdateCallback(&MeshSelector::parameterChanged);
    this->MakeSlotAvailable(&this->_meshNumberSlot);

    this->_splitMeshSlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->_splitMeshSlot);

    this->_deployMeshSlot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &MeshSelector::getData);
    this->_deployMeshSlot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &MeshSelector::getMetaData);
    this->MakeSlotAvailable(&this->_deployMeshSlot);

    this->_getDataSlot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->_getDataSlot);
}

MeshSelector::~MeshSelector() {
    this->Release();
}

bool MeshSelector::create() {
    return true;
}

void MeshSelector::release() {}

bool MeshSelector::InterfaceIsDirty() {
    return (this->_meshNumberSlot.IsDirty() || this->_splitMeshSlot.IsDirty());
}


void MeshSelector::connectedMesh(const uint32_t idx) {

    for (auto neighbor : _neighbor_map[idx]) {
        auto it = _face_to_mesh_map.find(neighbor);
        if (it == _face_to_mesh_map.end()) {
            _face_to_mesh_map[neighbor] = _current_mesh_id;
            this->connectedMesh(neighbor);
        }
    }
}


bool MeshSelector::splitMesh(const mesh::MeshDataAccessCollection::Mesh& mesh) {


    auto cd = this->_getDataSlot.CallAs<mesh::CallMesh>();
    if (cd == nullptr)
        return false;

    if (!_mu) {
        _mu = std::make_shared<MeshUtility>();
        _mu->inputData(mesh);
        _numFaces = _mu->getNumTotalFaces();
    }

#pragma omp parallel for
    for (int idx = 0; idx < _numFaces; idx++) {
        _neighbor_map[idx] = _mu->getNeighboringTriangles(idx, 1);
    }

    // still produces stack overflow
    //
    //_current_mesh_id = 0;
    //_face_to_mesh_map[0] = _current_mesh_id;
    // uint32_t face_to_mesh_map_size = _face_to_mesh_map.size();
    // uint32_t start_id = 0;
    // while (face_to_mesh_map_size < _numFaces) {
    //    this->connectedMesh(start_id);
    //    face_to_mesh_map_size = _face_to_mesh_map.size();
    //    _current_mesh_id++;
    //    auto it = _face_to_mesh_map.find(start_id);
    //    while (it != _face_to_mesh_map.end()) {
    //        start_id++;
    //        it = _face_to_mesh_map.find(start_id);
    //    }
    //}


    _current_mesh_id = 0;
    uint32_t start_id = 0;
    std::vector<uint32_t> mesh_size;

    while (_face_to_mesh_map.size() < _neighbor_map.size()) {

        _face_to_mesh_map[start_id] = _current_mesh_id;
        std::vector<uint32_t> current_neighbors = _neighbor_map[start_id];

        bool not_yet_all_neighbors_detected = true;
        while (not_yet_all_neighbors_detected) {
            std::vector<uint32_t> open_ends;
            for (auto neighbor : current_neighbors) {
                auto it = _face_to_mesh_map.find(neighbor);
                if (it == _face_to_mesh_map.end()) {
                    open_ends.emplace_back(neighbor);
                    _face_to_mesh_map[neighbor] = _current_mesh_id;
                    if (mesh_size.size() < _current_mesh_id + 1) {
                        mesh_size.emplace_back(1);
                    } else {
                        mesh_size[_current_mesh_id]++;
                    }
                }
            }
            if (open_ends.empty())
                not_yet_all_neighbors_detected = false;
            current_neighbors.clear();
            for (auto open_end : open_ends) {
                for (auto open_end_neighbor : _neighbor_map[open_end]) {
                    current_neighbors.emplace_back(open_end_neighbor);
                }
            }
        }

        bool new_entry_not_found = true;
        uint32_t i = 1;
        while (new_entry_not_found) {
            auto it = _face_to_mesh_map.find(i);
            if (it == _face_to_mesh_map.end()) {
                start_id = i;
                new_entry_not_found = false;
            }
            i++;
        }

        _current_mesh_id++;
    }

    _split_mesh = std::make_shared<mesh::MeshDataAccessCollection>();

    //_grouped_vertices.resize(_current_mesh_id + 1);
    //_new_indice_map.resize(_current_mesh_id + 1);

    // assert(_grouped_vertices.size() == mesh_size.size());

    // for (int i = 0; i < _grouped_vertices.size(); ++i) {
    //    _grouped_vertices[i].reserve(mesh_size[i] * 3);
    //}

    // auto mesh_ptr = cd->getData()->accessMesh().data();
    // auto index_ptr = reinterpret_cast<uint32_t*>(mesh_ptr->indices.data);
    // auto vertex_ptr = reinterpret_cast<float*>(mesh_ptr->attributes[0].data);

    // uint32_t idx = 0;
    // for (auto face : _face_to_mesh_map) {
    //    for (int j = 0; j < 3; ++j) {
    //        auto current_index = index_ptr[3 * face.first + j];

    //        auto it = _new_indice_map[face.second].find(current_index);
    //        if (it == _new_indice_map[face.second].end()) {
    //            _new_indice_map[face.second][idx] = current_index;
    //            _grouped_vertices[face.second].emplace_back(vertex_ptr[current_index]);
    //        }
    //    }
    //}

    // current_mesh_id gets additionally advanced in the end of the while loop
    _grouped_indices.resize(_current_mesh_id);
    auto index_ptr = reinterpret_cast<uint32_t*>(cd->getData()->accessMeshes().begin()->second.indices.data);
    for (int i = 0; i < _grouped_indices.size(); ++i) {
        _grouped_indices[i].reserve(mesh_size[i] * 3);
    }
    for (auto face : _face_to_mesh_map) {
        std::array<uint32_t, 3> triangle;
        for (int j = 0; j < 3; ++j) {
            triangle[j] = index_ptr[3 * face.first + j];
        }
        _grouped_indices[face.second].emplace_back(triangle);
    }

    return true;
}

bool MeshSelector::getData(core::Call& call) {

    bool something_changed = _recalc;

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr)
        return false;

    auto cd = this->_getDataSlot.CallAs<mesh::CallMesh>();
    if (cd == nullptr)
        return false;

    if (!(*cd)(0))
        return false;

    if (cd->hasUpdate()) {
        something_changed = true;
        auto data_source_meta_data = cd->getMetaData();

        auto mesh_meta_data = cm->getMetaData();
        mesh_meta_data.m_bboxs = data_source_meta_data.m_bboxs;
        _bbox = mesh_meta_data.m_bboxs;
        mesh_meta_data.m_frame_cnt = data_source_meta_data.m_frame_cnt;
        cm->setMetaData(mesh_meta_data);
    }

    auto const& first_mesh = cd->getData()->accessMeshes().begin()->second;

    if (something_changed || _recalc) {
        _meshNumberSlot.Param<core::param::FlexEnumParam>()->ClearValues();
        int const num_meshes = cd->getData()->accessMeshes().size();
        for (int i = 0; i < num_meshes; i++) {
            auto const current_mesh = std::next(cd->getData()->accessMeshes().begin(), i);
            for (auto& attr : current_mesh->second.attributes) {
                if (attr.semantic == mesh::MeshDataAccessCollection::POSITION) {
                    _meshNumberSlot.Param<core::param::FlexEnumParam>()->AddValue(std::to_string(i));
                }
            }
        }

        _selected_mesh = std::stoi(_meshNumberSlot.Param<core::param::FlexEnumParam>()->Value());
        auto const selected_mesh = std::next(cd->getData()->accessMeshes().begin(), _selected_mesh);

        if (_splitMeshSlot.Param<core::param::BoolParam>()->Value()) {
            if (_grouped_indices.empty() || cd->hasUpdate())
                this->splitMesh(selected_mesh->second);

            assert(_grouped_indices.size() > 0);

            for (int i = 0; i < _grouped_indices.size(); ++i) {
                _meshNumberSlot.Param<core::param::FlexEnumParam>()->AddValue(std::to_string(i));
            }
        } else {

            _mesh_attribs = selected_mesh->second.attributes;

            _mesh_indices.resize(1);
            _mesh_indices[0] = selected_mesh->second.indices;
        }

        ++_version;
    }

    // put data in mesh
    mesh::MeshDataAccessCollection out_mesh_collection;
    std::string identifier = std::string(FullName()) + "_mesh";

    if (_splitMeshSlot.Param<core::param::BoolParam>()->Value()) {
        assert(_selected_mesh < _grouped_indices.size());
        _mesh_indices.resize(_grouped_indices.size());
        _mesh_indices[0].type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
        _mesh_indices[0].byte_size = _grouped_indices[_selected_mesh].size() * sizeof(std::array<uint32_t, 3>);
        _mesh_indices[0].data = reinterpret_cast<uint8_t*>(_grouped_indices[_selected_mesh].data());
        auto const selected_mesh = std::next(cd->getData()->accessMeshes().begin(), _selected_mesh);
        out_mesh_collection.addMesh(
            identifier, selected_mesh->second.attributes, _mesh_indices[0], mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES);
        // debugging stuff
        // for (int i = 0; i < _grouped_indices.size(); ++i) {
        //    _mesh_indices[i].type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
        //    _mesh_indices[i].byte_size = _grouped_indices[i].size() * sizeof(std::array<uint32_t, 3>);
        //    _mesh_indices[i].data = reinterpret_cast<uint8_t*>(_grouped_indices[i].data());

        //    mesh.addMesh(_mesh_attribs, _mesh_indices[i],
        //    mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES);
        //}
    } else {
        out_mesh_collection.addMesh(
            identifier, _mesh_attribs, _mesh_indices[0], mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES);
    }
    cm->setData(std::make_shared<mesh::MeshDataAccessCollection>(std::move(out_mesh_collection)), _version);
    _recalc = false;

    return true;
}

bool MeshSelector::getMetaData(core::Call& call) {

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr)
        return false;

    auto cd = this->_getDataSlot.CallAs<mesh::CallMesh>();
    if (cd == nullptr)
        return false;

    auto meta_data = cm->getMetaData();
    auto data_source_meta_data = cd->getMetaData();

    // get metadata from adios
    data_source_meta_data.m_frame_ID = meta_data.m_frame_ID;
    cd->setMetaData(data_source_meta_data);

    if (!(*cd)(1))
        return false;

    data_source_meta_data = cd->getMetaData();

    // put metadata in mesh call
    meta_data.m_bboxs = data_source_meta_data.m_bboxs;
    meta_data.m_frame_cnt = data_source_meta_data.m_frame_cnt;
    cm->setMetaData(meta_data);

    return true;
}

bool MeshSelector::parameterChanged(core::param::ParamSlot& p) {

    _recalc = true;

    return true;
}

} // namespace probe
} // namespace megamol
