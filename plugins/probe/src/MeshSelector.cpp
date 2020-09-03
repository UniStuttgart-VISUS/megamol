/*
 * MeshSelector.cpp
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "MeshSelector.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/FlexEnumParam.h"


namespace megamol {
namespace probe {

MeshSelector::MeshSelector()
    : Module()
    , _getDataSlot("getDataSlot", "")
    , _deployMeshSlot("deployMeshSlot", "")
    , _meshNumberSlot("meshNumber", "") {

    this->_meshNumberSlot << new core::param::FlexEnumParam("0");
    this->_meshNumberSlot.SetUpdateCallback(&MeshSelector::parameterChanged);
    this->MakeSlotAvailable(&this->_meshNumberSlot);

    this->_deployMeshSlot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &MeshSelector::getData);
    this->_deployMeshSlot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &MeshSelector::getMetaData);
    this->MakeSlotAvailable(&this->_deployMeshSlot);

    this->_getDataSlot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->_getDataSlot);
}

MeshSelector::~MeshSelector() { this->Release(); }

bool MeshSelector::create() { return true; }

void MeshSelector::release() {}

bool MeshSelector::InterfaceIsDirty() { return this->_meshNumberSlot.IsDirty(); }

bool MeshSelector::getData(core::Call& call) {

    bool something_changed = _recalc;

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataSlot.CallAs<mesh::CallMesh>();
    if (cd == nullptr) return false;

    if (!(*cd)(0)) return false;

    if (cd->hasUpdate()) {
        something_changed = true;
        auto data_source_meta_data = cd->getMetaData();

        auto mesh_meta_data = cm->getMetaData();
        mesh_meta_data.m_bboxs = data_source_meta_data.m_bboxs;
        _bbox = mesh_meta_data.m_bboxs;
        mesh_meta_data.m_frame_cnt = data_source_meta_data.m_frame_cnt;
        cm->setMetaData(mesh_meta_data);
    }

    auto mesh_ptr = cd->getData()->accessMesh().data();




    if (something_changed || _recalc) {
        _meshNumberSlot.Param<core::param::FlexEnumParam>()->ClearValues();
        for (int i = 0; i < mesh_ptr->attributes.size(); i++) {
            _meshNumberSlot.Param<core::param::FlexEnumParam>()->AddValue(std::to_string(i));
        }
        int selected_mesh = std::stoi(_meshNumberSlot.Param<core::param::FlexEnumParam>()->Value());

        _mesh_attribs.resize(1);
        _mesh_attribs[0] = mesh_ptr->attributes[0];

        _mesh_indices = mesh_ptr->indices;

        ++_version;
    }

    // put data in mesh
    mesh::MeshDataAccessCollection mesh;

    mesh.addMesh(_mesh_attribs, _mesh_indices, mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES);
    cm->setData(std::make_shared<mesh::MeshDataAccessCollection>(std::move(mesh)), _version);
    _recalc = false;

    return true;
}

bool MeshSelector::getMetaData(core::Call& call) {

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataSlot.CallAs<mesh::CallMesh>();
    if (cd == nullptr) return false;

    auto meta_data = cm->getMetaData();
    auto data_source_meta_data = cd->getMetaData();

    // get metadata from adios
    data_source_meta_data.m_frame_ID = meta_data.m_frame_ID;
    cd->setMetaData(data_source_meta_data);

    if (!(*cd)(1)) return false;

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
