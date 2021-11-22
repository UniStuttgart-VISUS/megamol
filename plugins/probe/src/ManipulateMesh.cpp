/*
 * ManipulateMesh.cpp
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ManipulateMesh.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/param/IntParam.h"


namespace megamol {
namespace probe {

ManipulateMesh::ManipulateMesh()
        : Module()
        , _getDataSlot("getData", "")
        , _deployMeshSlot("deployMesh", "")
        , _deployNormalsSlot("deployNormals", "")
        , _numFacesSlot("NumFaces", "")
        , _pointsDebugSlot("pointsDebug", "") {

    this->_numFacesSlot << new core::param::IntParam(1000);
    this->_numFacesSlot.SetUpdateCallback(&ManipulateMesh::parameterChanged);
    this->MakeSlotAvailable(&this->_numFacesSlot);

    this->_deployMeshSlot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &ManipulateMesh::getData);
    this->_deployMeshSlot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &ManipulateMesh::getMetaData);
    this->MakeSlotAvailable(&this->_deployMeshSlot);

    this->_pointsDebugSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(0), &ManipulateMesh::getParticleData);
    this->_pointsDebugSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(1), &ManipulateMesh::getParticleMetaData);
    this->MakeSlotAvailable(&this->_pointsDebugSlot);

    this->_getDataSlot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->_getDataSlot);
}

ManipulateMesh::~ManipulateMesh() {
    this->Release();
}

bool ManipulateMesh::create() {
    return true;
}

void ManipulateMesh::release() {}

bool ManipulateMesh::InterfaceIsDirty() {
    return this->_numFacesSlot.IsDirty();
}

bool ManipulateMesh::performMeshOperation(const mesh::MeshDataAccessCollection::Mesh& mesh) {


    //_mu->fillMeshFaces(patch_indices, this->_mesh_faces);
    //_mu->fillMeshVertices(patch_vertices, this->_mesh_vertices);


    // Decimate
    // Eigen::MatrixXd new_vertices;
    // Eigen::MatrixXi new_faces;
    // Eigen::VectorXi J;
    // igl::decimate(_vertices, _faces, this->_numFacesSlot.Param<core::param::IntParam>()->Value(), new_vertices,
    // new_faces, J);
    //_vertices = new_vertices;
    //_faces = new_faces;

    return true;
}

bool ManipulateMesh::convertToMesh() {

    _mesh_attribs.resize(1);
    _mesh_attribs[0].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
    _mesh_attribs[0].byte_size = _mesh_vertices.size() * sizeof(float);
    _mesh_attribs[0].component_cnt = 3;
    _mesh_attribs[0].stride = 3 * sizeof(float);
    _mesh_attribs[0].offset = 0;
    _mesh_attribs[0].data = reinterpret_cast<uint8_t*>(_mesh_vertices.data());
    _mesh_attribs[0].semantic = mesh::MeshDataAccessCollection::POSITION;

    //_mesh_attribs[1].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
    //_mesh_attribs[1].byte_size = _normals.size() * sizeof(std::array<float, 3>);
    //_mesh_attribs[1].component_cnt = 3;
    //_mesh_attribs[1].stride = sizeof(std::array<float, 3>);
    //_mesh_attribs[1].offset = 0;
    //_mesh_attribs[1].data = reinterpret_cast<uint8_t*>(_normals.data());
    //_mesh_attribs[1].semantic = mesh::MeshDataAccessCollection::NORMAL;

    _mesh_indices.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
    _mesh_indices.byte_size = _mesh_faces.size() * sizeof(uint32_t);
    _mesh_indices.data = reinterpret_cast<uint8_t*>(_mesh_faces.data());

    return true;
}


bool ManipulateMesh::getData(core::Call& call) {

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

    // auto mesh_ptr = cd->getData()->accessMesh().data();
    if (something_changed || _recalc) {
        // this->performMeshOperation(*mesh_ptr);
        this->convertToMesh();
        ++_version;
    }

    // put data in mesh
    mesh::MeshDataAccessCollection mesh;

    std::string identifier = std::string(FullName());
    mesh.addMesh(identifier, _mesh_attribs, _mesh_indices, mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES);
    cm->setData(std::make_shared<mesh::MeshDataAccessCollection>(std::move(mesh)), _version);
    _recalc = false;

    return true;
}

bool ManipulateMesh::getMetaData(core::Call& call) {

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

bool ManipulateMesh::getParticleMetaData(core::Call& call) {

    auto cd = this->_getDataSlot.CallAs<mesh::CallMesh>();
    if (cd == nullptr)
        return false;

    auto cpd = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
    if (cpd == nullptr)
        return false;

    bool something_changed = _recalc;

    if (!(*cd)(0))
        return false;
    if (cd->hasUpdate()) {
        something_changed = true;
        auto data_source_meta_data = cd->getMetaData();

        cpd->AccessBoundingBoxes().SetObjectSpaceBBox(data_source_meta_data.m_bboxs.BoundingBox());
        cpd->AccessBoundingBoxes().SetObjectSpaceClipBox(data_source_meta_data.m_bboxs.ClipBox());
        _bbox = data_source_meta_data.m_bboxs;
        cpd->SetFrameCount(data_source_meta_data.m_frame_cnt);
    }

    // auto mesh_ptr = cd->getData()->accessMesh().data();
    if (something_changed || _recalc) {
        // this->performMeshOperation(*mesh_ptr);
        this->convertToMesh();
        ++_version;
    }

    cpd->SetParticleListCount(1);
    cpd->AccessParticles(0).SetCount(_points.size() / 3);
    cpd->AccessParticles(0).SetGlobalRadius(0.01f);
    cpd->AccessParticles(0).SetGlobalColour(255, 255, 255, 255);
    cpd->AccessParticles(0).SetVertexData(
        geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, _points.data(), 3 * sizeof(float));
    _recalc = false;


    return true;
}

bool ManipulateMesh::getParticleData(core::Call& call) {

    auto cd = this->_getDataSlot.CallAs<mesh::CallMesh>();
    if (cd == nullptr)
        return false;

    auto cpd = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
    if (cpd == nullptr)
        return false;

    auto data_source_meta_data = cd->getMetaData();

    data_source_meta_data.m_frame_ID = cpd->FrameID();
    cd->setMetaData(data_source_meta_data);

    if (!(*cd)(1))
        return false;

    data_source_meta_data = cd->getMetaData();

    // put metadata in particle call
    cpd->AccessBoundingBoxes().SetObjectSpaceBBox(data_source_meta_data.m_bboxs.BoundingBox());
    cpd->AccessBoundingBoxes().SetObjectSpaceClipBox(data_source_meta_data.m_bboxs.ClipBox());
    cpd->SetFrameCount(data_source_meta_data.m_frame_cnt);

    return true;
}

bool ManipulateMesh::parameterChanged(core::param::ParamSlot& p) {

    _recalc = true;

    return true;
}

} // namespace probe
} // namespace megamol
