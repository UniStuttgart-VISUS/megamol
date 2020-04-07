/*
 * ManipulateMesh.cpp
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ManipulateMesh.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/IntParam.h"


namespace megamol {
namespace probe {

float coulomb_force(float r) { return (1.0f / pow(r, 2)); }

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

    this->_pointsDebugSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &ManipulateMesh::getParticleData);
    this->_pointsDebugSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &ManipulateMesh::getParticleMetaData);
    this->MakeSlotAvailable(&this->_pointsDebugSlot);

    this->_getDataSlot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->_getDataSlot);
}

ManipulateMesh::~ManipulateMesh() { this->Release(); }

bool ManipulateMesh::create() { return true; }

void ManipulateMesh::release() {}

bool ManipulateMesh::InterfaceIsDirty() { return this->_numFacesSlot.IsDirty(); }

bool ManipulateMesh::performMeshOperation() {

    int full_iterations = 1;
    int iterations_per_triangle = 1;
    double initial_delta_t = 1e-3;
    float samples_per_area = 10.0f;

    if (!_mu) {
        _mu = std::make_shared<MeshUtility>();
        _mu->inputData(_mesh_constptr);
        _numFaces = _mu->getNumTotalFaces();
    }

    float total_area = 0;
    uint32_t total_points = 0;
    _pointsPerFace.resize(_numFaces);

    //#pragma omp parallel for
    for (int idx = 0; idx < _numFaces; ++idx) {

        auto area = _mu->calcTriangleArea(idx);
        total_area += area;
        auto num_points = static_cast<uint32_t>(area * samples_per_area);
        total_points += num_points;

        _neighborMap[idx] = _mu->getNeighboringTriangles(idx);
        // Eigen::MatrixXd patch_vertices;
        // Eigen::MatrixXi patch_indices;
        //_mu->getPatch(idx, patch_vertices, patch_indices, _neighborMap[idx]);

        // Eigen::MatrixXd vertices_uv;
        //_mu->UVMapping(patch_indices, patch_vertices, vertices_uv);

        _mu->seedPoints(idx, num_points, _pointsPerFace[idx]);

        //_mu->fillMeshFaces(patch_indices, this->_mesh_faces);
        //_mu->fillMeshVertices(patch_vertices, this->_mesh_vertices);
        //_mu->fillMeshVertices(vertices_uv, this->_mesh_vertices);


        // project uv mapping onto bbox
        // auto scale = std::min(_bbox.BoundingBox().Width(), _bbox.BoundingBox().Height());
        // std::array<float, 3> mid = {_bbox.BoundingBox().Left() + _bbox.BoundingBox().Width() / 2,
        //    _bbox.BoundingBox().Bottom() + _bbox.BoundingBox().Height()/2, _bbox.BoundingBox().Front()};
        // for (int i = 0; i < this->_mesh_vertices.size() / 3; ++i) {
        //    for (int j = 0; j < 3; ++j) {
        //        _mesh_vertices[3 * i + j] *= scale;
        //        _mesh_vertices[3 * i + j] += mid[j];
        //    }
        //}
    }

    for (int full_it = 0; full_it < full_iterations; ++full_it) {
        //#pragma omp parallel for
        for (uint32_t idx = 0; idx < _numFaces; ++idx) {
            //uint32_t idx = 0;
            // transform samples points, so we can perform relaxation in 2D
            Eigen::MatrixXd patch_vertices;
            Eigen::MatrixXi patch_indices;
            _mu->getPatch(idx, patch_vertices, patch_indices, _neighborMap[idx]);

            int num_patch_points = 0;
            int num_fixed_points = 0;
            for (auto neighbor : _neighborMap[idx]) {
                num_patch_points += _pointsPerFace[neighbor].rows();
                if (neighbor != idx) {
                    num_fixed_points += _pointsPerFace[neighbor].rows();
                }
            }

            Eigen::MatrixXd fixed_points(num_fixed_points, 3);
            int n = 0;
            for (auto neighbor : _neighborMap[idx]) {
                if (neighbor != idx) {
                    for (int i = 0; i < _pointsPerFace[neighbor].rows(); ++i) {
                        for (int j = 0; j < _pointsPerFace[neighbor].cols(); ++j) {
                            fixed_points(n, j) = _pointsPerFace[neighbor](i, j);
                        }
                        n++;
                    }
                }
            }
            Eigen::MatrixXd transformed_fixed_points;
            _mu->perform2Dprojection(idx, fixed_points, transformed_fixed_points);
            Eigen::MatrixXd patch_samples;
            patch_samples.resize(_pointsPerFace[idx].rows(), 3);
            for (int i = 0; i < patch_samples.rows(); ++i) {
                for (int j = 0; j < patch_samples.cols(); ++j) {
                    patch_samples(i, j) = _pointsPerFace[idx](i, j);
                }
            }
            Eigen::MatrixXd transformed_patch_samples;
            _mu->perform2Dprojection(idx, patch_samples, transformed_patch_samples);

            // project triangle vertices as well for easy inTriangle checks
            Eigen::Matrix3d idx_verts;
            _mu->getTriangleVertices(idx, idx_verts);
            Eigen::MatrixXd transformed_idx_verts;
            _mu->perform2Dprojection(idx, idx_verts, transformed_idx_verts);

            std::vector<Eigen::Matrix3d> neighbor_verts(_neighborMap[idx].size() - 1);
            std::vector<Eigen::MatrixXd> transformed_neighbor_verts(_neighborMap[idx].size() - 1);
            n = 0;
            for (int i = 0; i < _neighborMap[idx].size(); ++i) {
                if (_neighborMap[idx][i] != idx) {
                    _mu->getTriangleVertices(_neighborMap[idx][i], neighbor_verts[n]);
                    _mu->perform2Dprojection(idx, neighbor_verts[n], transformed_neighbor_verts[n]);
                    ++n;
                }
            }

            // perform relaxation (omit n-dimension)
            // using the verlet integration scheme
            auto delta_t = initial_delta_t;
            Eigen::MatrixXd old_transfromed_patch_samples = transformed_patch_samples;
            for (int step = 0; step < iterations_per_triangle; ++step) {
                glm::vec2 force = {0.0f, 0.0f};
                Eigen::MatrixXd before_movement_buffer;
                before_movement_buffer.resizeLike(transformed_patch_samples);
                // sum up forces
                // for every (not fixed) particle
//#    pragma omp parallel for
                for (int k = 0; k < transformed_patch_samples.rows(); ++k) {
                    glm::vec2 current_particle = {transformed_patch_samples(k, 0), transformed_patch_samples(k, 1)};
                    // interaction with fixed samples
                    for (int l = 0; l < transformed_fixed_points.rows(); ++l) {
                        glm::vec2 current_partner = {transformed_fixed_points(l, 0), transformed_fixed_points(l, 1)};
                        glm::vec2 dif = current_partner - current_particle;
                        auto amount_force = coulomb_force(glm::length(dif));
                        force += amount_force * glm::normalize(dif);
                    }
                    // interaction with not fixed samples
                    for (int h = 0; h < transformed_patch_samples.rows(); ++h) {
                        if (h != k) {
                            glm::vec2 current_partner = {transformed_patch_samples(h, 0), transformed_patch_samples(h, 1)};
                            glm::vec2 dif = current_partner - current_particle;
                            auto amount_force = coulomb_force(glm::length(dif));
                            force += amount_force * glm::normalize(dif);
                        }
                    }
                    // buffer new positions
                    before_movement_buffer(k, 0) = 2 * transformed_patch_samples(k, 0) -
                                                   old_transfromed_patch_samples(k, 0) + force.x * std::pow(delta_t, 2);
                    before_movement_buffer(k, 1) = 2 * transformed_patch_samples(k, 1) -
                                                   old_transfromed_patch_samples(k, 1) + force.y * std::pow(delta_t, 2);
                    before_movement_buffer(k, 2) = transformed_patch_samples(k, 2);

                    // Check if new positions are still inside the triangle

                } // end for patch samples



                // save old positions
                old_transfromed_patch_samples = transformed_patch_samples;
                // move all particles
                transformed_patch_samples = before_movement_buffer;

                //delta_t *= 2;
            } // iterations per triangle

            // transform relaxed points back and put them in _pointsPerFace
            _mu->performInverse2Dprojection(idx, transformed_patch_samples, patch_samples);
            _pointsPerFace[idx] = patch_samples;

            // debug patch
            //_mu->fillMeshFaces(patch_indices, this->_mesh_faces);
            //_mu->fillMeshVertices(patch_vertices, this->_mesh_vertices);

            //for (int i = 0; i < _pointsPerFace[idx].rows(); ++i) {
            //    for (int j = 0; j < 3; ++j) {
            //        _points.emplace_back(_pointsPerFace[idx](i,j));
            //    }
            //}

        } // end for every triangle
    }     // end full_iterations

    // fill debug points
    _points.reserve(total_points * 3);
    for (uint32_t idx = 0; idx < _numFaces; ++idx) {
        for (int i = 0; i < _pointsPerFace[idx].rows(); ++i) {
            for (int j = 0; j < 3; ++j) {
                _points.emplace_back(_pointsPerFace[idx](i, j));
            }
        }
    }


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

    _mesh_constptr = std::make_shared<const mesh::MeshDataAccessCollection::Mesh>(*cd->getData()->accessMesh().data());
    if (something_changed || _recalc) {
        this->performMeshOperation();
        this->convertToMesh();
        ++_version;
    }

    // put data in mesh
    mesh::MeshDataAccessCollection mesh;

    mesh.addMesh(_mesh_attribs, _mesh_indices, mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES);
    cm->setData(std::make_shared<mesh::MeshDataAccessCollection>(std::move(mesh)), _version);
    _recalc = false;

    return true;
}

bool ManipulateMesh::getMetaData(core::Call& call) {

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

bool ManipulateMesh::getParticleMetaData(core::Call& call) {

    auto cd = this->_getDataSlot.CallAs<mesh::CallMesh>();
    if (cd == nullptr) return false;

    auto cpd = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
    if (cpd == nullptr) return false;

    bool something_changed = _recalc;

    if (!(*cd)(0)) return false;
    if (cd->hasUpdate()) {
        something_changed = true;
        auto data_source_meta_data = cd->getMetaData();

        cpd->AccessBoundingBoxes().SetObjectSpaceBBox(data_source_meta_data.m_bboxs.BoundingBox());
        cpd->AccessBoundingBoxes().SetObjectSpaceClipBox(data_source_meta_data.m_bboxs.ClipBox());
        _bbox = data_source_meta_data.m_bboxs;
        cpd->SetFrameCount(data_source_meta_data.m_frame_cnt);
    }

    _mesh_constptr = std::make_shared<const mesh::MeshDataAccessCollection::Mesh>(*cd->getData()->accessMesh().data());
    if (something_changed || _recalc) {
        this->performMeshOperation();
        this->convertToMesh();
        ++_version;
    }

    cpd->SetParticleListCount(1);
    cpd->AccessParticles(0).SetCount(_points.size() / 3);
    cpd->AccessParticles(0).SetGlobalRadius(0.01f);
    cpd->AccessParticles(0).SetGlobalColour(255, 255, 255, 255);
    cpd->AccessParticles(0).SetVertexData(
        core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, _points.data(), 3 * sizeof(float));
    _recalc = false;


    return true;
}

bool ManipulateMesh::getParticleData(core::Call& call) {

    auto cd = this->_getDataSlot.CallAs<mesh::CallMesh>();
    if (cd == nullptr) return false;

    auto cpd = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
    if (cpd == nullptr) return false;

    auto data_source_meta_data = cd->getMetaData();

    data_source_meta_data.m_frame_ID = cpd->FrameID();
    cd->setMetaData(data_source_meta_data);

    if (!(*cd)(1)) return false;

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
