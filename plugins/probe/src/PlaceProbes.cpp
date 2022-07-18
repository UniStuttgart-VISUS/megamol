/*
 * PlaceProbes.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "PlaceProbes.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "probe/MeshUtilities.h"
#include "probe/ProbeCalls.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/ClipPlane.h"
#include <random>


namespace megamol {
namespace probe {

megamol::probe::PlaceProbes::PlaceProbes()
        : Module()
        , _version(0)
        , _mesh_slot("getMesh", "")
        , _probe_slot("deployProbes", "")
        , _centerline_slot("getCenterLine", "")
        , _method_slot("method", "")
        , _probes_per_unit_slot("Probes_per_unit", "Sets the average probe count per unit area")
        , _probe_positions_slot("deployProbePositions", "Safe probe positions to a file")
        , _load_probe_positions_slot("loadProbePositions", "Load saved probe positions")
        , _scale_probe_begin_slot("distanceFromSurfaceFactor", "")
        , _override_probe_length_slot("overrideProbeLength", "")
        , _clipplane_slot("getClipplane", "")
        , _longest_edge_index(0) {

    this->_probe_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(0), &PlaceProbes::getData);
    this->_probe_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(1), &PlaceProbes::getMetaData);
    this->MakeSlotAvailable(&this->_probe_slot);

    this->_probe_positions_slot.SetCallback(
        adios::CallADIOSData::ClassName(), adios::CallADIOSData::FunctionName(0), &PlaceProbes::getADIOSData);
    this->_probe_positions_slot.SetCallback(
        adios::CallADIOSData::ClassName(), adios::CallADIOSData::FunctionName(1), &PlaceProbes::getADIOSMetaData);
    this->MakeSlotAvailable(&this->_probe_positions_slot);

    this->_load_probe_positions_slot.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->_load_probe_positions_slot);

    this->_mesh_slot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->_mesh_slot);

    this->_centerline_slot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->_centerline_slot);

    this->_clipplane_slot.SetCompatibleCall<core::view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->_clipplane_slot);

    core::param::EnumParam* ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "vertices");
    ep->SetTypePair(1, "dart_throwing");
    ep->SetTypePair(2, "force_directed");
    ep->SetTypePair(3, "load_existing");
    ep->SetTypePair(4, "vertices+normals");
    ep->SetTypePair(5, "face+normals");
    this->_method_slot << ep;
    this->_method_slot.SetUpdateCallback(&PlaceProbes::parameterChanged);
    this->MakeSlotAvailable(&this->_method_slot);


    this->_probes_per_unit_slot << new core::param::FloatParam(1, 0);
    this->_probes_per_unit_slot.SetUpdateCallback(&PlaceProbes::parameterChanged);
    this->MakeSlotAvailable(&this->_probes_per_unit_slot);

    this->_override_probe_length_slot << new core::param::FloatParam(0);
    this->_override_probe_length_slot.SetUpdateCallback(&PlaceProbes::parameterChanged);
    this->MakeSlotAvailable(&this->_override_probe_length_slot);

    this->_scale_probe_begin_slot << new core::param::FloatParam(1.0f);
    this->_scale_probe_begin_slot.SetUpdateCallback(&PlaceProbes::parameterChanged);
    this->MakeSlotAvailable(&this->_scale_probe_begin_slot);

    /* Feasibility test */
    _probes = std::make_shared<ProbeCollection>();
    _probes->addProbe(FloatProbe());

    auto retrieved_probe = _probes->getProbe<FloatProbe>(0);

    float data;
    retrieved_probe.probe(&data);

    auto result = retrieved_probe.getSamplingResult();
}

megamol::probe::PlaceProbes::~PlaceProbes() {
    this->Release();
}

bool megamol::probe::PlaceProbes::create() {
    return true;
}

void megamol::probe::PlaceProbes::release() {}

bool megamol::probe::PlaceProbes::getData(core::Call& call) {

    auto* pc = dynamic_cast<CallProbes*>(&call);
    mesh::CallMesh* cm = this->_mesh_slot.CallAs<mesh::CallMesh>();
    mesh::CallMesh* ccl = this->_centerline_slot.CallAs<mesh::CallMesh>();

    if (cm == nullptr)
        return false;

    if (!(*cm)(0))
        return false;

    bool something_changed = cm->hasUpdate();

    auto mesh_meta_data = cm->getMetaData();
    auto probe_meta_data = pc->getMetaData();

    probe_meta_data.m_bboxs = mesh_meta_data.m_bboxs;
    _bbox = mesh_meta_data.m_bboxs;

    _mesh = cm->getData();
    core::Spatial3DMetaData centerline_meta_data;
    if (ccl != nullptr) {
        if (!(*ccl)(0))
            return false;
        something_changed = something_changed || ccl->hasUpdate();
        centerline_meta_data = ccl->getMetaData();
        _centerline = ccl->getData();
    }

    // react to user selecting different method
    if (_method_slot.IsDirty()) {
        _method_slot.ResetDirty();
        something_changed = true;
    }

    // here something really happens
    if (something_changed || _recalc) {
        ++_version;

        if (mesh_meta_data.m_bboxs.IsBoundingBoxValid()) {
            _whd = {mesh_meta_data.m_bboxs.BoundingBox().Width(), mesh_meta_data.m_bboxs.BoundingBox().Height(),
                mesh_meta_data.m_bboxs.BoundingBox().Depth()};
        } else if (centerline_meta_data.m_bboxs.IsBoundingBoxValid()) {
            _whd = {centerline_meta_data.m_bboxs.BoundingBox().Width(),
                centerline_meta_data.m_bboxs.BoundingBox().Height(),
                centerline_meta_data.m_bboxs.BoundingBox().Depth()};
        }
        _longest_edge_index = std::distance(_whd.begin(), std::max_element(_whd.begin(), _whd.end()));

        this->placeProbes();
    }

    pc->setData(this->_probes, _version);

    pc->setMetaData(probe_meta_data);
    _recalc = false;
    return true;
}

bool megamol::probe::PlaceProbes::getMetaData(core::Call& call) {

    auto* pc = dynamic_cast<CallProbes*>(&call);
    mesh::CallMesh* cm = this->_mesh_slot.CallAs<mesh::CallMesh>();
    mesh::CallMesh* ccl = this->_centerline_slot.CallAs<mesh::CallMesh>();

    if (cm == nullptr)
        return false;

    // set frame id before callback
    auto mesh_meta_data = cm->getMetaData();
    auto probe_meta_data = pc->getMetaData();

    mesh_meta_data.m_frame_ID = probe_meta_data.m_frame_ID;

    cm->setMetaData(mesh_meta_data);

    if (!(*cm)(1))
        return false;

    mesh_meta_data = cm->getMetaData();

    if (ccl != nullptr) {
        auto centerline_meta_data = ccl->getMetaData();
        centerline_meta_data.m_frame_ID = probe_meta_data.m_frame_ID;
        ccl->setMetaData(centerline_meta_data);
        if (!(*ccl)(1))
            return false;
        centerline_meta_data = ccl->getMetaData();
    }

    probe_meta_data.m_frame_cnt = mesh_meta_data.m_frame_cnt;
    probe_meta_data.m_bboxs = mesh_meta_data.m_bboxs; // normally not available here

    pc->setMetaData(probe_meta_data);

    return true;
}

void megamol::probe::PlaceProbes::dartSampling(mesh::MeshDataAccessCollection::VertexAttribute& vertices,
    mesh::MeshDataAccessCollection::IndexData indexData, float distanceIndicator) {

    uint32_t nu_triangles = indexData.byte_size / (mesh::MeshDataAccessCollection::getByteSize(indexData.type) * 3);
    auto indexDataAccessor = reinterpret_cast<uint32_t*>(indexData.data);
    auto vertexAccessor = reinterpret_cast<float*>(vertices.data);
    auto nu_verts = vertices.byte_size / vertices.stride;

    std::mt19937 rnd;
    rnd.seed(std::random_device()());
    // rnd.seed(666);
    std::uniform_real_distribution<float> fltdist(0, 1);
    std::uniform_int_distribution<uint32_t> dist(0, nu_triangles - 1);

    uint32_t nu_probes = 4000;
    _probePositions.reserve(nu_probes);

    uint32_t indx = 0;
    uint32_t error_index = 0;
    while (indx != (nu_probes - 1) && error_index < nu_probes) { //&& error_index < (nu_triangles - indx)) {

        uint32_t triangle = dist(rnd);
        std::array<float, 3> vert0;
        vert0[0] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 0] + 0];
        vert0[1] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 0] + 1];
        vert0[2] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 0] + 2];

        std::array<float, 3> vert1;
        vert1[0] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 1] + 0];
        vert1[1] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 1] + 1];
        vert1[2] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 1] + 2];

        std::array<float, 3> vert2;
        vert2[0] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 2] + 0];
        vert2[1] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 2] + 1];
        vert2[2] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 2] + 2];


        auto rnd1 = fltdist(rnd);
        auto rnd2 = fltdist(rnd);
        rnd2 *= (1 - rnd1);
        float rnd3 = 1 - (rnd1 + rnd2);

        std::array<float, 3> triangle_middle;
        triangle_middle[0] = (rnd1 * vert0[0] + rnd2 * vert1[0] + rnd3 * vert2[0]);
        triangle_middle[1] = (rnd1 * vert0[1] + rnd2 * vert1[1] + rnd3 * vert2[1]);
        triangle_middle[2] = (rnd1 * vert0[2] + rnd2 * vert1[2] + rnd3 * vert2[2]);

        bool do_placement = true;
        for (uint32_t j = 0; j < indx; j++) {
            std::array<float, 3> dist_vec;
            dist_vec[0] = triangle_middle[0] - _probePositions[j][0];
            dist_vec[1] = triangle_middle[1] - _probePositions[j][1];
            dist_vec[2] = triangle_middle[2] - _probePositions[j][2];

            const float distance =
                std::sqrt(dist_vec[0] * dist_vec[0] + dist_vec[1] * dist_vec[1] + dist_vec[2] * dist_vec[2]);

            if (distance <= distanceIndicator) {
                do_placement = false;
                break;
            }
        }

        if (do_placement) {
            _probePositions.emplace_back(
                std::array<float, 4>({triangle_middle[0], triangle_middle[1], triangle_middle[2], 1.0f}));
            indx++;
            error_index = 0;
        } else {
            error_index++;
        }
    }
    _probePositions.shrink_to_fit();
}

void megamol::probe::PlaceProbes::forceDirectedSampling(const mesh::MeshDataAccessCollection::Mesh& mesh) {


    int full_iterations = 1;
    int iterations_per_triangle = 1;
    float samples_per_area = _probes_per_unit_slot.Param<core::param::FloatParam>()->Value();
    double initial_delta_t = 1.0 / static_cast<double>(samples_per_area);


    if (!_mu) {
        _mu = std::make_shared<MeshUtility>();
        _mu->inputData(mesh);
        _numFaces = _mu->getNumTotalFaces();
    }

    float total_area = 0;
    uint32_t total_points = 0;
    _pointsPerFace.resize(_numFaces);

#pragma omp parallel for
    for (int idx = 0; idx < _numFaces; ++idx) {
        _neighborMap[idx] = _mu->getNeighboringTriangles(idx);
    }
    //#pragma omp parallel for
    for (int idx = 0; idx < _numFaces; ++idx) {

        auto area = _mu->calcTriangleArea(idx);
        total_area += area;
        auto num_points = static_cast<uint32_t>(area * samples_per_area);
        if (num_points == 0)
            num_points = 1;
        total_points += num_points;


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
            // uint32_t idx = 0;
            // transform samples points, so we can perform relaxation in 2D
            Eigen::MatrixXd patch_vertices;
            Eigen::MatrixXi patch_indices;
            _mu->getPatch(idx, patch_vertices, patch_indices, _neighborMap[idx]);

            // get longest edge
            const auto longest_edge_length = _mu->getLongestEdgeLength(idx);

            int nu_patch_points = 0;
            int nu_fixed_points = 0;
            for (auto neighbor : _neighborMap[idx]) {
                nu_patch_points += _pointsPerFace[neighbor].rows();
                if (neighbor != idx) {
                    nu_fixed_points += _pointsPerFace[neighbor].rows();
                }
            }

            Eigen::MatrixXd fixed_points(nu_fixed_points, 3);
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
            std::array<glm::vec2, 3> current_transformed_vertices;
            for (int i = 0; i < current_transformed_vertices.size(); ++i) {
                for (int j = 0; j < 2; ++j) {
                    current_transformed_vertices[i][j] = transformed_idx_verts(i, j);
                }
            }

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
#pragma omp parallel for
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
                            glm::vec2 current_partner = {
                                transformed_patch_samples(h, 0), transformed_patch_samples(h, 1)};
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
                    glm::vec2 current_point = {before_movement_buffer(k, 0), before_movement_buffer(k, 1)};
                    glm::vec2 start = {transformed_patch_samples(k, 0), transformed_patch_samples(k, 1)};
                    glm::vec2 end = {before_movement_buffer(k, 0), before_movement_buffer(k, 1)};
                    auto dif = end - start;
                    auto dist = glm::length(dif);
                    while (dist > 0.1 * longest_edge_length) {
                        end = 0.1f * dif + start;
                        dif = end - start;
                        dist = glm::length(dif);
                    }
                    if (!_mu->pointInTriangle(current_transformed_vertices, current_point)) {
                        // dont move
                        // auto new_end = 0.1f * dif + start;
                        before_movement_buffer(k, 0) = transformed_patch_samples(k, 0);
                        before_movement_buffer(k, 1) = transformed_patch_samples(k, 1);
                    }

                } // end for patch samples


                // save old positions
                old_transfromed_patch_samples = transformed_patch_samples;
                // move all particles
                transformed_patch_samples = before_movement_buffer;

                // delta_t *= 2;
            } // iterations per triangle

            // transform relaxed points back and put them in _pointsPerFace
            _mu->performInverse2Dprojection(idx, transformed_patch_samples, patch_samples);
            _pointsPerFace[idx] = patch_samples;

            // debug patch
            //_mu->fillMeshFaces(patch_indices, this->_mesh_faces);
            //_mu->fillMeshVertices(patch_vertices, this->_mesh_vertices);

            // for (int i = 0; i < _pointsPerFace[idx].rows(); ++i) {
            //    for (int j = 0; j < 3; ++j) {
            //        _points.emplace_back(_pointsPerFace[idx](i,j));
            //    }
            //}

        } // end for every triangle
    }     // end full_iterations

    // fill debug points
    _probePositions.reserve(total_points);
    for (uint32_t idx = 0; idx < _numFaces; ++idx) {
        for (int i = 0; i < _pointsPerFace[idx].rows(); ++i) {
            std::array<float, 4> point = {
                _pointsPerFace[idx](i, 0), _pointsPerFace[idx](i, 1), _pointsPerFace[idx](i, 2), 1.0f};
            _probePositions.emplace_back(point);
        }
    }
}


void megamol::probe::PlaceProbes::vertexSampling(mesh::MeshDataAccessCollection::VertexAttribute& vertices) {

    uint32_t probe_count = vertices.byte_size / vertices.stride;
    _probePositions.resize(probe_count);
    _probeVertices.resize(probe_count);

    auto vertex_accessor = reinterpret_cast<float*>(vertices.data);
    auto vertex_step = vertices.stride / sizeof(float);

#pragma omp parallel for
    for (int i = 0; i < probe_count; i++) {
        _probePositions[i][0] = vertex_accessor[vertex_step * i + 0];
        _probePositions[i][1] = vertex_accessor[vertex_step * i + 1];
        _probePositions[i][2] = vertex_accessor[vertex_step * i + 2];
        _probePositions[i][3] = 1.0f;
        _probeVertices[i] = i;
    }
}

void megamol::probe::PlaceProbes::vertexNormalSampling(mesh::MeshDataAccessCollection::VertexAttribute& vertices,
    mesh::MeshDataAccessCollection::VertexAttribute& normals,
    mesh::MeshDataAccessCollection::VertexAttribute& probe_ids) {

    uint32_t probe_count = vertices.byte_size / vertices.stride;

    auto vertex_accessor = reinterpret_cast<float*>(vertices.data);
    auto vertex_step = vertices.stride / sizeof(float);

    auto normal_accessor = reinterpret_cast<float*>(normals.data);
    auto normal_step = normals.stride / sizeof(float);

    auto probe_id_accessor = reinterpret_cast<uint32_t*>(probe_ids.data);

    //#pragma omp parallel for
    for (int i = 0; i < probe_count; i++) {

        BaseProbe probe;

        probe.m_position = {vertex_accessor[vertex_step * i + 0], vertex_accessor[vertex_step * i + 1],
            vertex_accessor[vertex_step * i + 2]};
        probe.m_direction = {-normal_accessor[normal_step * i + 0], -normal_accessor[normal_step * i + 1],
            -normal_accessor[normal_step * i + 2]};
        probe.m_begin = -2.0 * _scale_probe_begin_slot.Param<core::param::FloatParam>()->Value();
        probe.m_end = 50.0;
        probe.m_cluster_id = -1;

        auto probe_idx = this->_probes->getProbeCount();
        this->_probes->addProbe(std::move(probe));

        probe_id_accessor[i] = probe_idx;
    }
}

void PlaceProbes::faceNormalSampling(mesh::MeshDataAccessCollection::VertexAttribute& vertices,
    mesh::MeshDataAccessCollection::VertexAttribute& normals,
    mesh::MeshDataAccessCollection::VertexAttribute& probe_ids, mesh::MeshDataAccessCollection::IndexData& indices) {

    auto vertex_accessor = reinterpret_cast<float*>(vertices.data);
    auto vertex_step = vertices.stride / sizeof(float);

    auto normal_accessor = reinterpret_cast<float*>(normals.data);
    auto normal_step = normals.stride / sizeof(float);

    auto probe_id_accessor = reinterpret_cast<uint32_t*>(probe_ids.data);

    auto index_accessor = reinterpret_cast<uint32_t*>(indices.data);

    uint32_t probe_count = (indices.byte_size / sizeof(uint32_t)) / 4;

    //#pragma omp parallel for
    for (int i = 0; i < probe_count; i++) {

        BaseProbe probe;

        auto i00 = index_accessor[(i * 4)];
        auto i01 = index_accessor[(i * 4) + 1];
        auto i11 = index_accessor[(i * 4) + 2];
        auto i10 = index_accessor[(i * 4) + 3];

        glm::vec3 v00 = glm::vec3(vertex_accessor[vertex_step * i00 + 0], vertex_accessor[vertex_step * i00 + 1],
            vertex_accessor[vertex_step * i00 + 2]);
        glm::vec3 v01 = glm::vec3(vertex_accessor[vertex_step * i01 + 0], vertex_accessor[vertex_step * i01 + 1],
            vertex_accessor[vertex_step * i01 + 2]);
        glm::vec3 v11 = glm::vec3(vertex_accessor[vertex_step * i11 + 0], vertex_accessor[vertex_step * i11 + 1],
            vertex_accessor[vertex_step * i11 + 2]);
        glm::vec3 v10 = glm::vec3(vertex_accessor[vertex_step * i10 + 0], vertex_accessor[vertex_step * i10 + 1],
            vertex_accessor[vertex_step * i10 + 2]);

        glm::vec3 n00 = glm::vec3(normal_accessor[normal_step * i00 + 0], normal_accessor[normal_step * i00 + 1],
            normal_accessor[normal_step * i00 + 2]);
        glm::vec3 n01 = glm::vec3(normal_accessor[normal_step * i01 + 0], normal_accessor[normal_step * i01 + 1],
            normal_accessor[normal_step * i01 + 2]);
        glm::vec3 n11 = glm::vec3(normal_accessor[normal_step * i11 + 0], normal_accessor[normal_step * i11 + 1],
            normal_accessor[normal_step * i11 + 2]);
        glm::vec3 n10 = glm::vec3(normal_accessor[normal_step * i10 + 0], normal_accessor[normal_step * i10 + 1],
            normal_accessor[normal_step * i10 + 2]);

        glm::vec3 p_c = (v00 + v01 + v11 + v10) / 4.0f;
        glm::vec3 n_c = (n00 + n01 + n11 + n10) / 4.0f;

        probe.m_position = {p_c.x, p_c.y, p_c.z};
        probe.m_direction = {-n_c.x, -n_c.y, -n_c.z};
        probe.m_begin = -2.0 * _scale_probe_begin_slot.Param<core::param::FloatParam>()->Value();
        probe.m_end = 50.0;
        probe.m_cluster_id = -1;

        auto probe_idx = this->_probes->getProbeCount();
        this->_probes->addProbe(std::move(probe));

        probe_id_accessor[i00] = probe_idx;
        probe_id_accessor[i01] = probe_idx;
        probe_id_accessor[i11] = probe_idx;
        probe_id_accessor[i10] = probe_idx;
    }
}

bool megamol::probe::PlaceProbes::placeProbes() {


    _probes = std::make_shared<ProbeCollection>();

    //assert(_mesh->accessMeshes().size() == 1);

    mesh::MeshDataAccessCollection::VertexAttribute vertices;
    mesh::MeshDataAccessCollection::VertexAttribute centerline;

    for (auto& attribute : _mesh->accessMeshes().begin()->second.attributes) {
        if (attribute.semantic == mesh::MeshDataAccessCollection::POSITION) {
            vertices = attribute;
        }
    }

    const auto smallest_edge_index = std::distance(_whd.begin(), std::min_element(_whd.begin(), _whd.end()));
    const float distanceIndicator = _whd[smallest_edge_index] / 20;

    if (this->_method_slot.Param<core::param::EnumParam>()->Value() == 0) {
        this->vertexSampling(vertices);
        processClipplane();
        mesh::CallMesh* ccl = this->_centerline_slot.CallAs<mesh::CallMesh>();
        if (ccl == nullptr) {
            this->placeByCenterpoint();
        } else {
            assert(_centerline->accessMeshes().size() == 1);
            for (auto& attribute : _centerline->accessMeshes().begin()->second.attributes) {
                if (attribute.semantic == mesh::MeshDataAccessCollection::POSITION) {
                    centerline = attribute;
                }
            }
            this->placeByCenterline(_longest_edge_index, centerline);
        }
    } else if (this->_method_slot.Param<core::param::EnumParam>()->Value() == 1) {
        this->dartSampling(vertices, _mesh->accessMeshes().begin()->second.indices, distanceIndicator);
        processClipplane();
    } else if (this->_method_slot.Param<core::param::EnumParam>()->Value() == 2) {
        this->forceDirectedSampling(_mesh->accessMeshes().begin()->second);
        processClipplane();
    } else if (this->_method_slot.Param<core::param::EnumParam>()->Value() == 3) {
        this->loadFromFile();
        processClipplane();
    } else if (this->_method_slot.Param<core::param::EnumParam>()->Value() == 4) {

        for (auto& mesh : _mesh->accessMeshes()) {

            for (auto& attribute : mesh.second.attributes) {
                if (attribute.semantic == mesh::MeshDataAccessCollection::POSITION) {
                    vertices = attribute;
                }
            }

            mesh::MeshDataAccessCollection::VertexAttribute normals;
            for (auto& attribute : mesh.second.attributes) {
                if (attribute.semantic == mesh::MeshDataAccessCollection::NORMAL) {
                    normals = attribute;
                }
            }

            mesh::MeshDataAccessCollection::VertexAttribute probe_ids;
            for (auto& attribute : mesh.second.attributes) {
                if (attribute.semantic == mesh::MeshDataAccessCollection::ID) {
                    probe_ids = attribute;
                }
            }

            this->vertexNormalSampling(vertices, normals, probe_ids);
            processClipplane();
        }
    } else if (this->_method_slot.Param<core::param::EnumParam>()->Value() == 5) {

        for (auto& mesh : _mesh->accessMeshes()) {

            for (auto& attribute : mesh.second.attributes) {
                if (attribute.semantic == mesh::MeshDataAccessCollection::POSITION) {
                    vertices = attribute;
                }
            }

            mesh::MeshDataAccessCollection::VertexAttribute normals;
            for (auto& attribute : mesh.second.attributes) {
                if (attribute.semantic == mesh::MeshDataAccessCollection::NORMAL) {
                    normals = attribute;
                }
            }

            mesh::MeshDataAccessCollection::VertexAttribute probe_ids;
            for (auto& attribute : mesh.second.attributes) {
                if (attribute.semantic == mesh::MeshDataAccessCollection::ID) {
                    probe_ids = attribute;
                }
            }

            this->faceNormalSampling(vertices, normals, probe_ids, mesh.second.indices);
            processClipplane();
        }
    }

    return true;
}

bool megamol::probe::PlaceProbes::placeByCenterline(
    uint32_t lei, mesh::MeshDataAccessCollection::VertexAttribute& centerline) {


    uint32_t probe_count = _probePositions.size();
    // uint32_t probe_count = 1;
    uint32_t centerline_vert_count = centerline.byte_size / centerline.stride;
    //uint32_t centerline_vert_count = 0.5 * centerline.byte_size / centerline.stride;

    auto vertex_accessor = reinterpret_cast<float*>(_probePositions.data()->data());
    auto centerline_accessor = reinterpret_cast<float*>(centerline.data);

    auto vertex_step = 4;
    auto centerline_step = centerline.stride / sizeof(centerline.component_type);
    //auto centerline_step = 2 * centerline.stride / sizeof(centerline.component_type);

    std::string mesh_id;
    if (_mesh->accessMeshes().size() == 1) {
        for (auto& m : _mesh->accessMeshes()) {
            mesh_id = m.first;
        }
    }

    for (uint32_t i = 0; i < probe_count; i++) {
        BaseProbe probe;

        std::vector<float> distances(centerline_vert_count);
        for (uint32_t j = 0; j < centerline_vert_count; j++) {
            // std::array<float, 3> diffvec = {
            //    vertex_accessor[vertex_step * i + 0] - centerline_accessor[centerline_step * j + 0],
            //    vertex_accessor[vertex_step * i + 1] - centerline_accessor[centerline_step * j + 1],
            //    vertex_accessor[vertex_step * i + 2] - centerline_accessor[centerline_step * j + 2]};
            // distances[j] = std::sqrt(diffvec[0] * diffvec[0] + diffvec[1] * diffvec[1] + diffvec[2] *
            // diffvec[2]);
            auto vert = glm::vec3(vertex_accessor[vertex_step * i + 0], vertex_accessor[vertex_step * i + 1],
                vertex_accessor[vertex_step * i + 2]);
            auto centerline_vert = glm::vec3(centerline_accessor[centerline_step * j + 0],
                centerline_accessor[centerline_step * j + 1], centerline_accessor[centerline_step * j + 2]);
            distances[j] = glm::length(vert - centerline_vert);
        }

        auto min_iter = std::min_element(distances.begin(), distances.end());
        auto min_index = std::distance(distances.begin(), min_iter);
        distances[min_index] = std::numeric_limits<float>::max();

        auto second_min_iter = distances.begin();

        if (min_iter == distances.begin()) {
            second_min_iter = std::next(distances.begin());
        } else if (min_iter == std::prev(distances.end())) {
            second_min_iter = std::prev(min_iter);
        } else {
            auto left = std::prev(min_iter);
            auto right = std::next(min_iter);
            if (*left < *right) {
                second_min_iter = left;
            } else {
                second_min_iter = right;
            }
        }

        //auto second_min_iter = std::min_element(distances.begin(), distances.end());
        auto second_min_index = std::distance(distances.begin(), second_min_iter);

        // calc normal in plane between vert, min and second_min
        std::array<float, 3> along_centerline;
        along_centerline[0] = centerline_accessor[centerline_step * min_index + 0] -
                              centerline_accessor[centerline_step * second_min_index + 0];
        along_centerline[1] = centerline_accessor[centerline_step * min_index + 1] -
                              centerline_accessor[centerline_step * second_min_index + 1];
        along_centerline[2] = centerline_accessor[centerline_step * min_index + 2] -
                              centerline_accessor[centerline_step * second_min_index + 2];

        std::array<float, 3> min_to_vert;
        min_to_vert[0] = vertex_accessor[vertex_step * i + 0] - centerline_accessor[centerline_step * min_index + 0];
        min_to_vert[1] = vertex_accessor[vertex_step * i + 1] - centerline_accessor[centerline_step * min_index + 1];
        min_to_vert[2] = vertex_accessor[vertex_step * i + 2] - centerline_accessor[centerline_step * min_index + 2];

        std::array<float, 3> bitangente;
        bitangente[0] = along_centerline[1] * min_to_vert[2] - along_centerline[2] * min_to_vert[1];
        bitangente[1] = along_centerline[2] * min_to_vert[0] - along_centerline[0] * min_to_vert[2];
        bitangente[2] = along_centerline[0] * min_to_vert[1] - along_centerline[1] * min_to_vert[0];

        std::array<float, 3> normal;
        normal[0] = along_centerline[1] * bitangente[2] - along_centerline[2] * bitangente[1];
        normal[1] = along_centerline[2] * bitangente[0] - along_centerline[0] * bitangente[2];
        normal[2] = along_centerline[0] * bitangente[1] - along_centerline[1] * bitangente[0];

        // normalize normal
        float normal_length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        normal[0] /= normal_length;
        normal[1] /= normal_length;
        normal[2] /= normal_length;

        // do the projection
        float final_dist = normal[0] * min_to_vert[0] + normal[1] * min_to_vert[1] + normal[2] * min_to_vert[2];

        // flip normal to point inwards
        if (final_dist > 0) {
            normal[0] *= -1;
            normal[1] *= -1;
            normal[2] *= -1;
        } else {
            final_dist *= -1;
        }

        // Do clamping if end pos is not between center line points
        std::array<float, 3> end_pos;
        end_pos[0] = vertex_accessor[vertex_step * i + 0] + final_dist * normal[0];
        end_pos[1] = vertex_accessor[vertex_step * i + 1] + final_dist * normal[1];
        end_pos[2] = vertex_accessor[vertex_step * i + 2] + final_dist * normal[2];

        std::array<float, 3> end_pos_to_min;
        end_pos_to_min[0] = centerline_accessor[centerline_step * min_index + 0] - end_pos[0];
        end_pos_to_min[1] = centerline_accessor[centerline_step * min_index + 1] - end_pos[1];
        end_pos_to_min[2] = centerline_accessor[centerline_step * min_index + 2] - end_pos[2];

        std::array<float, 3> end_pos_to_second_min;
        end_pos_to_second_min[0] = centerline_accessor[centerline_step * second_min_index + 0] - end_pos[0];
        end_pos_to_second_min[1] = centerline_accessor[centerline_step * second_min_index + 1] - end_pos[1];
        end_pos_to_second_min[2] = centerline_accessor[centerline_step * second_min_index + 2] - end_pos[2];

        const float between_check = end_pos_to_min[0] * end_pos_to_second_min[0] +
                                    end_pos_to_min[1] * end_pos_to_second_min[1] +
                                    end_pos_to_min[2] * end_pos_to_second_min[2];

        if (between_check > 0) {
            normal[0] = centerline_accessor[centerline_step * min_index + 0] - vertex_accessor[vertex_step * i + 0];
            normal[1] = centerline_accessor[centerline_step * min_index + 1] - vertex_accessor[vertex_step * i + 1];
            normal[2] = centerline_accessor[centerline_step * min_index + 2] - vertex_accessor[vertex_step * i + 2];

            // normalize normal
            normal_length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
            normal[0] /= normal_length;
            normal[1] /= normal_length;
            normal[2] /= normal_length;
        }

        probe.m_position = {vertex_accessor[vertex_step * i + 0], vertex_accessor[vertex_step * i + 1],
            vertex_accessor[vertex_step * i + 2]};
        probe.m_direction = normal;
        auto begin = -0.1 * final_dist * _scale_probe_begin_slot.Param<core::param::FloatParam>()->Value();
        probe.m_begin = std::isfinite(begin) ? begin : 0.0f;
        probe.m_end = std::isfinite(final_dist) ? final_dist : 0.0f;
        probe.m_cluster_id = -1;
        if (!mesh_id.empty()) {
            probe.m_geo_ids.emplace_back(mesh_id);
            probe.m_vert_ids.emplace_back(_probeVertices[i]);
        }

        this->_probes->addProbe(std::move(probe));
    }

    return true;
}

bool megamol::probe::PlaceProbes::placeByCenterpoint() {
    float z_center;
    if (std::copysign(1.0, _bbox.BoundingBox().Front()) == std::copysign(1.0, _bbox.BoundingBox().Back())) {
        z_center = _bbox.BoundingBox().Front() - _bbox.BoundingBox().Back();
    } else {
        z_center = _bbox.BoundingBox().Front() + _bbox.BoundingBox().Back();
    }
    std::array<float, 3> center = {
        _bbox.BoundingBox().CalcCenter().GetX(), _bbox.BoundingBox().CalcCenter().GetY(), z_center};

    uint32_t probe_count = _probePositions.size();

    auto probe_accessor = reinterpret_cast<float*>(_probePositions.data()->data());
    auto probe_step = 4;

    std::string mesh_id;
    if (_mesh->accessMeshes().size() == 1) {
        for (auto& m : _mesh->accessMeshes()) {
            mesh_id = m.first;
        }
    }

    for (uint32_t i = 0; i < probe_count; i++) {
        BaseProbe probe;


        std::array<float, 3> normal;
        normal[0] = center[0] - probe_accessor[probe_step * i + 0];
        normal[1] = center[1] - probe_accessor[probe_step * i + 1];
        normal[2] = center[2] - probe_accessor[probe_step * i + 2];

        // normalize normal
        auto normal_length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        normal[0] /= normal_length;
        normal[1] /= normal_length;
        normal[2] /= normal_length;

        auto override_length = _override_probe_length_slot.Param<core::param::FloatParam>()
            ->Value();

        if (override_length > 0.0f){
            normal_length = override_length;
        }

        probe.m_position = {
            probe_accessor[probe_step * i + 0], probe_accessor[probe_step * i + 1], probe_accessor[probe_step * i + 2]};
        probe.m_direction = normal;
        probe.m_begin = -0.02 * normal_length * _scale_probe_begin_slot.Param<core::param::FloatParam>()->Value();
        probe.m_end = normal_length;
        probe.m_cluster_id = -1;
        if (!mesh_id.empty()) {
            probe.m_geo_ids.emplace_back(mesh_id);
            probe.m_vert_ids.emplace_back(_probeVertices[i]);
        }

        this->_probes->addProbe(std::move(probe));
    }
    return true;
}

bool megamol::probe::PlaceProbes::getADIOSData(core::Call& call) {

    auto* cadios = dynamic_cast<adios::CallADIOSData*>(&call);
    if (cadios == nullptr)
        return false;
    mesh::CallMesh* cm = this->_mesh_slot.CallAs<mesh::CallMesh>();
    mesh::CallMesh* ccl = this->_centerline_slot.CallAs<mesh::CallMesh>();

    if (cm == nullptr)
        return false;
    if (!(*cm)(0))
        return false;

    bool something_changed = cm->hasUpdate();

    auto mesh_meta_data = cm->getMetaData();

    _bbox = mesh_meta_data.m_bboxs;
    _mesh = cm->getData();

    core::Spatial3DMetaData centerline_meta_data;
    if (ccl != nullptr) {
        if (!(*ccl)(0))
            return false;
        something_changed = something_changed || ccl->hasUpdate();
        centerline_meta_data = ccl->getMetaData();
        _centerline = ccl->getData();
    }


    // here something really happens
    if (something_changed) {
        ++_version;

        if (mesh_meta_data.m_bboxs.IsBoundingBoxValid()) {
            _whd = {mesh_meta_data.m_bboxs.BoundingBox().Width(), mesh_meta_data.m_bboxs.BoundingBox().Height(),
                mesh_meta_data.m_bboxs.BoundingBox().Depth()};
        } else if (centerline_meta_data.m_bboxs.IsBoundingBoxValid()) {
            _whd = {centerline_meta_data.m_bboxs.BoundingBox().Width(),
                centerline_meta_data.m_bboxs.BoundingBox().Height(),
                centerline_meta_data.m_bboxs.BoundingBox().Depth()};
        }
        _longest_edge_index = std::distance(_whd.begin(), std::max_element(_whd.begin(), _whd.end()));

        this->placeProbes();
    }

    if (_probePositions.empty())
        return false;

    auto dataContainer = std::make_shared<adios::FloatContainer>();
    std::vector<float>& tmp_data = dataContainer->getVec();
    tmp_data.resize(_probePositions.size() * _probePositions.data()->size());
    dataContainer->shape = {_probePositions.size(), _probePositions.data()->size()};
    for (int i = 0; i < _probePositions.size(); ++i) {
        for (int j = 0; j < _probePositions.data()->size(); ++j) {
            tmp_data[_probePositions.data()->size() * i + j] = _probePositions[i][j];
        }
    }
    dataMap["xyzw"] = std::move(dataContainer);

    cadios->setData(std::make_shared<adios::adiosDataMap>(dataMap));
    // cadios->setDataHash();
    return true;
}

bool megamol::probe::PlaceProbes::getADIOSMetaData(core::Call& call) {

    auto* cadios = dynamic_cast<adios::CallADIOSData*>(&call);
    mesh::CallMesh* cm = this->_mesh_slot.CallAs<mesh::CallMesh>();

    if (cadios == nullptr)
        return false;
    if (cm == nullptr)
        return false;

    auto mesh_meta_data = cm->getMetaData();
    // mesh_meta_data._frame_ID = cadios->getFrameIDtoLoad();
    mesh_meta_data.m_frame_ID = 0;
    cm->setMetaData(mesh_meta_data);

    if (!(*cm)(1))
        return false;
    mesh_meta_data = cm->getMetaData();

    cadios->setFrameCount(mesh_meta_data.m_frame_cnt);

    std::vector<std::string> availVars = {"xyzw"};
    cadios->setAvailableVars(availVars);

    return true;
}

bool megamol::probe::PlaceProbes::loadFromFile() {

    auto cd = this->_load_probe_positions_slot.CallAs<adios::CallADIOSData>();
    if (cd == nullptr)
        return false;

    cd->setFrameIDtoLoad(0);
    if (!(*cd)(1))
        return false;
    auto availVars = cd->getAvailableVars();
    if (availVars.empty())
        return false;
    cd->inquireVar(availVars[0]);
    if (!(*cd)(0))
        return false;

    auto tmp_data = cd->getData(availVars[0])->GetAsFloat();
    _probePositions.resize(tmp_data.size() / 4);
    for (int i = 0; i < _probePositions.size(); ++i) {
        for (int j = 0; j < _probePositions.data()->size(); ++j) {
            _probePositions[i][j] = tmp_data[_probePositions.data()->size() * i + j];
        }
    }


    return true;
}

bool PlaceProbes::parameterChanged(core::param::ParamSlot& p) {
    _recalc = true;

    return true;
}

void PlaceProbes::processClipplane() {
    // process clipplane
    auto ccp = this->_clipplane_slot.CallAs<core::view::CallClipPlane>();

    if ((ccp != nullptr) && (*ccp)()) {
        glm::vec3 normal = {
            ccp->GetPlane().Normal().GetX(), ccp->GetPlane().Normal().GetY(), ccp->GetPlane().Normal().GetZ()};
        float d = ccp->GetPlane().D();

        auto probePositions = _probePositions;
        auto probeVertices = _probeVertices;
        _probePositions.clear();
        _probeVertices.clear();
        _probePositions.reserve(probePositions.size());
        _probeVertices.reserve(probePositions.size());
        for (int i = 0; i < probePositions.size(); ++i) {
            auto point = glm::vec3(
                probePositions[i][0], probePositions[i][1], probePositions[i][2]);
            if (glm::dot(normal, point ) + d > 0) {
                _probePositions.emplace_back(probePositions[i]);
                _probeVertices.emplace_back(probeVertices[i]);
            }
        }
        _probePositions.shrink_to_fit();
        _probeVertices.shrink_to_fit();

    }
}

} // namespace probe
} // namespace megamol
