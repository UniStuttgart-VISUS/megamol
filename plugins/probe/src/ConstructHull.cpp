/*
 * ConstructHull.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ConstructHull.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "iterator"
#include "mmadios/CallADIOSData.h"
#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
// normals
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/pca_estimate_normals.h>
//
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/clip.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <filesystem>
#include <random>

#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/IO/facets_in_complex_2_to_triangle_mesh.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/Polygon_mesh_processing/detect_features.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/shape_predicates.h>
#include <CGAL/Polygon_mesh_processing/smooth_mesh.h>
#include <CGAL/Polygon_mesh_processing/smooth_shape.h>
#include <CGAL/Polygonal_surface_reconstruction.h>
#include <CGAL/Shape_detection/Efficient_RANSAC.h>
#include <CGAL/convex_hull_3_to_face_graph.h>
#include <CGAL/grid_simplify_point_set.h>

#include <CGAL/Plane_3.h>
#include <CGAL/Polyhedron_3.h>

#include <glm/gtc/constants.hpp>

namespace megamol {
namespace probe {

// default triangulation for Surface_mesher
typedef CGAL::Surface_mesh<Point> Surface_mesh;
// c2t3
typedef CGAL::Complex_2_in_triangulation_3<Tr> C2t3;

typedef GT::Sphere_3 Sphere;
typedef GT::FT FT;
typedef FT (*Function)(Point);
typedef CGAL::Implicit_surface_3<GT, Function> Surface_3;

float a, b, c;
FT ellipsoid_function(Point p) {
    const FT x2 = (p.x() * p.x()) / (a * a), y2 = (p.y() * p.y()) / (b * b), z2 = (p.z() * p.z()) / (c * c);
    return x2 + y2 + z2 - 1;
}

ConstructHull::ConstructHull()
        : Module()
        , _getDataCall("getData", "")
        , _deployMeshCall("deployMesh", "")
        , _numSlices("numSlices", "")
        , _averageDistance("averageDistance", "")
        , _xSlot("x", "")
        , _ySlot("y", "")
        , _zSlot("z", "")
        , _xyzSlot("xyz", "")
        , _formatSlot("format", "")
        , _isoValue("isoValue", "")
        , _meshOutputSlot("meshOutput", "")
        , _meshToDiscCall("deployADIOS", "")
        , _meshFromDiscCall("getMeshElements", "")
        , _useBBoxAsHull("useBBoxAsHull", "")
        , _showAverageMeshDist("currentAvgMeshDist", "")
        , _showAverageParticleDist("avgParticleDist", "") {

    this->_useBBoxAsHull << new core::param::BoolParam(false);
    this->_useBBoxAsHull.SetUpdateCallback(&ConstructHull::parameterChanged);
    this->MakeSlotAvailable(&this->_useBBoxAsHull);

    this->_numSlices << new core::param::IntParam(64);
    this->_numSlices.SetUpdateCallback(&ConstructHull::parameterChanged);
    this->MakeSlotAvailable(&this->_numSlices);

    this->_isoValue << new core::param::FloatParam(1.0f);
    this->_isoValue.SetUpdateCallback(&ConstructHull::parameterChanged);
    this->MakeSlotAvailable(&this->_isoValue);

    this->_averageDistance << new core::param::FloatParam(-1.0f);
    this->_averageDistance.SetUpdateCallback(&ConstructHull::parameterChanged);
    this->MakeSlotAvailable(&this->_averageDistance);

    this->_showAverageParticleDist << new core::param::FloatParam(-1.0f);
    this->MakeSlotAvailable(&this->_showAverageParticleDist);
    this->_showAverageParticleDist.Parameter()->SetGUIReadOnly(true);

    this->_showAverageMeshDist << new core::param::FloatParam(-1.0f);
    this->MakeSlotAvailable(&this->_showAverageMeshDist);
    this->_showAverageMeshDist.Parameter()->SetGUIReadOnly(true);


    core::param::EnumParam* fp = new core::param::EnumParam(0);
    fp->SetTypePair(0, "separated");
    fp->SetTypePair(1, "interleaved");
    this->_formatSlot << fp;
    this->MakeSlotAvailable(&this->_formatSlot);

    core::param::FlexEnumParam* xEp = new core::param::FlexEnumParam("undef");
    this->_xSlot << xEp;
    this->MakeSlotAvailable(&this->_xSlot);

    core::param::FlexEnumParam* yEp = new core::param::FlexEnumParam("undef");
    this->_ySlot << yEp;
    this->MakeSlotAvailable(&this->_ySlot);

    core::param::FlexEnumParam* zEp = new core::param::FlexEnumParam("undef");
    this->_zSlot << zEp;
    this->MakeSlotAvailable(&this->_zSlot);

    core::param::FlexEnumParam* xyzEp = new core::param::FlexEnumParam("undef");
    this->_xyzSlot << xyzEp;
    this->MakeSlotAvailable(&this->_xyzSlot);

    this->_deployMeshCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &ConstructHull::getData);
    this->_deployMeshCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &ConstructHull::getMetaData);
    this->MakeSlotAvailable(&this->_deployMeshCall);

    this->_getDataCall.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->_getDataCall);
    this->_getDataCall.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);

    this->_meshToDiscCall.SetCallback(
        adios::CallADIOSData::ClassName(), adios::CallADIOSData::FunctionName(0), &ConstructHull::getADIOSData);
    this->_meshToDiscCall.SetCallback(
        adios::CallADIOSData::ClassName(), adios::CallADIOSData::FunctionName(1), &ConstructHull::getADIOSMetaData);
    this->MakeSlotAvailable(&this->_meshToDiscCall);

    this->_meshFromDiscCall.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->_meshFromDiscCall);
}

ConstructHull::~ConstructHull() {
    this->Release();
}

bool ConstructHull::create() {
    return true;
}

void ConstructHull::release() {}

bool ConstructHull::InterfaceIsDirty() {
    return this->_numSlices.IsDirty() || this->_formatSlot.IsDirty() || this->_averageDistance.IsDirty();
}

bool ConstructHull::sliceData() {

    const int num_slices = _numSlices.Param<core::param::IntParam>()->Value();

    _slice_data.resize(num_slices);
    _slice_ellipsoid.resize(num_slices);
    _sliced_positions.resize(num_slices);
    _sliced_positions_whalo.resize(num_slices);
    _sliced_vertices.resize(num_slices);
    _slice_data_center_of_mass.resize(num_slices, glm::vec3(0));
    _slice_ellipsoid_center_of_mass.resize(num_slices, glm::vec3(0));

    auto slice_begin = _bbox.BoundingBox().GetLeftBottomFront()[_main_axis];
    auto slice_width = (_bbox.BoundingBox().GetSize()[_main_axis] + 1e-3) /
                       (num_slices); // otherwise were getting exact num_slices as factor

    // slice data
    for (int i = 0; i < _particle_positions.size(); ++i) {
        int factor = (_particle_positions[i][_main_axis] - slice_begin) / slice_width;
        if (factor < 0 || factor >= _slice_data.size())
            return false;
        _slice_data[factor].emplace_back(i);
        std::array<float, 3> current_pos = {
            _particle_positions[i][0], _particle_positions[i][1], _particle_positions[i][2]};
        _sliced_positions[factor].emplace_back(current_pos);
        _sliced_positions_whalo[factor].emplace_back(current_pos);
        _slice_data_center_of_mass[factor].x += current_pos[0];
        _slice_data_center_of_mass[factor].y += current_pos[1];
        _slice_data_center_of_mass[factor].z += current_pos[2];

        if (factor == 0) {
            _sliced_positions_whalo[factor + 1].emplace_back(current_pos);
        } else if (factor == (num_slices - 1)) {
            _sliced_positions_whalo[factor - 1].emplace_back(current_pos);
        } else {
            _sliced_positions_whalo[factor + 1].emplace_back(current_pos);

            _sliced_positions_whalo[factor - 1].emplace_back(current_pos);
        }
    }

    // slice ellipsoid
    for (int j = 0; j < _vertices.size(); ++j) {
        int factor = (_vertices[j][_main_axis] - slice_begin) / slice_width;
        if (factor == num_slices)
            factor -= 1;
        if (factor < 0 || factor >= _slice_ellipsoid.size())
            return false;
        _slice_ellipsoid[factor].emplace_back(j);
        std::array<float, 3> current_pos = {_vertices[j][0], _vertices[j][1], _vertices[j][2]};
        _sliced_vertices[factor].emplace_back(current_pos);
        _slice_ellipsoid_center_of_mass[factor].x += current_pos[0];
        _slice_ellipsoid_center_of_mass[factor].y += current_pos[1];
        _slice_ellipsoid_center_of_mass[factor].z += current_pos[2];
    }

    // complete average for center of mass
    for (int n = 0; n < num_slices; ++n) {
        _slice_data_center_of_mass[n].x /= _sliced_positions[n].size();
        _slice_data_center_of_mass[n].y /= _sliced_positions[n].size();
        _slice_data_center_of_mass[n].z /= _sliced_positions[n].size();

        _slice_ellipsoid_center_of_mass[n].x /= _sliced_vertices[n].size();
        _slice_ellipsoid_center_of_mass[n].y /= _sliced_vertices[n].size();
        _slice_ellipsoid_center_of_mass[n].z /= _sliced_vertices[n].size();

        // put beginning and ending center of mass a bit more inside the data set
        if (n == 0) {
            _slice_data_center_of_mass[n][_main_axis] = slice_begin + slice_width;
            _slice_ellipsoid_center_of_mass[n][_main_axis] = slice_begin + slice_width;
        } else if (n == num_slices - 1) {
            _slice_data_center_of_mass[n][_main_axis] = slice_begin + (num_slices - 1) * slice_width;
            _slice_ellipsoid_center_of_mass[n][_main_axis] = slice_begin + (num_slices - 1) * slice_width;
        }
    }

    size_t tot_vertices = 0;
    std::for_each(_slice_ellipsoid.begin(), _slice_ellipsoid.end(),
        [&](std::vector<uint32_t> slice) { tot_vertices += slice.size(); });

    assert(tot_vertices == _vertices.size());

    return true;
}

void ConstructHull::generateEllipsoid() {
    const int num_ellipsoid_res_theta = this->_averageDistance.Param<core::param::IntParam>()->Value();
    const int num_ellipsoid_res_phi = 2 * num_ellipsoid_res_theta;
    const int num_ellipsoid_elements = (num_ellipsoid_res_phi) * (num_ellipsoid_res_theta) + num_ellipsoid_res_theta;
    std::vector<int> indices;
    indices.reserve(num_ellipsoid_elements * 3);
    const float radius_x = _bbox.BoundingBox().GetSize().Width() / 2;
    const float radius_y = _bbox.BoundingBox().GetSize().Height() / 2;
    const float radius_z = _bbox.BoundingBox().GetSize().Depth() / 2;

    std::vector<float> elVert;
    elVert.reserve((num_ellipsoid_res_phi * num_ellipsoid_res_theta) * 3);

    glm::vec3 leftbottmfront = {_bbox.BoundingBox().GetLeftBottomFront().GetX(),
        _bbox.BoundingBox().GetLeftBottomFront().GetY(), _bbox.BoundingBox().GetLeftBottomFront().GetZ()};
    glm::vec3 righttopback = {_bbox.BoundingBox().GetRightTopBack().GetX(),
        _bbox.BoundingBox().GetRightTopBack().GetY(), _bbox.BoundingBox().GetRightTopBack().GetZ()};


    for (int i = 0; i < num_ellipsoid_res_theta + 1; i++) {

        float S = static_cast<float>(i) / static_cast<float>(num_ellipsoid_res_theta);
        float phi = S * 3.141592653589793;

        for (int j = 0; j < num_ellipsoid_res_phi + 1; j++) {
            float T = static_cast<float>(j) / static_cast<float>(num_ellipsoid_res_phi);
            float theta = T * (3.141592653589793 * 2);

            // Vertices with spherical coordinates
            float x = radius_x * cos(theta) * sin(phi);
            float y = radius_y * cos(phi);
            float z = radius_z * sin(theta) * sin(phi);

            elVert.push_back(x);
            elVert.push_back(y);
            elVert.push_back(z);
        }
    }

    // Indices
    for (int i = 0; i < num_ellipsoid_elements - 1; i++) {
        indices.push_back(i);
        indices.push_back(i + num_ellipsoid_res_phi + 1);
        indices.push_back(i + num_ellipsoid_res_phi);

        indices.push_back(i + num_ellipsoid_res_phi + 1);
        indices.push_back(i);
        indices.push_back(i + 1);
    }

    _vertices.resize(elVert.size() / 3);
    _triangles.resize(indices.size() / 3);
    _normals.resize(_vertices.size());


    for (int i = 0; i < _vertices.size(); ++i) {

        _vertices[i][0] = elVert[3 * i + 0];
        _vertices[i][1] = elVert[3 * i + 1];
        _vertices[i][2] = elVert[3 * i + 2];
        _vertices[i][3] = 1.0f;
    }
    for (int i = 0; i < _triangles.size(); ++i) {
        _triangles[i][0] = indices[3 * i + 0];
        _triangles[i][1] = indices[3 * i + 1];
        _triangles[i][2] = indices[3 * i + 2];

        //std::array<int, 3> triangle = {
        //    _triangles[i][0], _triangles[i][1],
        //    _triangles[i][2]};

        //assert(_triangles[i][0] < _vertices.size());
        //assert(_triangles[i][1] < _vertices.size());
        //assert(_triangles[i][2] < _vertices.size());

        //glm::vec3 vert0 = {
        //    _vertices[_triangles[i][0]][0],
        //    _vertices[_triangles[i][0]][1],
        //    _vertices[_triangles[i][0]][2]};
        //glm::vec3 vert1 = {
        //    _vertices[_triangles[i][1]][0],
        //    _vertices[_triangles[i][1]][1],
        //    _vertices[_triangles[i][1]][2]};
        //glm::vec3 vert2 = {
        //    _vertices[_triangles[i][2]][0],
        //    _vertices[_triangles[i][2]][1],
        //    _vertices[_triangles[i][2]][2]};

        //auto dif01 = vert1 - vert0;
        //auto dif02 = vert2 - vert0;
        ////auto difcenter = origin - vert0;
        //auto normal = glm::cross(dif01, dif02);

        //if (glm::dot(normal, vert0) > 0) {
        //    normal *= -1;
        //}
        //normal = glm::normalize(normal);

        //_normals[_triangles[i][0]][0] = normal.x;
        //_normals[_triangles[i][0]][1] = normal.y;
        //_normals[_triangles[i][0]][2] = normal.z;
    }

// translate to center of bbox
#pragma omp parallel for
    for (int i = 0; i < _vertices.size(); ++i) {
        _vertices[i][0] += _data_origin.x;
        _vertices[i][1] += _data_origin.y;
        _vertices[i][2] += _data_origin.z;
    }
}

void ConstructHull::generateEllipsoid_2() {
    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;

    float scale_factor = _bbox.BoundingBox().GetSize()[_main_axis];

    megamol::probe::a = (_bbox.BoundingBox().GetSize().Width() / 2) / scale_factor;
    megamol::probe::b = _bbox.BoundingBox().GetSize().Height() / 2 / scale_factor;
    megamol::probe::c = _bbox.BoundingBox().GetSize().Depth() / 2 / scale_factor;

    Tr tr;         // 3D-Delaunay triangulation
    C2t3 c2t3(tr); // 2D-complex in 3D-Delaunay triangulation
    // defining the surface
    Surface_3 surface(ellipsoid_function, // pointer to function
        Sphere(CGAL::ORIGIN, 2.));        // bounding sphere
    // Note that "2." above is the *squared* radius of the bounding sphere!
    //
    // defining meshing criteria
    CGAL::Surface_mesh_default_criteria_3<Tr> criteria(5., // angular bound
        0.01,                                              // radius bound
        0.01);                                             // distance bound
    // meshing surface
    _sm.clear();
    CGAL::make_surface_mesh(c2t3, surface, criteria, CGAL::Non_manifold_tag());
    CGAL::facets_in_complex_2_to_triangle_mesh(c2t3, _sm);

    //CGAL::Aff_transformation_3<CGAL::Epick> trafo_scale(Kernel::RT(scale_factor), Kernel::RT(0), Kernel::RT(0),
    //    Kernel::RT(0), Kernel::RT(0), Kernel::RT(scale_factor), Kernel::RT(0),
    //    Kernel::RT(0), Kernel::RT(0), Kernel::RT(0), Kernel::RT(scale_factor),
    //    Kernel::RT(0));

    //CGAL::Aff_transformation_3<CGAL::Epick> trafo_translate(Kernel::RT(0), Kernel::RT(0), Kernel::RT(0),
    //    Kernel::RT(_data_origin[0]), Kernel::RT(0), Kernel::RT(0), Kernel::RT(0),
    //    Kernel::RT(_data_origin[1]), Kernel::RT(0), Kernel::RT(0), Kernel::RT(0),
    //    Kernel::RT(_data_origin[2]));

    _vertices.clear();
    _vertices.resize(_sm.num_vertices());
    _ellipsoid_backup.clear();
    _ellipsoid_backup.resize(_sm.num_vertices());
    for (int i = 0; i < _sm.num_vertices(); ++i) {
        auto it = std::next(_sm.points().begin(), i);

        glm::vec3 p = {it->x(), it->y(), it->z()};

        p.x = p.x * scale_factor + _data_origin[0];
        p.y = p.y * scale_factor + _data_origin[1];
        p.z = p.z * scale_factor + _data_origin[2];

        const Point point(p.x, p.y, p.z);
        *it = point;

        _vertices[i][0] = it->x();
        _vertices[i][1] = it->y();
        _vertices[i][2] = it->z();

        if (!std::isfinite(_vertices[i][0]) || !std::isfinite(_vertices[i][1]) || !std::isfinite(_vertices[i][2])) {
            throw std::exception("[ConstructHull] Non-finite vertices detected.");
        }
    }

    _triangles.clear();
    _triangles.resize(_sm.num_faces());
    auto triangle = _triangles.begin();
    for (CGAL::Surface_mesh<Point>::face_index fi : _sm.faces()) {
        auto hf = _sm.halfedge(fi);
        auto triangle_index = triangle->begin();
        for (CGAL::Surface_mesh<Point>::Halfedge_index hi : _sm.halfedges_around_face(hf)) {
            *triangle_index = CGAL::target(hi, _sm);
            triangle_index = std::next(triangle_index);
        }
        triangle = std::next(triangle);
    }
}

void ConstructHull::generateEllipsoid_3() {

    const float radius_x = _bbox.BoundingBox().GetSize().Width() / 2;
    const float radius_y = _bbox.BoundingBox().GetSize().Height() / 2;
    const float radius_z = _bbox.BoundingBox().GetSize().Depth() / 2;

    int num_samples = 5000;

    _vertices.resize(num_samples);

    glm::vec3 leftbottmfront = {_bbox.BoundingBox().GetLeftBottomFront().GetX(),
        _bbox.BoundingBox().GetLeftBottomFront().GetY(), _bbox.BoundingBox().GetLeftBottomFront().GetZ()};
    glm::vec3 righttopback = {_bbox.BoundingBox().GetRightTopBack().GetX(),
        _bbox.BoundingBox().GetRightTopBack().GetY(), _bbox.BoundingBox().GetRightTopBack().GetZ()};

    std::mt19937 rnd;
    rnd.seed(std::random_device()());
    std::normal_distribution<float> dist_phi(0, glm::two_pi<float>());
    std::normal_distribution<float> dist_theta(0, glm::pi<float>());


#pragma omp parallel for
    for (int i = 0; i < num_samples; i++) {

        float phi = dist_phi(rnd);

        float theta = dist_theta(rnd);

        // Vertices with spherical coordinates
        float x = radius_x * cos(theta) * sin(phi);
        float y = radius_y * cos(phi);
        float z = radius_z * sin(theta) * sin(phi);

        _vertices[i][0] = x + _data_origin.x;
        _vertices[i][1] = y + _data_origin.y;
        _vertices[i][2] = z + _data_origin.z;
    }
}

float ConstructHull::compute_avg_particle_distance() {
    // create adaptor
    auto data2kd = std::make_shared<const data2KD>(_particle_positions);

    // construct a kd-tree index:
    auto kd_indices = std::make_shared<my_kd_tree_t>(
        3 /*dim*/, *data2kd, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    kd_indices->buildIndex();

    int num_results = 2;
    float distances = 0.0f;
    for (int i = 0; i < _particle_positions.size(); ++i) {
        std::vector<size_t> ret_index(num_results);
        std::vector<float> sqr_dist(num_results);
        nanoflann::KNNResultSet<float> resultSet(num_results);
        resultSet.init(ret_index.data(), sqr_dist.data());


        kd_indices->findNeighbors(resultSet, &_particle_positions[i][0], nanoflann::SearchParams(10));
        size_t neighbor_index = ret_index[1];
        glm::vec3 current_pos = {_particle_positions[i][0], _particle_positions[i][1], _particle_positions[i][2]};
        glm::vec3 neighbor_pos = {_particle_positions[neighbor_index][0], _particle_positions[neighbor_index][1],
            _particle_positions[neighbor_index][2]};

        distances += glm::length(neighbor_pos - current_pos);
    }
    distances /= _particle_positions.size();

    return distances;
}

void ConstructHull::tighten() {

    // generate kd stuff
    _kd_indices.clear();
    _kd_indices.resize(_sliced_positions_whalo.size());
    _data2kd.clear();
    _data2kd.resize(_sliced_positions_whalo.size());

    for (int i = 0; i < _kd_indices.size(); ++i) {

        // create adaptor
        _data2kd[i] = std::make_shared<const data2KD>(_sliced_positions_whalo[i]);

        // construct a kd-tree index:
        _kd_indices[i] = std::make_shared<my_kd_tree_t>(
            3 /*dim*/, *_data2kd[i], nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        _kd_indices[i]->buildIndex();
    }

    std::map<int, int> not_main_axis;
    not_main_axis[0] = 0;
    not_main_axis[1] = 1;
    not_main_axis[2] = 2;
    not_main_axis.erase(_main_axis);

    for (int n = 0; n < _slice_ellipsoid.size(); ++n) {
        //for (int n = 0; n < 1; ++n) {
        glm::vec3 center_of_mass_dif = _slice_data_center_of_mass[n] - _slice_ellipsoid_center_of_mass[n];

        for (int i = 0; i < _slice_ellipsoid[n].size(); i++) {
            glm::vec3 vertex = {_vertices[_slice_ellipsoid[n][i]][0], _vertices[_slice_ellipsoid[n][i]][1],
                _vertices[_slice_ellipsoid[n][i]][2]};
            for (auto axis : not_main_axis) {
                vertex[axis.second] += center_of_mass_dif[axis.second];
            }
            glm::vec3 direction = vertex - _slice_data_center_of_mass[n];
            direction = glm::normalize(direction);

            int main_direction = 0;
            // main direction of vertex translation
            //if (n == 0 || n == _slice_ellipsoid.size() - 1) {
            if (std::abs(direction.x) < std::abs(direction.y)) {
                main_direction = 1;
                if (std::abs(direction.y) < std::abs(direction.z)) {
                    main_direction = 2;
                }
            } else {
                if (std::abs(direction.x) < std::abs(direction.z)) {
                    main_direction = 2;
                }
            }
            //} else {
            //    std::array<int,2> element;
            //    int g = 0;
            //    for (auto axis : not_main_axis) {
            //        element[g] = axis.second;
            //        ++g;
            //    }
            //    if (std::abs(direction[element[0]]) > std::abs(direction[element[1]])) {
            //        main_direction = element[0];
            //    } else {
            //        main_direction = element[1];
            //    }
            //}


            float plane_push_amount = _bbox.BoundingBox().GetSize()[main_direction] * 0.2f; //0.01f;
            float dist_to_plane = 0;
            glm::vec3 point_on_plane;
            if (direction[main_direction] < 0.0f) {
                dist_to_plane = _bbox.BoundingBox().GetLeftBottomFront()[main_direction] - plane_push_amount;
                point_on_plane = {_bbox.BoundingBox().GetLeftBottomFront()[0],
                    _bbox.BoundingBox().GetLeftBottomFront()[1], _bbox.BoundingBox().GetLeftBottomFront()[2]};
                point_on_plane[main_direction] -= plane_push_amount;
            } else {
                dist_to_plane = _bbox.BoundingBox().GetRightTopBack()[main_direction] + plane_push_amount;
                point_on_plane = {_bbox.BoundingBox().GetRightTopBack()[0], _bbox.BoundingBox().GetRightTopBack()[1],
                    _bbox.BoundingBox().GetRightTopBack()[2]};
                point_on_plane[main_direction] += plane_push_amount;
            }
            glm::vec3 plane_normal(0.0f);
            plane_normal[main_direction] = 1.0f;

            float t = (glm::dot(plane_normal, point_on_plane) - glm::dot(plane_normal, vertex)) /
                      (glm::dot(plane_normal, direction));

            glm::vec3 fi_start = vertex + direction * t;
            float length = glm::length(fi_start - _slice_data_center_of_mass[n]);

            const int num_samples_along_path = 10;
            float fi_sample_step = length / num_samples_along_path;

            const size_t num_results = 1;
            std::vector<float> fi_distances(num_samples_along_path);
            for (int k = 0; k < num_samples_along_path; ++k) {
                size_t ret_index;
                float sqr_dist;
                nanoflann::KNNResultSet<float> resultSet(num_results);
                resultSet.init(&ret_index, &sqr_dist);

                std::array<float, 3> query_point;
                query_point[0] = fi_start.x - direction.x * fi_sample_step * k;
                query_point[1] = fi_start.y - direction.y * fi_sample_step * k;
                query_point[2] = fi_start.z - direction.z * fi_sample_step * k;

                _kd_indices[n]->findNeighbors(resultSet, &query_point[0], nanoflann::SearchParams(10));
                fi_distances[k] = std::sqrt(sqr_dist);
            }
            const float iso_value = _isoValue.Param<core::param::FloatParam>()->Value();
            auto fi_max_k =
                std::distance(fi_distances.begin(), std::max_element(fi_distances.begin(), fi_distances.end()));
            auto fi_zero_element = std::distance(fi_distances.begin(),
                std::find_if(fi_distances.begin(), fi_distances.end(), [iso_value](auto x) { return x < iso_value; }));
            auto fi_min_element = std::min_element(fi_distances.begin(), fi_distances.end()) - fi_distances.begin();
            if (fi_zero_element == 10) {
                fi_zero_element = fi_min_element;
            }

            auto fi_gradient = 0.0f;
            for (int l = fi_max_k; l < fi_zero_element - 1; ++l) {
                fi_gradient += (fi_distances[l + 1] - fi_distances[l]);
            }
            auto fi_length_compensation = ((fi_zero_element - 1) - fi_max_k);
            fi_gradient /= fi_length_compensation;
            auto fi_predicted_k = -(fi_distances[fi_max_k]) / fi_gradient;

            //second query
            std::vector<float> si_distances(num_samples_along_path);
            glm::vec3 si_start;
            si_start.x = fi_start.x - direction.x * fi_sample_step * (fi_zero_element - 2);
            si_start.y = fi_start.y - direction.y * fi_sample_step * (fi_zero_element - 2);
            si_start.z = fi_start.z - direction.z * fi_sample_step * (fi_zero_element - 2);
            glm::vec3 new_end;
            new_end.x = fi_start.x - direction.x * fi_sample_step * (fi_zero_element + 1);
            new_end.y = fi_start.y - direction.y * fi_sample_step * (fi_zero_element + 1);
            new_end.z = fi_start.z - direction.z * fi_sample_step * (fi_zero_element + 1);

            float si_sample_step = glm::length(new_end - si_start) / num_samples_along_path;

            for (int k = 0; k < num_samples_along_path; ++k) {
                size_t ret_index;
                float sqr_dist;
                nanoflann::KNNResultSet<float> resultSet(num_results);
                resultSet.init(&ret_index, &sqr_dist);

                std::array<float, 3> query_point;
                query_point[0] = si_start.x - direction.x * si_sample_step * k;
                query_point[1] = si_start.y - direction.y * si_sample_step * k;
                query_point[2] = si_start.z - direction.z * si_sample_step * k;

                _kd_indices[n]->findNeighbors(resultSet, &query_point[0], nanoflann::SearchParams(10));
                si_distances[k] = std::sqrt(sqr_dist);
            }

            auto si_max_k =
                std::distance(fi_distances.begin(), std::max_element(fi_distances.begin(), fi_distances.end()));
            auto si_zero_element = std::distance(fi_distances.begin(),
                std::find_if(fi_distances.begin(), fi_distances.end(), [iso_value](auto x) { return x < iso_value; }));
            auto si_min_element = std::min_element(si_distances.begin(), si_distances.end()) - si_distances.begin();
            if (si_zero_element == 10) {
                si_zero_element = si_min_element;
            }

            float si_gradient = 0.0f;
            for (int l = si_max_k; l < si_zero_element - 1; ++l) {
                si_gradient += (si_distances[l + 1] - si_distances[l]);
            }
            auto si_length_compensation = ((si_zero_element - 1) - si_max_k);
            si_gradient /= si_length_compensation;
            auto predicted_k = -(si_distances[si_max_k]) / si_gradient;
            auto start = si_start;
            auto sample_step = si_sample_step;
            if (!std::isfinite(predicted_k)) {
                predicted_k = fi_predicted_k;
                start = fi_start;
                sample_step = fi_sample_step;
            }
            //assert(std::isfinite(predicted_k));


            // if (n == 0) {
            if (std::isfinite(predicted_k)) {
                _vertices[_slice_ellipsoid[n][i]][0] = start.x - direction.x * sample_step * predicted_k;
                _vertices[_slice_ellipsoid[n][i]][1] = start.y - direction.y * sample_step * predicted_k;
                _vertices[_slice_ellipsoid[n][i]][2] = start.z - direction.z * sample_step * predicted_k;
            } else {
                _vertices[_slice_ellipsoid[n][i]][0] = vertex[0];
                _vertices[_slice_ellipsoid[n][i]][1] = vertex[1];
                _vertices[_slice_ellipsoid[n][i]][2] = vertex[2];
            }
            //} else {
            //// DEBUG center of mass translation
            //_vertices[_slice_ellipsoid[n][i]][0] = vertex[0];
            //_vertices[_slice_ellipsoid[n][i]][1] = vertex[1];
            //_vertices[_slice_ellipsoid[n][i]][2] = vertex[2];
            //}
        }
    }

    // Push new positions in surface mesh
    //for (int i = 0; i < _sm.num_vertices(); ++i) {
    //    auto it = std::next(_sm.points().begin(), i);

    //    const Point point(_vertices[i][0], _vertices[i][1], _vertices[i][2]);
    //    *it = point;
    //}


    // Compute average spacing using neighborhood of 6 points
    //double spacing = CGAL::compute_average_spacing<CGAL::Parallel_if_available_tag>(points_for_triangulation, 6,
    //    CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
    //        .normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));
    // Simplify using a grid of size 2 * average spacing
    //CGAL::grid_simplify_point_set(points_for_triangulation, 2. * spacing,
    //    CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
    //        .normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));
    //auto removed = points_for_triangulation.number_of_removed_points();
    //points_for_triangulation.collect_garbage();

    //CGAL::pca_estimate_normals<CGAL::Parallel_if_available_tag>(points_for_triangulation, 12,
    //    CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
    //        .normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));
    // Orientation of normals, returns iterator to first unoriented point
    //std::vector<PointVectorPair>::iterator unoriented_points_begin =
    //    CGAL::mst_orient_normals(points_for_triangulation, 12,
    //        CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
    //            .normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));
    //points_for_triangulation.erase(unoriented_points_begin, points_for_triangulation.end());

    //CGAL::poisson_surface_reconstruction_delaunay(points_for_triangulation.begin(), points_for_triangulation.end(),
    //    CGAL::First_of_pair_property_map<PointVectorPair>(), CGAL::Second_of_pair_property_map<PointVectorPair>(),
    //    _sm, spacing);
}

void ConstructHull::compute_surface_from_vertices() {
    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef K::Point_3 Point_3;
    std::list<Point_3> points_for_triangulation;

    for (int i = 0; i < _vertices.size(); ++i) {
        points_for_triangulation.push_back(Point(_vertices[i][0], _vertices[i][1], _vertices[i][2]));
    }

    typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
    typedef Delaunay::Vertex_handle Vertex_handle;
    typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
    // void CGAL::facets_in_complex_2_to_triangle_mesh(c2t3, TriangleMesh & graph)

    Delaunay T(points_for_triangulation.begin(), points_for_triangulation.end());
    _sm.clear();
    CGAL::convex_hull_3_to_face_graph(T, _sm);
}

void ConstructHull::do_remeshing(Surface_mesh& mesh, float spacing_) {

    float spacing = 0.0f;
    if (spacing_ == 0.0f) {
        spacing = CGAL::compute_average_spacing<CGAL::Parallel_if_available_tag>(mesh.points(), 6);
    } else {
        spacing = spacing_;
    }

    CGAL::Polygon_mesh_processing::split_long_edges(mesh.edges(), spacing, mesh);

    CGAL::Polygon_mesh_processing::isotropic_remeshing(
        mesh.faces(), spacing, mesh); //, CGAL::Polygon_mesh_processing::parameters::number_of_iterations(3));
    mesh.collect_garbage();
}

void ConstructHull::generateNormals(Surface_mesh& mesh) {

    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
    typedef Kernel::Point_3 Point;
    typedef Kernel::Vector_3 Vector;
    typedef std::pair<Point, Vector> PointVectorPair;

    std::list<PointVectorPair> points;
    //for (auto point : mesh.points()) {
    //    Vector vector = {0,0,0};
    //    PointVectorPair pvp = std::make_pair(point,vector);
    //    points.emplace_back(pvp);
    //}
    for (int i = 0; i < mesh.num_vertices(); ++i) {
        auto it = std::next(mesh.points().begin(), i);
        points.emplace_back(std::make_pair(*it, Vector(0, 0, 0)));
    }

    int nb_neighbors = 24;

    try {
        CGAL::pca_estimate_normals<CGAL::Parallel_if_available_tag>(points, nb_neighbors,
            CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
                .normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));
    } catch (const std::exception& e) {
        core::utility::log::Log::DefaultLog.WriteError(
            std::string("[ConstructHull] NormalEstimation exited with ").append(e.what()).c_str());
    }

    // Orientation of normals, returns iterator to first unoriented point
    //std::list<PointVectorPair>::iterator unoriented_points_begin = CGAL::mst_orient_normals(points, nb_neighbors,
    //    CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
    //        .normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));


    _normals.resize(mesh.num_vertices());
    for (int i = 0; i < points.size(); ++i) {
        auto it = std::next(points.begin(), i);
        glm::vec3 normal = {it->second.x(), it->second.y(), it->second.z()};
        glm::vec3 vert0 = {it->first.x(), it->first.y(), it->first.z()};
        auto difcenter = vert0 - _data_origin;

        if (glm::dot(normal, difcenter) > 0.0f) {
            normal *= -1;
        }

        _normals[i] = {normal.x, normal.y, normal.z};
    }
}

void ConstructHull::generateNormals_2(Surface_mesh& mesh, std::vector<std::array<float, 3>>& normals) {

    typedef boost::graph_traits<Surface_mesh>::vertex_descriptor vertex_descriptor;
    typedef boost::graph_traits<Surface_mesh>::face_descriptor face_descriptor;
    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef K::Vector_3 Vector;

    namespace PMP = CGAL::Polygon_mesh_processing;

    auto vnormals = mesh.add_property_map<vertex_descriptor, Vector>("v:normals", CGAL::NULL_VECTOR).first;

    PMP::compute_vertex_normals(mesh, vnormals);

    normals.clear();
    normals.reserve(mesh.num_vertices());
    for (vertex_descriptor vd : vertices(mesh)) {
        std::array<float, 3> normal = {vnormals[vd].x(), vnormals[vd].y(), vnormals[vd].z()};
        normals.emplace_back(normal);
    }
}


void ConstructHull::do_smoothing(Surface_mesh& mesh) {
    std::set<Surface_mesh::Vertex_index> constrained_vertices;
    for (Surface_mesh::Vertex_index v : vertices(mesh)) {
        if (is_border(v, mesh))
            constrained_vertices.insert(v);
    }
    CGAL::Boolean_property_map<std::set<Surface_mesh::Vertex_index>> vcmap(constrained_vertices);
    CGAL::Polygon_mesh_processing::smooth_shape(mesh, 0.05,
        CGAL::Polygon_mesh_processing::parameters::number_of_iterations(1).vertex_is_constrained_map(vcmap));
    mesh.collect_garbage();
}

void ConstructHull::generateBox() {

    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef K::Point_3 Point_3;
    typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
    typedef Delaunay::Vertex_handle Vertex_handle;
    typedef CGAL::Surface_mesh<Point_3> Surface_mesh;

    Surface_mesh cube;

    Point lbf = Point(_bbox.BoundingBox().Left(), _bbox.BoundingBox().Bottom(), _bbox.BoundingBox().Front());
    Point rtb = Point(_bbox.BoundingBox().Right(), _bbox.BoundingBox().Top(), _bbox.BoundingBox().Back());

    std::list<Point_3> points_for_triangulation;
    points_for_triangulation.push_back(Point(lbf[0], lbf[1], lbf[2]));
    points_for_triangulation.push_back(Point(rtb[0], lbf[1], lbf[2]));
    points_for_triangulation.push_back(Point(lbf[0], lbf[1], rtb[2]));
    points_for_triangulation.push_back(Point(lbf[0], rtb[1], lbf[2]));
    points_for_triangulation.push_back(Point(rtb[0], lbf[1], rtb[2]));
    points_for_triangulation.push_back(Point(lbf[0], rtb[1], rtb[2]));
    points_for_triangulation.push_back(Point(rtb[0], rtb[1], lbf[2]));
    points_for_triangulation.push_back(Point(rtb[0], rtb[1], rtb[2]));


    Delaunay T(points_for_triangulation.begin(), points_for_triangulation.end());
    _sm.clear();
    CGAL::convex_hull_3_to_face_graph(T, _sm);

    // Surface_mesh cube;
    // CGAL::make_hexahedron(p0,p1,p2,p3,p4,p5,p6,p7,cube);
    // CGAL::Polygon_mesh_processing::triangulate_faces(cube);

    //_sm.clear();
    //_sm = cube;
    // CGAL::Polygon_mesh_processing::isotropic_remeshing(cube.faces(), (*min_d/10.0f), cube);
    // bool is_cube_triangle_mesh = CGAL::is_triangle_mesh(cube);
    // bool cube_does_self_intersect = CGAL::Polygon_mesh_processing::does_self_intersect(cube);
    // if (cube_does_self_intersect) {
    //    std::vector<std::pair<face_descriptor, face_descriptor>> intersected_tris;
    //    PMP::self_intersections(cube, std::back_inserter(intersected_tris));
    //    for (std::pair<face_descriptor, face_descriptor>& p : intersected_tris) {
    //        CGAL::Euler::remove_face(cube.halfedge(get<0>(p)), cube);
    //    }
    //}

    // bool is_triangle_mesh = CGAL::is_triangle_mesh(_shells[i]);
    // bool does_self_intersect = CGAL::Polygon_mesh_processing::does_self_intersect(_shells[i]);
}

bool ConstructHull::compute() {
    // find main axis
    std::map<int, int> axes;
    axes[0] = 0;
    axes[1] = 1;
    axes[2] = 2;
    std::array<float, 3> whd = {
        this->_bbox.BoundingBox().Width(), this->_bbox.BoundingBox().Height(), this->_bbox.BoundingBox().Depth()};
    _main_axis = std::distance(whd.begin(), std::max_element(whd.begin(), whd.end()));
    axes.erase(_main_axis);
    int l = 0;
    for (auto ax : axes) {
        _off_axes[l] = ax.first;
        ++l;
    }


    // get origin
    float center_z;
    if (std::copysign(1.0f, _bbox.BoundingBox().Front()) != std::copysign(1.0f, _bbox.BoundingBox().Back())) {
        center_z = (_bbox.BoundingBox().Front() + _bbox.BoundingBox().Back()) / 2;
    } else {
        center_z = _bbox.BoundingBox().Front() + (_bbox.BoundingBox().Back() - _bbox.BoundingBox().Front()) / 2;
    }
    _data_origin = {_bbox.BoundingBox().CalcCenter().GetX(), _bbox.BoundingBox().CalcCenter().GetY(), center_z};

    auto userDist = _averageDistance.Param<core::param::FloatParam>()->Value();
    if (_useBBoxAsHull.Param<core::param::BoolParam>()->Value()) {
        this->generateBox();
        auto spacing = CGAL::compute_average_spacing<CGAL::Parallel_if_available_tag>(_sm.points(), 8);
        _showAverageMeshDist.Param<core::param::FloatParam>()->SetValue(spacing);
        if (userDist > 0.0f) {
            this->do_remeshing(_sm, userDist);
        }
    } else {
        //auto avg_p_dist = compute_avg_particle_distance();
        //_showAverageParticleDist.Param<core::param::FloatParam>()->SetValue(avg_p_dist);
        this->generateEllipsoid_3();
        if (!this->sliceData())
            return false;
        this->tighten();
        this->compute_surface_from_vertices();
        auto avg_mesh_spacing = CGAL::compute_average_spacing<CGAL::Parallel_if_available_tag>(_sm.points(), 8);
        _showAverageMeshDist.Param<core::param::FloatParam>()->SetValue(avg_mesh_spacing);
        if (userDist > 0.0f) {
            this->do_remeshing(_sm, userDist);
        } else {
            this->do_remeshing(_sm);
        }
        this->do_smoothing(_sm);
    }
    this->generateNormals(_sm);
    return true;
}

bool ConstructHull::processRawData(adios::CallADIOSData* call, bool& something_changed) {
    std::vector<std::string> toInq;
    toInq.clear();
    if (this->_formatSlot.Param<core::param::EnumParam>()->Value() == 0) {
        toInq.emplace_back(std::string(this->_xSlot.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(std::string(this->_ySlot.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(std::string(this->_zSlot.Param<core::param::FlexEnumParam>()->ValueString()));
    } else {
        toInq.emplace_back(std::string(this->_xyzSlot.Param<core::param::FlexEnumParam>()->ValueString()));
    }

    // get data from adios
    for (auto var : toInq) {
        if (!call->inquireVar(var)) {
            return false;
        }
    }

    if (!(*call)(0))
        return false;

    // get data from volumetric call
    if (call->getDataHash() != _old_datahash) {
        something_changed = true;

        if (this->_formatSlot.Param<core::param::EnumParam>()->Value() == 0) {
            auto x = call->getData(std::string(this->_xSlot.Param<core::param::FlexEnumParam>()->ValueString()))
                         ->GetAsFloat();
            auto y = call->getData(std::string(this->_ySlot.Param<core::param::FlexEnumParam>()->ValueString()))
                         ->GetAsFloat();
            auto z = call->getData(std::string(this->_zSlot.Param<core::param::FlexEnumParam>()->ValueString()))
                         ->GetAsFloat();
            assert(x.size() == y.size());
            assert(y.size() == z.size());
            _particle_positions.resize(x.size());
            for (int i = 0; i < x.size(); ++i) {
                _particle_positions[i][0] = x[i];
                _particle_positions[i][1] = y[i];
                _particle_positions[i][2] = z[i];
            }
            auto xminmax = std::minmax_element(x.begin(), x.end());
            auto yminmax = std::minmax_element(y.begin(), y.end());
            auto zminmax = std::minmax_element(z.begin(), z.end());
            _bbox.SetBoundingBox(
                *xminmax.first, *yminmax.first, *zminmax.second, *xminmax.second, *yminmax.second, *zminmax.first);
        } else {
            const std::string varname = std::string(_xyzSlot.Param<core::param::FlexEnumParam>()->ValueString());
            auto positions = call->getData(varname)->GetAsFloat();
            float xmin = std::numeric_limits<float>::max();
            float xmax = -std::numeric_limits<float>::max();
            float ymin = std::numeric_limits<float>::max();
            float ymax = -std::numeric_limits<float>::max();
            float zmin = std::numeric_limits<float>::max();
            float zmax = -std::numeric_limits<float>::max();
            _particle_positions.clear();
            _particle_positions.resize(positions.size() / 3);
            for (int i = 0; i < positions.size() / 3; ++i) {
                _particle_positions[i][0] = positions[3 * i + 0];
                _particle_positions[i][1] = positions[3 * i + 1];
                _particle_positions[i][2] = positions[3 * i + 2];
                xmin = std::min(xmin, _particle_positions[i][0]);
                xmax = std::max(xmax, _particle_positions[i][0]);
                ymin = std::min(ymin, _particle_positions[i][1]);
                ymax = std::max(ymax, _particle_positions[i][1]);
                zmin = std::min(zmin, _particle_positions[i][2]);
                zmax = std::max(zmax, _particle_positions[i][2]);
            }
            _bbox.SetBoundingBox(xmin, ymin, zmax, xmax, ymax, zmin);
        }
    }
    return true;
}


bool ConstructHull::getData(core::Call& call) {

    bool something_changed = _recalc;

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr)
        return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr)
        return false;

    if (!this->processRawData(cd, something_changed)) {
        core::utility::log::Log::DefaultLog.WriteError("Could not process incoming data. Abort.");
        return false;
    }

    if (something_changed && !_particle_positions.empty()) {
        ++_version;
        auto mfdc = this->_meshFromDiscCall.CallAs<adios::CallADIOSData>();
        if (!mfdc) {
            if (!this->compute()) {
                core::utility::log::Log::DefaultLog.WriteError("[ConstructHull] Error during hull computation");
                return false;
            }
        } else {
            if (!this->readHullFromFile()) {
                core::utility::log::Log::DefaultLog.WriteError(
                    "[ConstructHull] Could not load mesh from File. Starting computation ...");
                if (!this->compute())
                    return false;
            }
        }

        this->activateMesh(_sm, _vertices, _triangles);
        this->generateNormals(_sm);

        _mesh_for_call = nullptr;

        _mesh.second.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute());
        _mesh.second.back().component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
        _mesh.second.back().byte_size = _vertices.size() * sizeof(std::array<float, 3>);
        _mesh.second.back().component_cnt = 3;
        _mesh.second.back().stride = sizeof(std::array<float, 3>);
        _mesh.second.back().offset = 0;
        _mesh.second.back().data = reinterpret_cast<uint8_t*>(_vertices.data());
        _mesh.second.back().semantic = mesh::MeshDataAccessCollection::POSITION;

        if (!_normals.empty()) {
            assert(_normals.size() == _vertices.size());
            _mesh.second.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute());
            _mesh.second.back().component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
            _mesh.second.back().byte_size = _normals.size() * sizeof(std::array<float, 3>);
            _mesh.second.back().component_cnt = 3;
            _mesh.second.back().stride = sizeof(std::array<float, 3>);
            _mesh.second.back().offset = 0;
            _mesh.second.back().data = reinterpret_cast<uint8_t*>(_normals.data());
            _mesh.second.back().semantic = mesh::MeshDataAccessCollection::NORMAL;
        }

        _mesh.first.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
        _mesh.first.byte_size = _triangles.size() * sizeof(std::array<uint32_t, 3>);
        _mesh.first.data = reinterpret_cast<uint8_t*>(_triangles.data());
    } // something changed


    if (!_mesh_for_call) {
        // put data in mesh
        _mesh_for_call = std::make_shared<mesh::MeshDataAccessCollection>();
        std::string identifier = std::string(FullName()) + "_mesh";
        _mesh_for_call->addMesh(
            identifier, _mesh.second, _mesh.first, mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES);
    }

    cm->setData(_mesh_for_call, _version);

    _old_datahash = cd->getDataHash();
    _recalc = false;

    auto meta_data = cm->getMetaData();
    meta_data.m_bboxs = _bbox;
    cm->setMetaData(meta_data);

    return true;
}

bool ConstructHull::getADIOSMetaData(core::Call& call) {
    auto out = dynamic_cast<adios::CallADIOSData*>(&call);
    if (out == nullptr)
        return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr)
        return false;

    // get metadata
    cd->setFrameIDtoLoad(0); // TODO: just one frame supported now
    if (!(*cd)(1))
        return false;
    if (cd->getDataHash() == _old_datahash) {
        auto vars = cd->getAvailableVars();
        for (auto var : vars) {
            this->_xSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->_ySlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->_zSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->_xyzSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        }
    }
    out->setFrameCount(cd->getFrameCount());
    return true;
}

bool ConstructHull::getADIOSData(core::Call& call) {
    bool something_changed = _recalc;

    auto out = dynamic_cast<adios::CallADIOSData*>(&call);
    if (out == nullptr)
        return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr)
        return false;

    if (!this->processRawData(cd, something_changed)) {
        core::utility::log::Log::DefaultLog.WriteError("Could not process incoming data. Abort.");
        return false;
    }

    if (something_changed && !_particle_positions.empty()) {
        auto mfdc = this->_meshFromDiscCall.CallAs<adios::CallADIOSData>();
        if (!mfdc) {
            this->compute();
        } else {
            if (!this->readHullFromFile()) {
                core::utility::log::Log::DefaultLog.WriteError(
                    "[ConstructHull] Could not load mesh from File. Starting computation ...");
                this->compute();
            }
        }
        assert(!_sm.is_empty());

        auto isBBoxCont = std::make_shared<adios::Int32Container>(adios::Int32Container());
        std::vector<int>& isBBox_vec = isBBoxCont->getVec();
        isBBox_vec.resize(1);
        isBBox_vec[0] = _useBBoxAsHull.Param<core::param::BoolParam>()->Value();

        auto hullCont = std::make_shared<adios::CharContainer>(adios::CharContainer());
        std::vector<char>& hull_vec = hullCont->getVec();

        std::stringstream hull_sstream;
        hull_sstream << std::setprecision(17) << _sm;
        auto hull_str = hull_sstream.str();

        hull_vec.reserve(hull_str.size());
        std::copy(hull_str.begin(), hull_str.end(), std::back_inserter(hull_vec));

        auto boxCont = std::make_shared<adios::FloatContainer>(adios::FloatContainer());
        std::vector<float>& box_vec = boxCont->getVec();
        box_vec = {_bbox.BoundingBox().Left(), _bbox.BoundingBox().Bottom(), _bbox.BoundingBox().Back(),
            _bbox.BoundingBox().Right(), _bbox.BoundingBox().Top(), _bbox.BoundingBox().Front()};

        _dataMap["hull"] = std::move(hullCont);
        _dataMap["bbox"] = std::move(boxCont);
        _dataMap["isBBox"] = std::move(isBBoxCont);
    }

    out->setData(std::make_shared<adios::adiosDataMap>(_dataMap));
    out->setDataHash(cd->getDataHash());

    _old_datahash = cd->getDataHash();
    _recalc = false;

    return true;
}

bool ConstructHull::readHullFromFile() {

    auto cd = _meshFromDiscCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr)
        return false;

    cd->setFrameIDtoLoad(0); // TODO: maybe support more frames in the future
    if (!(*cd)(1))
        return false;

    auto vars = cd->getAvailableVars();

    // get data from adios
    for (auto var : vars) {
        if (!cd->inquireVar(var)) {
            core::utility::log::Log::DefaultLog.WriteError(
                (std::string("[ConstructHull] Could not inquire ") + var).c_str());
            return false;
        }
    }

    if (!(*cd)(0))
        return false;

    auto hull = cd->getData("hull")->GetAsChar();
    std::string hull_str(hull.begin(), hull.end());
    _sm.clear();
    std::stringstream(hull_str) >> _sm;

    return true;
}

bool ConstructHull::getMetaData(core::Call& call) {

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr)
        return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr)
        return false;

    auto meta_data = cm->getMetaData();

    // get metadata for volumetric call
    cd->setFrameIDtoLoad(meta_data.m_frame_ID);
    if (!(*cd)(1))
        return false;
    if (cd->getDataHash() == _old_datahash) {
        auto vars = cd->getAvailableVars();
        for (auto var : vars) {
            this->_xSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->_ySlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->_zSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->_xyzSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        }
    }

    //if (cd->DataHash() == _old_datahash && !_recalc)
    //    return true;

    // put metadata in mesh call
    //meta_data.m_bboxs = cd->AccessBoundingBoxes();
    meta_data.m_bboxs = _bbox;
    meta_data.m_frame_cnt = cd->getFrameCount();
    cm->setMetaData(meta_data);

    return true;
}

bool ConstructHull::parameterChanged(core::param::ParamSlot& p) {
    _recalc = true;

    return true;
}

void ConstructHull::activateMesh(const Surface_mesh& shell, std::vector<std::array<float, 3>>& vertices,
    std::vector<std::array<uint32_t, 3>>& indices) {

    vertices.clear();
    vertices.resize(shell.num_vertices());
    for (int i = 0; i < shell.num_vertices(); ++i) {
        auto it = std::next(shell.points().begin(), i);

        vertices[i][0] = it->x();
        vertices[i][1] = it->y();
        vertices[i][2] = it->z();
    }

    indices.clear();
    indices.resize(shell.num_faces());
    auto triangle = indices.begin();
    for (CGAL::Surface_mesh<Point>::face_index fi : shell.faces()) {
        auto hf = shell.halfedge(fi);
        auto triangle_index = triangle->begin();
        for (CGAL::Surface_mesh<Point>::Halfedge_index hi : shell.halfedges_around_face(hf)) {
            *triangle_index = CGAL::target(hi, shell);
            triangle_index = std::next(triangle_index);
        }
        triangle = std::next(triangle);
    }
}


} // namespace probe
} // namespace megamol
