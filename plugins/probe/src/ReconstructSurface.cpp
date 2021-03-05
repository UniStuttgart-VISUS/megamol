/*
 * ReconstructSurface.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ReconstructSurface.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FloatParam.h"
#include "adios_plugin/CallADIOSData.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/BoundingBoxes_2.h"
#include "iterator"
// normals
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>
// 
#include <filesystem>
#include <random>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/clip.h>

#include <CGAL/convex_hull_3_to_face_graph.h>
#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/IO/facets_in_complex_2_to_triangle_mesh.h>
#include <CGAL/Polygonal_surface_reconstruction.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/Shape_detection/Efficient_RANSAC.h>
#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/Polygon_mesh_processing/smooth_mesh.h>
#include <CGAL/Polygon_mesh_processing/detect_features.h>
#include <CGAL/Polygon_mesh_processing/smooth_shape.h>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/Plane_3.h>

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
    
    float a,b,c;
    FT ellipsoid_function(Point p) {
        const FT x2 = (p.x() * p.x()) / (a * a), y2 = (p.y() * p.y()) / (b * b), z2 = (p.z() * p.z()) / (c * c);
        return x2 + y2 + z2 - 1;
    }

    ReconstructSurface::ReconstructSurface()
            : Module()
            , _getDataCall("getData", "")
            , _deployMeshCall("deployMesh", "")
            , _deployNormalsCall("deployNormals", "")
            , _numSlices("HullGeneration::numSlices", "")
            , _numSamples("HullGeneration::numSamples", "")
            , _xSlot("x", "")
            , _ySlot("y", "")
            , _zSlot("z", "")
            , _xyzSlot("xyz", "")
            , _formatSlot("format", "")
            , _isoValue("HullGeneration::isoValue", "")
            , _showShellSlot("Shells::showShell", "")
            , _numShellsSlot("Shells::numShells", "")
            , _meshOutputSlot("meshOutput", "")
            , _meshToDiscCall("deployADIOS","")
            , _meshFromDiscCall("getMeshElements", "")
            , _shellSplitsAxis("Elements::splitsAxis", "")
            , _shellSplitsAngle("Elements::slitsAngle","")
    {

        this->_numSlices << new core::param::IntParam(64);
        this->_numSlices.SetUpdateCallback(&ReconstructSurface::parameterChanged);
        this->MakeSlotAvailable(&this->_numSlices);

        this->_isoValue << new core::param::FloatParam(1.0f);
        this->_isoValue.SetUpdateCallback(&ReconstructSurface::parameterChanged);
        this->MakeSlotAvailable(&this->_isoValue);

        this->_numSamples << new core::param::IntParam(64);
        this->_numSamples.SetUpdateCallback(&ReconstructSurface::parameterChanged);
        this->MakeSlotAvailable(&this->_numSamples);

        core::param::EnumParam* fp = new core::param::EnumParam(0);
        fp->SetTypePair(0, "separated");
        fp->SetTypePair(1, "interleaved");
        this->_formatSlot << fp;
        this->MakeSlotAvailable(&this->_formatSlot);

        core::param::EnumParam* mos = new core::param::EnumParam(0);
        mos->SetTypePair(0, "all");
        mos->SetTypePair(1, "singleShell");
        mos->SetTypePair(2, "singleElement");
        this->_meshOutputSlot << mos;
        this->MakeSlotAvailable(&this->_meshOutputSlot);

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

        this->_showShellSlot << new core::param::IntParam(-1);
        this->_showShellSlot.SetUpdateCallback(&ReconstructSurface::shellToShowChanged);
        this->MakeSlotAvailable(&this->_showShellSlot);

        this->_numShellsSlot << new core::param::IntParam(10);
        this->_numShellsSlot.SetUpdateCallback(&ReconstructSurface::parameterChanged);
        this->MakeSlotAvailable(&this->_numShellsSlot);

        this->_shellSplitsAxis << new core::param::IntParam(10);
        this->_shellSplitsAxis.SetUpdateCallback(&ReconstructSurface::parameterChanged);
        this->MakeSlotAvailable(&this->_shellSplitsAxis);

        this->_shellSplitsAngle << new core::param::IntParam(8);
        this->_shellSplitsAngle.SetUpdateCallback(&ReconstructSurface::parameterChanged);
        this->MakeSlotAvailable(&this->_shellSplitsAngle);

        this->_deployMeshCall.SetCallback(
            mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &ReconstructSurface::getData);
        this->_deployMeshCall.SetCallback(
            mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &ReconstructSurface::getMetaData);
        this->MakeSlotAvailable(&this->_deployMeshCall);

        this->_deployNormalsCall.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
            core::moldyn::MultiParticleDataCall::FunctionName(0), &ReconstructSurface::getNormalData);
        this->_deployNormalsCall.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
            core::moldyn::MultiParticleDataCall::FunctionName(1), &ReconstructSurface::getNormalMetaData);
        this->MakeSlotAvailable(&this->_deployNormalsCall);

        this->_getDataCall.SetCompatibleCall<adios::CallADIOSDataDescription>();
        this->MakeSlotAvailable(&this->_getDataCall);

        this->_meshToDiscCall.SetCallback(adios::CallADIOSData::ClassName(), adios::CallADIOSData::FunctionName(0),
            &ReconstructSurface::getADIOSData);
        this->_meshToDiscCall.SetCallback(adios::CallADIOSData::ClassName(), adios::CallADIOSData::FunctionName(1),
            &ReconstructSurface::getADIOSMetaData);
        this->MakeSlotAvailable(&this->_meshToDiscCall);

        this->_meshFromDiscCall.SetCompatibleCall<adios::CallADIOSDataDescription>();
        this->MakeSlotAvailable(&this->_meshFromDiscCall);
    }

    ReconstructSurface::~ReconstructSurface() {
        this->Release();
    }

    bool ReconstructSurface::create() {
        return true;
    }

    void ReconstructSurface::release() {}

    bool ReconstructSurface::InterfaceIsDirty() {
        return this->_numSlices.IsDirty() || this->_formatSlot.IsDirty() || this->_numSamples.IsDirty();
    }

    void ReconstructSurface::sliceData() {
 
        const int num_slices = _numSlices.Param<core::param::IntParam>()->Value();

        _slice_data.resize(num_slices);
        _slice_ellipsoid.resize(num_slices);
        _sliced_positions.resize(num_slices);
        _sliced_positions_whalo.resize(num_slices);
        _sliced_vertices.resize(num_slices);
        _slice_data_center_of_mass.resize(num_slices, glm::vec3(0));
        _slice_ellipsoid_center_of_mass.resize(num_slices, glm::vec3(0));

        auto slice_begin = _bbox.BoundingBox().GetLeftBottomFront()[_main_axis];
        auto slice_width = (_bbox.BoundingBox().GetSize()[_main_axis] + 1e-3) / (num_slices); // otherwise were getting exact num_slices as factor

        // slice data
        for (int i = 0; i < _raw_positions.size() / 3; ++i) {
            int factor = (_raw_positions[3 * i + _main_axis] - slice_begin) / slice_width;

            _slice_data[factor].emplace_back(i);
            std::array<float, 3> current_pos = {_raw_positions[3 * i + 0], _raw_positions[3 * i + 1],
                _raw_positions[3 * i + 2]};
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
            if (factor == num_slices) factor -= 1;
            _slice_ellipsoid[factor].emplace_back(j);
            std::array<float,3> current_pos = { _vertices[j][0], _vertices[j][1], _vertices[j][2] };
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

    }

    void ReconstructSurface::generateEllipsoid() {
        const int num_ellipsoid_res_theta = this->_numSamples.Param<core::param::IntParam>()->Value();
        const int num_ellipsoid_res_phi = 2 * num_ellipsoid_res_theta;
        const int num_ellipsoid_elements =
            (num_ellipsoid_res_phi) * (num_ellipsoid_res_theta) + num_ellipsoid_res_theta;
        std::vector<int> indices;
        indices.reserve(num_ellipsoid_elements*3);
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

    void ReconstructSurface::generateEllipsoid_2() {
        typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;

        float scale_factor = _bbox.BoundingBox().GetSize()[_main_axis];

        megamol::probe::a = (_bbox.BoundingBox().GetSize().Width() / 2) / scale_factor;
        megamol::probe::b = _bbox.BoundingBox().GetSize().Height() / 2 / scale_factor;
        megamol::probe::c = _bbox.BoundingBox().GetSize().Depth() / 2 / scale_factor;

        Tr tr;         // 3D-Delaunay triangulation
        C2t3 c2t3(tr); // 2D-complex in 3D-Delaunay triangulation
        // defining the surface
        Surface_3 surface(ellipsoid_function, // pointer to function
            Sphere(CGAL::ORIGIN, 2.));   // bounding sphere
        // Note that "2." above is the *squared* radius of the bounding sphere!
        //
        // defining meshing criteria
        CGAL::Surface_mesh_default_criteria_3<Tr> criteria(5., // angular bound
            0.01,                                                // radius bound
            0.01);                                               // distance bound
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
                throw std::exception("[ReconstructSurface] Non-finite vertices detected.");
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

    void ReconstructSurface::generateEllipsoid_3() {

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

    void ReconstructSurface::tighten() {

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
                glm::vec3 vertex = {
                    _vertices[_slice_ellipsoid[n][i]][0], _vertices[_slice_ellipsoid[n][i]][1],
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


                float plane_push_amount = _bbox.BoundingBox().GetSize()[main_direction] * 0.01f;
                float dist_to_plane = 0;
                glm::vec3 point_on_plane;
                if (direction[main_direction] < 0.0f) {
                    dist_to_plane = _bbox.BoundingBox().GetLeftBottomFront()[main_direction] - plane_push_amount;
                    point_on_plane = {
                        _bbox.BoundingBox().GetLeftBottomFront()[0],
                        _bbox.BoundingBox().GetLeftBottomFront()[1],
                        _bbox.BoundingBox().GetLeftBottomFront()[2]};
                    point_on_plane[main_direction] -= plane_push_amount;
                } else {
                    dist_to_plane = _bbox.BoundingBox().GetRightTopBack()[main_direction] + plane_push_amount;
                    point_on_plane = {_bbox.BoundingBox().GetRightTopBack()[0],
                        _bbox.BoundingBox().GetRightTopBack()[1],
                        _bbox.BoundingBox().GetRightTopBack()[2]};
                    point_on_plane[main_direction] += plane_push_amount;
                }
                glm::vec3 plane_normal(0.0f);
                plane_normal[main_direction] = 1.0f;

                float t = (glm::dot(plane_normal, point_on_plane) - glm::dot(plane_normal, vertex)) /
                          (glm::dot(plane_normal,direction));

                glm::vec3 start = vertex + direction * t;
                float length = glm::length(start - _slice_data_center_of_mass[n]);

                const int num_samples_along_path = 10;
                float sample_step = length / num_samples_along_path;

                const size_t num_results = 1;
                std::vector<float> distances(num_samples_along_path);
                for (int k = 0; k < num_samples_along_path; ++k) {
                    size_t ret_index;
                    float sqr_dist;
                    nanoflann::KNNResultSet<float> resultSet(num_results);
                    resultSet.init(&ret_index, &sqr_dist);
                    
                    std::array<float,3> query_point;
                    query_point[0] = start.x - direction.x * sample_step * k;
                    query_point[1] = start.y - direction.y * sample_step * k;
                    query_point[2] = start.z - direction.z * sample_step * k;

                    _kd_indices[n]->findNeighbors(resultSet, &query_point[0], nanoflann::SearchParams(10));
                    distances[k] = std::sqrt(sqr_dist);

                }
                float iso_value = _isoValue.Param<core::param::FloatParam>()->Value();
                auto max_k = std::distance(distances.begin(),std::max_element(distances.begin(), distances.end()));
                auto zero_element = std::distance(distances.begin(),
                    std::find_if(distances.begin(), distances.end(), [iso_value](auto x) { return x < iso_value; }));
                //second query
                glm::vec3 new_start;
                new_start.x = start.x - direction.x * sample_step * (zero_element - 2);
                new_start.y = start.y - direction.y * sample_step * (zero_element - 2);
                new_start.z = start.z - direction.z * sample_step * (zero_element - 2);
                glm::vec3 new_end;
                new_end.x = start.x - direction.x * sample_step * (zero_element+1);
                new_end.y = start.y - direction.y * sample_step * (zero_element+1);
                new_end.z = start.z - direction.z * sample_step * (zero_element+1);

                float new_sample_step = glm::length(new_end - new_start) / num_samples_along_path;

                for (int k = 0; k < num_samples_along_path; ++k) {
                    size_t ret_index;
                    float sqr_dist;
                    nanoflann::KNNResultSet<float> resultSet(num_results);
                    resultSet.init(&ret_index, &sqr_dist);

                    std::array<float, 3> query_point;
                    query_point[0] = new_start.x - direction.x * new_sample_step * k;
                    query_point[1] = new_start.y - direction.y * new_sample_step * k;
                    query_point[2] = new_start.z - direction.z * new_sample_step * k;

                    _kd_indices[n]->findNeighbors(resultSet, &query_point[0], nanoflann::SearchParams(10));
                    distances[k] = std::sqrt(sqr_dist);
                }

                max_k = std::distance(distances.begin(), std::max_element(distances.begin(), distances.end()));
                zero_element = std::distance(distances.begin(),
                    std::find_if(distances.begin(), distances.end(), [iso_value](auto x) { return x < iso_value; }));

                float gradient = 0.0f;
                for (int l = max_k; l < zero_element - 1; ++l) {
                    gradient += (distances[l + 1] - distances[l]);
                }
                gradient /= ((zero_element -1) - max_k);
                auto predicted_k = -(distances[max_k]) / gradient;


                // if (n == 0) {
                    _vertices[_slice_ellipsoid[n][i]][0] = new_start.x - direction.x * new_sample_step * predicted_k;
                    _vertices[_slice_ellipsoid[n][i]][1] = new_start.y - direction.y * new_sample_step * predicted_k;
                    _vertices[_slice_ellipsoid[n][i]][2] = new_start.z - direction.z * new_sample_step * predicted_k;
                //} else {
                //// DEBUG center of mass translation
                //     _vertices[_slice_ellipsoid[n][i]][0] = vertex[0];
                //     _vertices[_slice_ellipsoid[n][i]][1] = vertex[1];
                //     _vertices[_slice_ellipsoid[n][i]][2] = vertex[2];
                //}
            }
        }

        // Push new positions in surface mesh
        //for (int i = 0; i < _sm.num_vertices(); ++i) {
        //    auto it = std::next(_sm.points().begin(), i);

        //    const Point point(_vertices[i][0], _vertices[i][1], _vertices[i][2]);
        //    *it = point;
        //}

        typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
        typedef K::Point_3 Point_3;
        std::list<Point_3> points_for_triangulation;

        for (int i = 0; i < _vertices.size(); ++i) {
            points_for_triangulation.push_back(Point(_vertices[i][0], _vertices[i][1], _vertices[i][2]));
        }

        typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
        typedef Delaunay::Vertex_handle Vertex_handle;
        typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
        //void CGAL::facets_in_complex_2_to_triangle_mesh(c2t3, TriangleMesh & graph) 	

        Delaunay T(points_for_triangulation.begin(), points_for_triangulation.end());
        _sm.clear();
        CGAL::convex_hull_3_to_face_graph(T, _sm);

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

    void ReconstructSurface::do_remeshing(Surface_mesh& mesh, float spacing_) {

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

    void ReconstructSurface::generateNormals(Surface_mesh& mesh) {

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
            auto it = std::next(mesh.points().begin(),i);
            points.emplace_back(std::make_pair(*it,Vector(0,0,0)));
        }

        int nb_neighbors = 24;

        try {
        CGAL::pca_estimate_normals<CGAL::Parallel_if_available_tag>(points, nb_neighbors,
            CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
                .normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));
        } catch (const std::exception& e) {
            core::utility::log::Log::DefaultLog.WriteError(std::string("[ReconstructSurface] NormalEstimation exited with ").append(e.what()).c_str());
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

             if (glm::dot(normal, difcenter) > 0) {
                normal *= -1;
            }

            _normals[i] = {normal.x, normal.y, normal.z};
        }
#if 0
        _normals.resize(_vertices.size());
        for (int i = 0; i < _vertices.size(); ++i) {
            glm::vec3 vert0 = {_vertices[i][0], _vertices[i][1], _vertices[i][2]};
            glm::vec3 normal = vert0 - _data_origin;

            normal = glm::normalize(normal);

            _normals[i] = {normal.x, normal.y, normal.z};
        }
#endif

        
    }

    void ReconstructSurface::generateNormals_2(Surface_mesh& mesh, std::vector<std::array<float,3>>& normals) {

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
            std::array<float,3> normal = {vnormals[vd].x(), vnormals[vd].y(), vnormals[vd].z()};
            normals.emplace_back(normal);
        }

    }

    //void ReconstructSurface::isotropicRemeshing() {
    //    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    //    typedef K::Point_3 Point;
    //    typedef CGAL::Surface_mesh<Point> Mesh;
    //    typedef boost::graph_traits<Mesh>::halfedge_descriptor halfedge_descriptor;
    //    typedef boost::graph_traits<Mesh>::edge_descriptor edge_descriptor;
    //    namespace PMP = CGAL::Polygon_mesh_processing;
    //    struct halfedge2edge {
    //        halfedge2edge(const Mesh& m, std::vector<edge_descriptor>& edges) : m_mesh(m), m_edges(edges) {}
    //        void operator()(const halfedge_descriptor& h) const {
    //            m_edges.push_back(edge(h, m_mesh));
    //        }
    //        const Mesh& m_mesh;
    //        std::vector<edge_descriptor>& m_edges;
    //    };

    //    Mesh mesh;
    //    for (int i = 0; i < _vertices.size(); ++i) {
    //        Point p = {_vertices[i][0], _vertices[i][1], _vertices[i][2]};
    //        mesh.add_vertex(p);
    //    }
    //    for (int j = 0; j < _triangles.size(); ++j) {

    //        auto i0 = static_cast<CGAL::Surface_mesh<Point>::Vertex_index>(_triangles[j][0]);
    //        auto i1 = static_cast<CGAL::Surface_mesh<Point>::Vertex_index>(_triangles[j][1]);
    //        auto i2 = static_cast<CGAL::Surface_mesh<Point>::Vertex_index>(_triangles[j][2]);
    //        mesh.add_face(i0, i1, i2);
    //    }

    //    double target_edge_length = 0.04;
    //    unsigned int nb_iter = 3;


    //    std::vector<edge_descriptor> border;
    //    PMP::border_halfedges(faces(mesh), mesh, boost::make_function_output_iterator(halfedge2edge(mesh, border)));
    //    PMP::split_long_edges(border, target_edge_length, mesh);

    //    PMP::isotropic_remeshing(faces(mesh), target_edge_length, mesh,
    //        PMP::parameters::number_of_iterations(nb_iter).protect_constraints(true) // i.e. protect border, here
    //    );

    //}

    void ReconstructSurface::onionize() {
        unsigned int num_shells = _numShellsSlot.Param<core::param::IntParam>()->Value();
        unsigned int num_splits_main_axis = 10;
        unsigned int num_splits_off_axis = 2;

        // translate vertices back to center
        for (int i = 0; i < _sm.num_vertices(); ++i) {
            auto it = std::next(_sm.points().begin(), i);

            glm::vec3 p = {it->x(), it->y(), it->z()};
            p.x -= _data_origin[0];
            p.y -= _data_origin[1];
            p.z -= _data_origin[2];

            const Point point(p.x, p.y, p.z);
            *it = point;
        }
        glm::vec3 whd = {
            this->_bbox.BoundingBox().Width(), this->_bbox.BoundingBox().Height(), this->_bbox.BoundingBox().Depth()};
        
        float min_length = std::min(whd.x, std::min(whd.y,whd.z));
        const float step_size = (0.5f*min_length) / (num_shells+1);

        //glm::vec3 scale_map = {0,0,0};
        //std::array<float, 3> whd = {
        //    _bbox.BoundingBox().Width(), _bbox.BoundingBox().Height(), _bbox.BoundingBox().Depth()};
        //auto min_d = std::min_element(whd.begin(), whd.end());

        //for (int i = 0; i < whd.size(); ++i) {
        //    scale_map[i] = 1.0f / (min_length / whd[i]);
        //}
        //scale_map = glm::normalize(scale_map);

        core::BoundingBoxes_2 origin_box;
        origin_box
            .SetBoundingBox(_bbox.BoundingBox().Left() - _data_origin[0],
                _bbox.BoundingBox().Bottom() - _data_origin[1],
                _bbox.BoundingBox().Back() - _data_origin[2],
                _bbox.BoundingBox().Right() - _data_origin[0],
                _bbox.BoundingBox().Top() - _data_origin[1],
                _bbox.BoundingBox().Front() - _data_origin[2]);

        float avg_spacing = 0.0f;

        // build initial shells
        _shells.clear();
        _shells.resize(num_shells);
        _scaledHulls.clear();
        _scaledHulls.resize(num_shells);
        _shellBBoxes.clear();
        _shellBBoxes.resize(num_shells);
        _scaledHulls[0] = _sm;
        _shellBBoxes[0].SetBoundingBox(_bbox.BoundingBox());
        for (int i = 1; i < num_shells; ++i) {
            _scaledHulls[i] = _sm;
            //_scaledHulls[i] = _scaledHulls[i-1];
            glm::vec3 scale = glm::vec3(1);
            for (int k = 0; k < 3; ++k) {
                scale[k] = (0.5f * whd[k] - i*step_size) / (0.5f*whd[k]);
            }
            //if (i > 0) {
            //    auto inv_scale = 1.0f - scale;
            //    inv_scale *= 1.0f / std::powf(static_cast<float>(i), 1.0f/3.0f);
            //    scale = 1.0f - inv_scale;
            //    // relative shell thickness stays the same
            //    // but in absolute values: shells get thinner
            //}
            // also scale bounding box
            float xmin = std::numeric_limits<float>::max();
            float xmax = -std::numeric_limits<float>::max();
            float ymin = std::numeric_limits<float>::max();
            float ymax = -std::numeric_limits<float>::max();
            float zmin = std::numeric_limits<float>::max();
            float zmax = -std::numeric_limits<float>::max();

            // scale hull
            this->generateNormals(_scaledHulls[i]);
            assert(_scaledHulls[i].num_vertices() == _normals.size());
            for (int j = 0; j < _scaledHulls[i].num_vertices(); ++j) {
                auto it = std::next(_scaledHulls[i].points().begin(), j);

                glm::vec3 n(_normals[j][0], _normals[j][1], _normals[j][2]);
                glm::vec3 p(it->x(), it->y(), it->z());
                p.x = p.x * scale.x + _data_origin[0];
                p.y = p.y * scale.y +_data_origin[1];
                p.z = p.z * scale.z + _data_origin[2];
                //p += _data_origin;

                // resize with normal
                // p += n * step_size;

                xmin = std::min(xmin, p.x);
                xmax = std::max(xmax, p.x);
                ymin = std::min(ymin, p.y);
                ymax = std::max(ymax, p.y);
                zmin = std::min(zmin, p.z);
                zmax = std::max(zmax, p.z);

                const Point point(p.x, p.y, p.z);
                *it = point;
            }
            _shellBBoxes[i].SetBoundingBox(xmin,ymin,zmax,xmax,ymax,zmin);

            // generate shell
            try {
                 //const Point point(_data_origin[0], _data_origin[1], _data_origin[2]);
                 //const Vector normal(1, 0, 0);
                 //const Plane plane(point, normal);


                bool si1 = CGAL::Polygon_mesh_processing::does_self_intersect(_scaledHulls[i - 1]);
                if (si1) {
                    this->remove_self_intersections(_scaledHulls[i - 1]);
                }
                bool si2 = CGAL::Polygon_mesh_processing::does_self_intersect(_scaledHulls[i]);
                if (si2) {
                    this->remove_self_intersections(_scaledHulls[i]);
                }

                Surface_mesh tm1 = _scaledHulls[i - 1];
                Surface_mesh tm2 = _scaledHulls[i];
                CGAL::Polygon_mesh_processing::corefine_and_compute_difference(
                    tm1, tm2, _shells[i - 1], CGAL::Polygon_mesh_processing::parameters::throw_on_self_intersection(true));

                //bool does_self_intersect = CGAL::Polygon_mesh_processing::does_self_intersect(_shells[i - 1]);
                //if (does_self_intersect) {

                //}
            } catch (const std::exception& e) {
                core::utility::log::Log::DefaultLog.WriteError(
                    std::string("[ReconstructSurface] Shell generation exited with ").append(e.what()).c_str());
            }
        }

        _shells[num_shells - 1] = _scaledHulls[num_shells - 1];
        assert(_shells.size() == num_shells);
    }

    void ReconstructSurface::cut() {
        typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
        typedef K::Plane_3 Plane;
        typedef K::Vector_3 Vector;

        //Point lbf = Point(_bbox.BoundingBox().Left(), _bbox.BoundingBox().Bottom(), _bbox.BoundingBox().Front());
        //Point rtb = Point(_bbox.BoundingBox().Right(), _bbox.BoundingBox().Top(), _data_origin[2]);

        //Point p0 = Point(lbf[0], lbf[1], lbf[2]);
        //Point p1 = Point(rtb[0], lbf[1], lbf[2]);
        //Point p2 = Point(lbf[0], lbf[1], rtb[2]);
        //Point p3 = Point(lbf[0], rtb[1], lbf[2]);

        //Point p4 = Point(rtb[0], lbf[1], rtb[2]);
        //Point p5 = Point(lbf[0], rtb[1], rtb[2]);
        //Point p6 = Point(rtb[0], rtb[1], lbf[2]);
        //Point p7 = Point(rtb[0], rtb[1], rtb[2]);

        //Surface_mesh cube;
        //CGAL::make_hexahedron(p0,p1,p2,p3,p4,p5,p6,p7,cube);
        //CGAL::Polygon_mesh_processing::triangulate_faces(cube);
        //CGAL::Polygon_mesh_processing::isotropic_remeshing(cube.faces(), (*min_d/10.0f), cube);
        //bool is_cube_triangle_mesh = CGAL::is_triangle_mesh(cube);
        //bool cube_does_self_intersect = CGAL::Polygon_mesh_processing::does_self_intersect(cube);
        //if (cube_does_self_intersect) {
        //    std::vector<std::pair<face_descriptor, face_descriptor>> intersected_tris;
        //    PMP::self_intersections(cube, std::back_inserter(intersected_tris));
        //    for (std::pair<face_descriptor, face_descriptor>& p : intersected_tris) {
        //        CGAL::Euler::remove_face(cube.halfedge(get<0>(p)), cube);
        //    }
        //}
        
        // bool is_triangle_mesh = CGAL::is_triangle_mesh(_shells[i]);
        // bool does_self_intersect = CGAL::Polygon_mesh_processing::does_self_intersect(_shells[i]);

        const int splits_main_axis = _shellSplitsAxis.Param<core::param::IntParam>()->Value();
        const int splits_phi = _shellSplitsAngle.Param<core::param::IntParam>()->Value();
        const float phi_step = glm::pi<float>() / (static_cast<float>(splits_phi)/2.0f);

        auto main_axis_normal0 = glm::vec3(0,0,0);
        auto main_axis_normal1 = glm::vec3(0, 0, 0);
        main_axis_normal0[_main_axis] = -1;
        main_axis_normal1[_main_axis] = 1;

        _shellElements.resize(_shells.size());
        for (int i = 0; i < _shells.size(); ++i) {
            _shellElements[i].reserve(splits_main_axis * splits_phi);

            // shell box values
            auto shell_lbf = glm::vec3(_shellBBoxes[i].BoundingBox().Left(), _shellBBoxes[i].BoundingBox().Bottom(),
                _shellBBoxes[i].BoundingBox().Front());
            auto shell_rtb = glm::vec3(_shellBBoxes[i].BoundingBox().Right(), _shellBBoxes[i].BoundingBox().Top(),
                _shellBBoxes[i].BoundingBox().Back());
            auto shell_whd = glm::vec3(_shellBBoxes[i].BoundingBox().Width(), _shellBBoxes[i].BoundingBox().Height(),
                _shellBBoxes[i].BoundingBox().Depth());
            auto shell_main_axis_step = glm::vec3(0);
            shell_main_axis_step[_main_axis] = shell_whd[_main_axis] / splits_main_axis;

            for (int n = 0; n < splits_main_axis; ++n) {
            //for (int n = 0; n < 1; ++n) {
                auto shell = _shells[i];
                // split in main axis direction
                auto p0 = shell_lbf;
                p0 += static_cast<float>(n)*shell_main_axis_step * main_axis_normal1;
                const Point point0(p0.x, p0.y, p0.z);
                Vector normal0(main_axis_normal0[0], main_axis_normal0[1], main_axis_normal0[2]);
                const Plane plane0(point0, normal0);

                auto p1 = shell_lbf;
                p1 += static_cast<float>(n+1) * shell_main_axis_step * main_axis_normal1;
                const Point point1(p1.x, p1.y, p1.z);
                Vector normal1(main_axis_normal1[0], main_axis_normal1[1], main_axis_normal1[2]);
                const Plane plane1(point1, normal1);
                try {
                    bool si = CGAL::Polygon_mesh_processing::does_self_intersect(shell);
                    if (si) {
                        this->remove_self_intersections(shell);
                    }
                    CGAL::Polygon_mesh_processing::clip(
                        shell, plane0, CGAL::Polygon_mesh_processing::parameters::clip_volume(true).throw_on_self_intersection(true));
                    shell.collect_garbage();

                    si = CGAL::Polygon_mesh_processing::does_self_intersect(shell);
                    if (si) {
                        this->remove_self_intersections(shell);
                    }
                    CGAL::Polygon_mesh_processing::clip(
                        shell, plane1, CGAL::Polygon_mesh_processing::parameters::clip_volume(true).throw_on_self_intersection(true));
                    shell.collect_garbage();

                } catch (const std::exception& e) {
                    core::utility::log::Log::DefaultLog.WriteError(
                        std::string("[ReconstructSurface] Element generation exited with ").append(e.what()).c_str());
                }


                glm::vec3 element_com(0,0,0);
                for (auto p : shell.points()) {
                    element_com.x += p.x();
                    element_com.y += p.y();
                    element_com.z += p.z();
                }
                element_com /= shell.num_vertices();

                auto p_origin = Point(element_com.x, element_com.y, element_com.z);
                glm::vec3 n2(0);
                n2[_off_axes[0]] = 1;
                auto rot_mx = get_rot_mx(phi_step);
                // DEBUG
                //_shellElements[i].emplace_back(shell);

                for (int k = 0; k < splits_phi; ++k) {
                    auto shell_copy = shell;
                    if (k > 0) {
                        n2 = rot_mx[_main_axis] * n2;
                    }
                    auto n3 = -rot_mx[_main_axis] * n2;

                    const Plane plane2(p_origin, Vector(n2.x,n2.y,n2.z));
                    const Plane plane3(p_origin, Vector(n3.x, n3.y, n3.z));

                    try {
                        bool si = CGAL::Polygon_mesh_processing::does_self_intersect(shell_copy);
                        if (si) {
                            this->remove_self_intersections(shell_copy);
                        }
                        CGAL::Polygon_mesh_processing::clip(
                            shell_copy, plane2, CGAL::Polygon_mesh_processing::parameters::clip_volume(true).throw_on_self_intersection(true));
                        shell_copy.collect_garbage();

                        si = CGAL::Polygon_mesh_processing::does_self_intersect(shell_copy);
                        if (si) {
                            this->remove_self_intersections(shell_copy);
                        }
                        CGAL::Polygon_mesh_processing::clip(
                            shell_copy, plane3, CGAL::Polygon_mesh_processing::parameters::clip_volume(true).throw_on_self_intersection(true));
                        shell_copy.collect_garbage();
                    } catch (const std::exception& e) {
                        core::utility::log::Log::DefaultLog.WriteError(
                            std::string("[ReconstructSurface] Element generation exited with ")
                                .append(e.what()).c_str());
                    }
                    shell_copy.collect_garbage();
                    _shellElements[i].emplace_back(shell_copy);
                }
            }
        }

        // fill shell elements vectors
        _shellElementsVertices.clear();
        _shellElementsVertices.resize(_shellElements.size());
        _shellElementsTriangles.clear();
        _shellElementsTriangles.resize(_shellElements.size());
        _shellElementsNormals.clear();
        _shellElementsNormals.resize(_shellElements.size());

        for (int i = 0; i < _shellElements.size(); ++i) {
            _shellElementsVertices[i].resize(_shellElements[i].size());
            _shellElementsTriangles[i].resize(_shellElements[i].size());
            _shellElementsNormals[i].resize(_shellElements[i].size());
            for (int j = 0; j < _shellElements[i].size(); ++j) {
                this->generateNormals_2(_shellElements[i][j], _shellElementsNormals[i][j]);
                this->activateMesh(_shellElements[i][j], _shellElementsVertices[i][j], _shellElementsTriangles[i][j]);
            }
        }



    }

    void ReconstructSurface::remove_self_intersections(Surface_mesh& mesh_) {
        typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
        namespace PMP = CGAL::Polygon_mesh_processing;
        namespace params = CGAL::Polygon_mesh_processing::parameters;
        typedef boost::graph_traits<Surface_mesh>::face_descriptor face_descriptor;
        typedef boost::graph_traits<Surface_mesh>::vertex_descriptor vertex_descriptor;
        auto mesh = mesh_; // copy
        try {
            bool intersect = CGAL::Polygon_mesh_processing::does_self_intersect(mesh);
            for (int i = 0; i < 10; ++i) {
                if (!intersect) break;
                std::vector<std::pair<face_descriptor, face_descriptor>> intersected_tris;
                PMP::self_intersections(mesh, std::back_inserter(intersected_tris));
                //std::set<face_descriptor> s(intersected_tris.begin(), intersected_tris.end());
                //intersected_tris.assign(s.begin(), s.end());
                std::sort(intersected_tris.begin(), intersected_tris.end());
                intersected_tris.erase(
                    std::unique(intersected_tris.begin(), intersected_tris.end()), intersected_tris.end());
                std::vector<face_descriptor> removed_list;
                for (std::pair<face_descriptor, face_descriptor>& p : intersected_tris) {

                    if (removed_list.empty() ||
                        std::find(removed_list.begin(), removed_list.end(), p.first) == removed_list.end()) {
                        auto he1 = mesh.halfedge(p.first);
                        CGAL::Euler::remove_face(he1, mesh);
                        removed_list.emplace_back(p.first);
                    }
                    //if (removed_list.empty() ||
                    //    std::find(removed_list.begin(), removed_list.end(), p.second) == removed_list.end()) {
                    //    auto he2 = mesh.halfedge(p.second);
                    //    CGAL::Euler::remove_face(he2, mesh);
                    //    removed_list.emplace_back(p.second);
                    //}
                }
                for (auto h : halfedges(mesh)) {
                    if (CGAL::is_border(h, mesh)) {
                        std::vector<face_descriptor> patch_facets;
                        std::vector<vertex_descriptor> patch_vertices;
                        bool success = std::get<0>(PMP::triangulate_refine_and_fair_hole(mesh, h,
                            std::back_inserter(patch_facets), std::back_inserter(patch_vertices),
                            CGAL::parameters::vertex_point_map(get(CGAL::vertex_point, mesh)).geom_traits(Kernel())));
                        if (!success) break;
                    }
                }
                mesh.collect_garbage();
                intersect = CGAL::Polygon_mesh_processing::does_self_intersect(mesh);
            }
            
            if (!intersect) {
                mesh_ = mesh;
                return;
            }
            // do shape smoothing
            for (int i = 0; i < 10; ++i) {
                if (!intersect) break;
                this->do_smoothing(mesh_);
                intersect = CGAL::Polygon_mesh_processing::does_self_intersect(mesh_);
            }
            if (!intersect) return;

            // do remeshing
            //float avg_spacing = CGAL::compute_average_spacing<CGAL::Parallel_if_available_tag>(mesh_.points(), 6);
            //this->do_remeshing(mesh_, avg_spacing);
            //intersect = CGAL::Polygon_mesh_processing::does_self_intersect(mesh_);
            //if (!intersect) return;

            // delete bad vertices and make new triangulation
            std::vector<std::pair<face_descriptor, face_descriptor>> intersected_tris;
            PMP::self_intersections(mesh_, std::back_inserter(intersected_tris));
            std::sort(intersected_tris.begin(), intersected_tris.end());
            intersected_tris.erase(
                std::unique(intersected_tris.begin(), intersected_tris.end()), intersected_tris.end());
            std::vector<face_descriptor> removed_list;
            for (std::pair<face_descriptor, face_descriptor>& p : intersected_tris) {
                if (removed_list.empty() || std::find(removed_list.begin(), removed_list.end(), p.first) == removed_list.end()) {
                    auto he = mesh_.halfedge(p.first);
                    auto index = CGAL::target(he, mesh_);
                    mesh_.remove_vertex(index);
                    removed_list.emplace_back(p.first);
                }
            }
            mesh_.collect_garbage();

            typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
            typedef K::Point_3 Point_3;
            std::list<Point_3> points_for_triangulation;

            for (auto point: mesh_.points()) {
                points_for_triangulation.push_back(Point(point.x(), point.y(), point.z()));
            }

            typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
            typedef Delaunay::Vertex_handle Vertex_handle;
            typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
            // void CGAL::facets_in_complex_2_to_triangle_mesh(c2t3, TriangleMesh & graph)

            Delaunay T(points_for_triangulation.begin(), points_for_triangulation.end());
            mesh_.clear();
            CGAL::convex_hull_3_to_face_graph(T, mesh_);

            intersect = CGAL::Polygon_mesh_processing::does_self_intersect(mesh_);
            if (intersect) {
                float avg_spacing = CGAL::compute_average_spacing<CGAL::Parallel_if_available_tag>(mesh_.points(), 6);
                this->do_remeshing(mesh_, avg_spacing);
            } else {
                return;
            }
            intersect = CGAL::Polygon_mesh_processing::does_self_intersect(mesh_);
            if (intersect) {
                this->do_smoothing(mesh_);
            } else {
                return;
            }
            intersect = CGAL::Polygon_mesh_processing::does_self_intersect(mesh_);

        } catch (const std::exception& e) {
            core::utility::log::Log::DefaultLog.WriteError(
                std::string("[ReconstructSurface] Remove self-intersections exited with ").append(e.what()).c_str());
        }
    }

    void ReconstructSurface::do_smoothing(Surface_mesh& mesh) {
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

    void ReconstructSurface::compute() {
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
            center_z = _bbox.BoundingBox().Front() +
                       std::copysign(1.0f, _bbox.BoundingBox().Front()) * _bbox.BoundingBox().Depth() / 2;
        }
        _data_origin = {_bbox.BoundingBox().CalcCenter().GetX(), _bbox.BoundingBox().CalcCenter().GetY(), center_z};


        this->generateEllipsoid_3();
        this->sliceData();
        this->tighten();
        this->do_remeshing(_sm);
        this->do_smoothing(_sm);

        this->generateNormals(_sm);
        this->onionize();
        this->cut();

    }

    bool ReconstructSurface::processRawData(adios::CallADIOSData* call, bool& something_changed) {
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
            if (!call->inquire(var)) {
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
                _raw_positions.resize(x.size() * 3);
                for (int i = 0; i < x.size(); ++i) {
                    _raw_positions[3 * i + 0] = x[i];
                    _raw_positions[3 * i + 1] = y[i];
                    _raw_positions[3 * i + 2] = z[i];
                }
                auto xminmax = std::minmax_element(x.begin(), x.end());
                auto yminmax = std::minmax_element(y.begin(), y.end());
                auto zminmax = std::minmax_element(z.begin(), z.end());
                _bbox.SetBoundingBox(
                    *xminmax.first, *yminmax.first, *zminmax.second, *xminmax.second, *yminmax.second, *zminmax.first);
            } else {
                const std::string varname = std::string(_xyzSlot.Param<core::param::FlexEnumParam>()->ValueString());
                _raw_positions = call->getData(varname)->GetAsFloat();
                float xmin = std::numeric_limits<float>::max();
                float xmax = -std::numeric_limits<float>::max();
                float ymin = std::numeric_limits<float>::max();
                float ymax = -std::numeric_limits<float>::max();
                float zmin = std::numeric_limits<float>::max();
                float zmax = -std::numeric_limits<float>::max();
                for (int i = 0; i < _raw_positions.size() / 3; ++i) {
                    xmin = std::min(xmin, _raw_positions[3 * i + 0]);
                    xmax = std::max(xmax, _raw_positions[3 * i + 0]);
                    ymin = std::min(ymin, _raw_positions[3 * i + 1]);
                    ymax = std::max(ymax, _raw_positions[3 * i + 1]);
                    zmin = std::min(zmin, _raw_positions[3 * i + 2]);
                    zmax = std::max(zmax, _raw_positions[3 * i + 2]);
                }
                _bbox.SetBoundingBox(xmin, ymin, zmax, xmax, ymax, zmin);
            }
        }
        return true;
    }


    bool ReconstructSurface::getData(core::Call& call) {

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

        if (something_changed && !_raw_positions.empty() || _shellToShowChanged) {

            auto mfdc = this->_meshFromDiscCall.CallAs<adios::CallADIOSData>();
            if (!mfdc) {
                if (!_shellToShowChanged || _shells.empty()) {
                    this->compute();
                }
            } else {
                if (!_shellToShowChanged || _shells.empty()) {
                    if (!this->readMeshElementsFromFile()) {
                        core::utility::log::Log::DefaultLog.WriteError("[ReconstructSurface] Could not load mesh from File. Starting computation ...");
                        this->compute();
                    }
                }
            }

            auto shell = _showShellSlot.Param<core::param::IntParam>()->Value();
            if (shell > -1) {
                if (this->_meshOutputSlot.Param<core::param::EnumParam>()->Value() == 2) {
                    int shell_i = shell / _shellElements[0].size();
                    int shell_j = shell % _shellElements[0].size();
                    this->activateMesh(_shellElements[shell_i][shell_j], _vertices, _triangles);
                    this->generateNormals(_shellElements[shell_i][shell_j]);
                } else if (this->_meshOutputSlot.Param<core::param::EnumParam>()->Value() == 1) {
                    shell = shell % _shells.size();
                    this->activateMesh(_shells[shell], _vertices, _triangles);
                    this->generateNormals(_shells[shell]);
                }
            }

            if (this->_meshOutputSlot.Param<core::param::EnumParam>()->Value() == 0) {
                _elementMesh.clear();
                _elementMesh.resize(_shellElements.size());
                for (int i = 0; i < _shellElements.size(); ++i) {
                    _elementMesh[i].resize(_shellElements[i].size());
                    for (int j = 0; j < _shellElements[i].size(); ++j) {
                        _elementMesh[i][j].second.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute());
                        _elementMesh[i][j].second.back().component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
                        _elementMesh[i][j].second.back().byte_size = _shellElementsVertices[i][j].size() * sizeof(std::array<float, 3>);
                        _elementMesh[i][j].second.back().component_cnt = 3;
                        _elementMesh[i][j].second.back().stride = sizeof(std::array<float, 3>);
                        _elementMesh[i][j].second.back().offset = 0;
                        _elementMesh[i][j].second.back().data =
                            reinterpret_cast<uint8_t*>(_shellElementsVertices[i][j].data());
                        _elementMesh[i][j].second.back().semantic = mesh::MeshDataAccessCollection::POSITION;

                        if (!_normals.empty()) {
                            _elementMesh[i][j].second.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute());
                            _elementMesh[i][j].second.back().component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
                            _elementMesh[i][j].second.back().byte_size =
                                _shellElementsNormals[i][j].size() * sizeof(std::array<float, 3>);
                            _elementMesh[i][j].second.back().component_cnt = 3;
                            _elementMesh[i][j].second.back().stride = sizeof(std::array<float, 3>);
                            _elementMesh[i][j].second.back().offset = 0;
                            _elementMesh[i][j].second.back().data =
                                reinterpret_cast<uint8_t*>(_shellElementsNormals[i][j].data());
                            _elementMesh[i][j].second.back().semantic = mesh::MeshDataAccessCollection::NORMAL;
                        }

                        _elementMesh[i][j].first.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
                        _elementMesh[i][j].first.byte_size = _shellElementsTriangles[i][j].size() * sizeof(std::array<uint32_t, 3>);
                        _elementMesh[i][j].first.data =
                            reinterpret_cast<uint8_t*>(_shellElementsTriangles[i][j].data());
                    }
                }
            } else {
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
                }
            ++_version;
        } // something changed

        // put data in mesh
        if (this->_meshOutputSlot.Param<core::param::EnumParam>()->Value() == 0) {
            mesh::MeshDataAccessCollection mesh;
            for (int i = 0; i < _shellElements.size(); ++i) {
                 for (int j = 0; j < _shellElements[i].size(); ++j) {
                    std::string identifier = std::string(FullName()) + "_mesh_" + std::to_string(i) + "," + std::to_string(j);
                    mesh.addMesh(identifier, _elementMesh[i][j].second, _elementMesh[i][j].first,
                    mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES);
                 }
            }
            cm->setData(std::make_shared<mesh::MeshDataAccessCollection>(std::move(mesh)), _version);
        } else {
            mesh::MeshDataAccessCollection mesh;
            std::string identifier = std::string(FullName()) + "_mesh";
            mesh.addMesh(
                identifier, _mesh.second, _mesh.first, mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES);
            cm->setData(std::make_shared<mesh::MeshDataAccessCollection>(std::move(mesh)), _version);
        }
        _old_datahash = cd->getDataHash();
        _recalc = false;
        _shellToShowChanged = false;

        auto meta_data = cm->getMetaData();
        meta_data.m_bboxs = _bbox;
        cm->setMetaData(meta_data);

        return true;
    }

    bool ReconstructSurface::getADIOSMetaData(core::Call& call) {
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

    bool ReconstructSurface::getADIOSData(core::Call& call) {
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

        if (something_changed && !_raw_positions.empty()) {
            this->compute();
        }

        assert(!_shellElements.empty());
        assert(!_shellElements[0].empty());

        auto elementsCont = std::make_shared<adios::CharContainer>(adios::CharContainer());
        auto elementsOffsetCont = std::make_shared<adios::UInt64Container>(adios::UInt64Container());
        elementsOffsetCont->shape = {_shellElements.size(), _shellElements[0].size()};
        std::vector<char>& elements_vec = elementsCont->getVec();
        std::vector<uint64_t>& elementOffsets_vec = elementsOffsetCont->getVec();

        elementOffsets_vec.reserve(_shellElements.size() * _shellElements[0].size());
        uint64_t current_element_offset = 0;
        for (int i = 0; i < _shellElements.size(); ++i) {
            for (int j = 0; j < _shellElements[0].size(); ++j) {
                std::stringstream sstream;
                sstream << _shellElements[i][j];
                auto current_string = sstream.str();
                std::copy(current_string.begin(), current_string.end(), std::back_inserter(elements_vec));
                elementOffsets_vec.emplace_back(current_element_offset);
                current_element_offset += current_string.size();
            }
        }

        auto shellsCont = std::make_shared<adios::CharContainer>(adios::CharContainer());
        auto shellsOffsetCont = std::make_shared<adios::UInt64Container>(adios::UInt64Container());
        std::vector<char>& shells_vec = shellsCont->getVec();
        std::vector<uint64_t>& shells_offsets_vec = shellsOffsetCont->getVec();
        shells_offsets_vec.reserve(_shells.size());
        uint64_t current_shell_offset = 0;
        for (int i = 0; i < _shells.size(); ++i) {
            std::stringstream sstream;
            sstream << _shells[i];
            auto current_string = sstream.str();
            std::copy(current_string.begin(), current_string.end(), std::back_inserter(shells_vec));
            shells_offsets_vec.emplace_back(current_shell_offset);
            current_shell_offset += current_string.size();
        }

        auto hullCont = std::make_shared<adios::CharContainer>(adios::CharContainer());
        std::vector<char>& hull_vec = hullCont->getVec();

        std::stringstream hull_sstream;
        hull_sstream << _sm;
        auto hull_str = hull_sstream.str();

        hull_vec.reserve(hull_str.size());
        std::copy(hull_str.begin(), hull_str.end(), std::back_inserter(hull_vec));
        
        auto boxCont = std::make_shared<adios::FloatContainer>(adios::FloatContainer());
        std::vector<float>& box_vec = boxCont->getVec();
        box_vec = {_bbox.BoundingBox().Left(), _bbox.BoundingBox().Bottom(), _bbox.BoundingBox().Back(),
                   _bbox.BoundingBox().Right(), _bbox.BoundingBox().Top(), _bbox.BoundingBox().Front()};


        _dataMap["hull"] = std::move(hullCont);
        _dataMap["elements"] = std::move(elementsCont);
        _dataMap["elements_offsets"] = std::move(elementsOffsetCont);
        _dataMap["shells"] = std::move(shellsCont);
        _dataMap["shells_offsets"] = std::move(shellsOffsetCont);
        _dataMap["bbox"] = std::move(boxCont);

        out->setData(std::make_shared<adios::adiosDataMap>(_dataMap));
        out->setDataHash(cd->getDataHash());

        _old_datahash = cd->getDataHash();
        _recalc = false;

        return true;

    }

    bool ReconstructSurface::readMeshElementsFromFile() {

        auto cd = _meshFromDiscCall.CallAs<adios::CallADIOSData>();
        if (cd == nullptr)
            return false;

        cd->setFrameIDtoLoad(0); // TODO: maybe support more frames in the future
        if (!(*cd)(1))
            return false;

        auto vars = cd->getAvailableVars();

        // get data from adios
        for (auto var : vars) {
            if (!cd->inquire(var)) {
                core::utility::log::Log::DefaultLog.WriteError((std::string("[ReconstructSurface] Could not inquire ") + var).c_str());
                return false;
            }
        }

        if (!(*cd)(0))
            return false;

        auto hull = cd->getData("hull")->GetAsChar();
        std::string hull_str(hull.begin(), hull.end());
        _sm.clear();
        std::stringstream(hull_str) >> _sm;

        auto shells = cd->getData("shells")->GetAsChar();
        auto shells_offsets = cd->getData("shells_offsets")->GetAsUInt64();
        _shells.clear();
        _shells.resize(shells_offsets.size());
        for (int i = 0; i < shells_offsets.size(); ++i) {
            std::string current_shell;
            if (i < shells_offsets.size() - 1) {
                current_shell = std::string(shells.begin() + shells_offsets[i], shells.begin() + shells_offsets[i+1]);
            } else {
                current_shell = std::string(shells.begin() + shells_offsets[i], shells.end());
            }
            std::stringstream(current_shell) >> _shells[i];
        }

        auto elements = cd->getData("elements")->GetAsChar();
        auto elements_offsets = cd->getData("elements_offsets")->GetAsUInt64();
        auto elements_shape = cd->getData("elements_offsets")->getShape();

        _shellElements.clear();
        _shellElements.resize(elements_shape[0]);
        _shellElementsVertices.clear();
        _shellElementsVertices.resize(_shellElements.size());
        _shellElementsTriangles.clear();
        _shellElementsTriangles.resize(_shellElements.size());
        _shellElementsNormals.clear();
        _shellElementsNormals.resize(_shellElements.size());
        for (int i = 0; i < elements_shape[0]; ++i) {
            _shellElements[i].resize(elements_shape[1]);
            _shellElementsVertices[i].resize(_shellElements[i].size());
            _shellElementsTriangles[i].resize(_shellElements[i].size());
            _shellElementsNormals[i].resize(_shellElements[i].size());
            for (int j = 0; j < elements_shape[1]; ++j) {
                std::string current_element;
                if ((i == (elements_shape[0] - 1)) && (j == (elements_shape[1] - 1))) {
                    current_element =
                        std::string(elements.begin() + elements_offsets[i * elements_shape[1] + j], elements.end());
                } else {
                    current_element = std::string(elements.begin() + elements_offsets[i * elements_shape[1] + j],
                        elements.begin() + elements_offsets[i * elements_shape[1] + j + 1]);
                }
                std::stringstream(current_element) >> _shellElements[i][j];
                this->generateNormals_2(_shellElements[i][j], _shellElementsNormals[i][j]);
                this->activateMesh(_shellElements[i][j], _shellElementsVertices[i][j], _shellElementsTriangles[i][j]);
            }
        }

        return true;
    }

    bool ReconstructSurface::getMetaData(core::Call& call) {

        auto cm = dynamic_cast<mesh::CallMesh*>(&call);
        if (cm == nullptr)
            return false;

        auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
        if (cd == nullptr)
            return false;

        auto meta_data = cm->getMetaData();

        // get metadata for volumetric call
        cd->setFrameIDtoLoad(meta_data.m_frame_ID);
        if (!(*cd)(1)) return false;
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

    bool ReconstructSurface::parameterChanged(core::param::ParamSlot& p) {
        _recalc = true;

        return true;
    }

    bool ReconstructSurface::shellToShowChanged(core::param::ParamSlot& p) {
        _shellToShowChanged = true;
        return true;
    }

    bool ReconstructSurface::getNormalData(core::Call& call) {
        bool something_changed = _recalc;

        auto mpd = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
        if (mpd == nullptr)
            return false;

        auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
        if (cd == nullptr)
            return false;

        if (!this->processRawData(cd, something_changed)) {
            core::utility::log::Log::DefaultLog.WriteError("Could not process incoming data. Abort.");
            return false;
        }

        if (something_changed && !_raw_positions.empty()) {
            this->compute();
        }

        // FOR DEBUG
        //std::vector<std::array<float, 4>> new_vec;
        //new_vec.reserve(112);
        //for (int i = _vertices.size() - 112; i < _vertices.size(); ++i) {
        //    new_vec.emplace_back(_vertices[i]);
        //}

        //_vertices = new_vec;

        //std::vector<std::array<float, 3>> new_norm;
        //new_vec.reserve(112);
        //for (int i = _normals.size() - 112; i < _normals.size(); ++i) {
        //    new_norm.emplace_back(_normals[i]);
        //}

        //_normals = new_norm;

        mpd->SetParticleListCount(1);
        mpd->AccessParticles(0).SetGlobalRadius(_bbox.BoundingBox().LongestEdge() * 1e-3);
        mpd->AccessParticles(0).SetCount(_vertices.size());
        mpd->AccessParticles(0).SetVertexData(core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
        _vertices.data(), sizeof(std::array<float,4>));
        mpd->AccessParticles(0).SetDirData(core::moldyn::MultiParticleDataCall::Particles::DIRDATA_FLOAT_XYZ,
        _normals.data(), sizeof(std::array<float,3>));

        _old_datahash = cd->getDataHash();
        _recalc = false;

        return true;
    }

    bool ReconstructSurface::getNormalMetaData(core::Call& call) {
        auto mpd = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
        if (mpd == nullptr)
            return false;

        auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
        if (cd == nullptr)
            return false;

        cd->setFrameIDtoLoad(mpd->FrameID());
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

        mpd->AccessBoundingBoxes().SetObjectSpaceBBox(_bbox.BoundingBox());
        mpd->SetFrameCount(cd->getFrameCount());
        cd->setFrameIDtoLoad(mpd->FrameID());
        mpd->SetDataHash(mpd->DataHash() + 1);


        return true;
    }

    void ReconstructSurface::activateMesh(const Surface_mesh& shell, std::vector<std::array<float,3>>& vertices, std::vector<std::array<uint32_t,3>>& indices) {

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
