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
#include "iterator"
// normals
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>
// 
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>

#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/IO/facets_in_complex_2_to_triangle_mesh.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Implicit_surface_3.h>

#include <CGAL/Plane_3.h>

namespace megamol {
namespace probe {

    // default triangulation for Surface_mesher
    typedef CGAL::Surface_mesh_default_triangulation_3 Tr;
    typedef Tr::Geom_traits GT;
    typedef GT::Point_3 Point;
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
            , _numSlices("numSlices", "")
            , _meshResolution("meshResolution", "")
            , _faceTypeSlot("FaceType", "")
            , _xSlot("x", "")
            , _ySlot("y", "")
            , _zSlot("z", "")
            , _xyzSlot("xyz", "")
            , _formatSlot("format", "")
            , _isoValue("isoValue", "")
            , _showShellSlot("showShell", "")
            , _numShellsSlot("numShells", "") {

        this->_numSlices << new core::param::IntParam(64);
        this->_numSlices.SetUpdateCallback(&ReconstructSurface::parameterChanged);
        this->MakeSlotAvailable(&this->_numSlices);

        this->_isoValue << new core::param::FloatParam(1.0f);
        this->_isoValue.SetUpdateCallback(&ReconstructSurface::parameterChanged);
        this->MakeSlotAvailable(&this->_isoValue);

        this->_meshResolution << new core::param::IntParam(64);
        this->_meshResolution.SetUpdateCallback(&ReconstructSurface::parameterChanged);
        this->MakeSlotAvailable(&this->_meshResolution);

        core::param::EnumParam* ep = new core::param::EnumParam(0);
        ep->SetTypePair(0, "Trianges");
        ep->SetTypePair(1, "Quads");
        this->_faceTypeSlot << ep;
        this->MakeSlotAvailable(&this->_faceTypeSlot);

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

        this->_showShellSlot << new core::param::IntParam(-1);
        this->_showShellSlot.SetUpdateCallback(&ReconstructSurface::parameterChanged);
        this->MakeSlotAvailable(&this->_showShellSlot);

        this->_numShellsSlot << new core::param::IntParam(10);
        this->_numShellsSlot.SetUpdateCallback(&ReconstructSurface::parameterChanged);
        this->MakeSlotAvailable(&this->_numShellsSlot);

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
    }

    ReconstructSurface::~ReconstructSurface() {
        this->Release();
    }

    bool ReconstructSurface::create() {
        return true;
    }

    void ReconstructSurface::release() {}

    bool ReconstructSurface::InterfaceIsDirty() {
        return this->_numSlices.IsDirty() || this->_formatSlot.IsDirty() || this->_meshResolution.IsDirty();
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
        auto slice_width = (_bbox.BoundingBox().GetSize()[_main_axis] + 1e-3) / (num_slices - 1); // otherwise were getting exact num_slices as factor

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
        const int num_ellipsoid_res_theta = this->_meshResolution.Param<core::param::IntParam>()->Value();
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
        _vertices.resize(_sm.num_vertices());
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
        }

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

    void ReconstructSurface::tighten() {

        // generate kd stuff
        _kd_indices.resize(_sliced_positions_whalo.size());
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
        for (int i = 0; i < _sm.num_vertices(); ++i) {
            auto it = std::next(_sm.points().begin(), i);

            const Point point(_vertices[i][0], _vertices[i][1], _vertices[i][2]);
            *it = point;
        }
    }

    void ReconstructSurface::generateNormals() {
        typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
        typedef Kernel::Point_3 Point;
        typedef Kernel::Vector_3 Vector;
        typedef std::pair<Point, Vector> PointVectorPair;

        std::list<PointVectorPair> points;
        for (int i = 0;i <_vertices.size(); ++i) {
            Point point = {_vertices[i][0], _vertices[i][1], _vertices[i][2]};
            Vector vector;
            PointVectorPair pvp = std::make_pair(point,vector);
            points.emplace_back(pvp);
        }

        int nb_neighbors = 18;

        CGAL::pca_estimate_normals<CGAL::Parallel_if_available_tag>(points, nb_neighbors,
            CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
                .normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));

        // Orientation of normals, returns iterator to first unoriented point
        //std::list<PointVectorPair>::iterator unoriented_points_begin = CGAL::mst_orient_normals(points, nb_neighbors,
        //    CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
        //        .normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));
        _normals.resize(_vertices.size());
        for (int i = 0; i < _vertices.size(); ++i) {
            auto it = std::next(points.begin(), i);
            glm::vec3 normal = {it->second.x(), it->second.y(), it->second.z()};
            glm::vec3 vert0 = {_vertices[i][0], _vertices[i][1], _vertices[i][2]};
            auto difcenter = vert0 - _data_origin;

             if (glm::dot(normal, difcenter) > 0) {
                normal *= -1;
            }

            _normals[i] = {normal.x, normal.y, normal.z};
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
        typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
        namespace PMP = CGAL::Polygon_mesh_processing;
        namespace params = CGAL::Polygon_mesh_processing::parameters;

        unsigned int num_shells = _numShellsSlot.Param<core::param::IntParam>()->Value();

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

        // build initial shells
        _shells.clear();
        _shells.resize(num_shells);
        for (int i = 0; i < num_shells; ++i) {
            _shells[i] = _sm;
            float scale_factor = 1 - i*(1.0f / static_cast<float>(num_shells));
            for (int j = 0; j < _shells[i].num_vertices(); ++j) {
                auto it = std::next(_shells[i].points().begin(), j);

                glm::vec3 p = {it->x(), it->y(), it->z()};
                p.x *= scale_factor;
                p.y *= scale_factor;
                p.z *= scale_factor;

                const Point point(p.x, p.y, p.z);
                *it = point;
            }
        }
        

    }


    bool ReconstructSurface::getData(core::Call& call) {

    bool something_changed = _recalc;

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr)
        return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr)
        return false;

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
        if (!cd->inquire(var)) {
            return false;
        }
    }

    if (!(*cd)(0)) return false;

    // get data from volumetric call
    if (cd->getDataHash() != _old_datahash) {
        something_changed = true;
        auto mesh_meta_data = cm->getMetaData();

        if (this->_formatSlot.Param<core::param::EnumParam>()->Value() == 0) {
            auto x = cd->getData(std::string(this->_xSlot.Param<core::param::FlexEnumParam>()->ValueString()))->GetAsFloat();
            auto y = cd->getData(std::string(this->_ySlot.Param<core::param::FlexEnumParam>()->ValueString()))->GetAsFloat();
            auto z = cd->getData(std::string(this->_zSlot.Param<core::param::FlexEnumParam>()->ValueString()))->GetAsFloat();
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
            _raw_positions = cd->getData(varname)->GetAsFloat();
            float xmin = std::numeric_limits<float>::max();
            float xmax = -std::numeric_limits<float>::max();
            float ymin = std::numeric_limits<float>::max();
            float ymax = -std::numeric_limits<float>::max();
            float zmin = std::numeric_limits<float>::max();
            float zmax = -std::numeric_limits<float>::max();
            for (int i = 0; i < _raw_positions.size()/3; ++i) {
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

    if (something_changed && !_raw_positions.empty()) {
        // find main axis
        std::array<float, 3> whd = {
            this->_bbox.BoundingBox().Width(), this->_bbox.BoundingBox().Height(), this->_bbox.BoundingBox().Depth()};
        _main_axis = std::distance(whd.begin(), std::max_element(whd.begin(), whd.end()));

        // get origin
        float center_z;
        if (std::copysign(1.0f, _bbox.BoundingBox().Front()) != std::copysign(1.0f, _bbox.BoundingBox().Back())) {
            center_z = (_bbox.BoundingBox().Front() + _bbox.BoundingBox().Back()) / 2;
        } else {
            center_z = _bbox.BoundingBox().Front() +
                       std::copysign(1.0f, _bbox.BoundingBox().Front()) * _bbox.BoundingBox().Depth() / 2;
        }
        _data_origin = {_bbox.BoundingBox().CalcCenter().GetX(), _bbox.BoundingBox().CalcCenter().GetY(), center_z};

        this->generateEllipsoid_2();
        this->sliceData();
        this->tighten();
        this->generateNormals();
        this->onionize();
        if (_showShellSlot.Param<core::param::IntParam>()->Value() > 0) {
            
        }

        _mesh_attribs.resize(2);
        _mesh_attribs[0].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
        _mesh_attribs[0].byte_size = _vertices.size() * sizeof(std::array<float, 4>);
        _mesh_attribs[0].component_cnt = 4;
        _mesh_attribs[0].stride = sizeof(std::array<float, 4>);
        _mesh_attribs[0].offset = 0;
        _mesh_attribs[0].data = reinterpret_cast<uint8_t*>(_vertices.data());
        _mesh_attribs[0].semantic = mesh::MeshDataAccessCollection::POSITION;

        _mesh_attribs[1].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
        _mesh_attribs[1].byte_size = _normals.size() * sizeof(std::array<float, 3>);
        _mesh_attribs[1].component_cnt = 3;
        _mesh_attribs[1].stride = sizeof(std::array<float, 3>);
        _mesh_attribs[1].offset = 0;
        _mesh_attribs[1].data = reinterpret_cast<uint8_t*>(_normals.data());
        _mesh_attribs[1].semantic = mesh::MeshDataAccessCollection::NORMAL;

        if (this->_faceTypeSlot.Param<core::param::EnumParam>()->Value() == 0) {
            _mesh_indices.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
            _mesh_indices.byte_size = _triangles.size() * sizeof(std::array<uint32_t, 3>);
            _mesh_indices.data = reinterpret_cast<uint8_t*>(_triangles.data());
            _mesh_type = mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES;
        } else {
            _mesh_indices.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
            _mesh_indices.byte_size = _faces.size() * sizeof(std::array<uint32_t, 4>);
            _mesh_indices.data = reinterpret_cast<uint8_t*>(_faces.data());
            _mesh_type = mesh::MeshDataAccessCollection::PrimitiveType::QUADS;
        }
        ++_version;
        }

        // put data in mesh
        mesh::MeshDataAccessCollection mesh;

        std::string identifier = std::string(FullName()) + "_mesh";
        mesh.addMesh(identifier, _mesh_attribs, _mesh_indices, _mesh_type);
        cm->setData(std::make_shared<mesh::MeshDataAccessCollection>(std::move(mesh)), _version);
        _old_datahash = cd->getDataHash();
        _recalc = false;

        auto meta_data = cm->getMetaData();
        meta_data.m_bboxs = _bbox;
        cm->setMetaData(meta_data);

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
        _vertices.clear();
        _normals.clear();
        _faces.clear();
        _triangles.clear();

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
            if (!cd->inquire(var)) {
                return false;
            }
        }

        if (!(*cd)(0))
            return false;

        // get data from volumetric call
        if (cd->getDataHash() != _old_datahash) {
            something_changed = true;

            if (this->_formatSlot.Param<core::param::EnumParam>()->Value() == 0) {
                auto x = cd->getData(std::string(this->_xSlot.Param<core::param::FlexEnumParam>()->ValueString()))
                             ->GetAsFloat();
                auto y = cd->getData(std::string(this->_ySlot.Param<core::param::FlexEnumParam>()->ValueString()))
                             ->GetAsFloat();
                auto z = cd->getData(std::string(this->_zSlot.Param<core::param::FlexEnumParam>()->ValueString()))
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
                _raw_positions = cd->getData(varname)->GetAsFloat();
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

        if (something_changed && !_raw_positions.empty()) {
            this->generateEllipsoid();
            this->sliceData();
            this->tighten();
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

    void ReconstructSurface::displayShell(int shell_number) {

        _vertices.clear();
        _vertices.resize(_shells[shell_number].num_vertices());
        for (int i = 0; i < _shells[shell_number].num_vertices(); ++i) {
            auto it = std::next(_shells[shell_number].points().begin(), i);

            glm::vec3 p = {it->x(), it->y(), it->z()};
            p.x += _data_origin[0];
            p.y += _data_origin[1];
            p.z += _data_origin[2];

            const Point point(p.x, p.y, p.z);
            *it = point;

            _vertices[i][0] = it->x();
            _vertices[i][1] = it->y();
            _vertices[i][2] = it->z();
        }

    }


} // namespace probe
} // namespace megamol
