/*
 * MeshUtilities.cpp
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once
#include "glm/glm.hpp"
#include "igl/boundary_loop.h"
#include "igl/decimate.h"
#include "igl/lscm.h"
#include "mesh/MeshCalls.h"
#include "nanoflann.hpp"
#include <random>

namespace megamol {
namespace probe {

inline float coulomb_force(float r) {
    return (1.0f / pow(r, 2));
}

// stackoverflow.com/questions/2550229/how-to-keep-only-duplicates-efficiently
template<class I, class P>
I remove_unique(I first, I last, P pred = P()) {
    I dest = first;
    while ((first = std::adjacent_find(first, last, pred)) != last) {
        *dest = *first;
        ++first;
        ++dest;
        if ((first = std::adjacent_find(first, last, std::not2(pred))) == last)
            break;
        ++first;
    }
    return dest;
}

template<class I>
I remove_unique(I first, I last) {
    return remove_unique(first, last, std::equal_to<typename std::iterator_traits<I>::value_type>());
}

template<typename Derived>
struct MeshAdaptor {
    typedef float coord_t;

    const Derived obj; //!< A const ref to the data set origin
    size_t point_count = 0;

    /// The constructor that sets the data set source
    MeshAdaptor(const Derived& obj_) : obj(obj_) {
        auto bs = derived()->byte_size;
        auto ts = mesh::MeshDataAccessCollection::getByteSize(derived()->component_type);
        auto tc = derived()->component_cnt;
        point_count = static_cast<size_t>(bs / (ts * tc));
    }

    /// CRTP helper method
    inline const Derived& derived() const {
        return obj;
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return point_count;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline coord_t kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0)
            return derived()->data[mesh::MeshDataAccessCollection::getByteSize(derived()->component_type) * (idx + 0)];
        else if (dim == 1)
            return derived()->data[mesh::MeshDataAccessCollection::getByteSize(derived()->component_type) * (idx + 1)];
        else
            return derived()->data[mesh::MeshDataAccessCollection::getByteSize(derived()->component_type) * (idx + 2)];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it
    //   again. Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template<class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const {
        return false;
    }
};


class MeshUtility {
public:
    void inputData(const mesh::MeshDataAccessCollection::Mesh& mesh_ptr) {

        this->_mesh = mesh_ptr;
        for (int i = 0; i < this->_mesh.attributes.size(); ++i) {
            if (this->_mesh.attributes[i].semantic == mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION) {
                this->_pos_attribute_idx = i;
                _va_ptr = &this->_mesh.attributes[i];
            } else if (this->_mesh.attributes[i].semantic ==
                       mesh::MeshDataAccessCollection::AttributeSemanticType::NORMAL) {
                this->_normal_attribute_idx = i;
            }
        }

        _mesh_indices = this->_mesh.indices;

        this->convertToEigenMatrices();
        this->createOrthonormalBasis();
        // this->buildKDTree(_va_ptr);
    }

    bool convertToMesh() {

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

    uint32_t getNumTotalFaces() {
        return _faces.rows();
    }

    float calcTriangleArea(uint32_t idx) {

        if (idx >= this->_faces.rows()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("[MeshUtility] Id is out of range");
            return -1;
        }

        Eigen::Matrix<int, 3, 2> other_indices;
        //other_indices << 1, 0, 0, 2, 2, 1;
        other_indices << 1, 2, 0, 2, 0, 1;

        auto vertices = this->getTriangleVertices(idx);

        std::array<glm::vec3, 3> edges = {
            vertices[1] - vertices[0], vertices[2] - vertices[1], vertices[0] - vertices[2]};
        std::array<float, 3> edge_lengths = {glm::length(edges[0]), glm::length(edges[1]), glm::length(edges[2])};
        int longest_edge =
            std::distance(edge_lengths.begin(), std::max_element(edge_lengths.begin(), edge_lengths.end()));


        float proj_on_longest_edge =
            glm::dot(glm::normalize(edges[longest_edge]), glm::normalize(edges[other_indices(longest_edge, 0)]));
        auto to_cut = edges[longest_edge] * std::abs(proj_on_longest_edge);
        auto e = to_cut + vertices[longest_edge];
        auto e_o1 = e - vertices[other_indices(longest_edge, 0)];
        auto e_o2 = e - vertices[other_indices(longest_edge, 1)];

        auto area = (glm::length(to_cut) * glm::length(e_o1)) / 2.0f + (glm::length(e_o1) * glm::length(e_o2)) / 2.0f;

        return area;
    }

    std::vector<uint32_t> getNeighboringTriangles(uint32_t idx, int min_common_points = 2) {

        // std::vector<uint32_t> result;
        // for (uint32_t i = 0; i < this->_faces.rows(); ++i) {
        //    int common_points = 0;
        //    for (int j = 0; j < this->_faces.cols(); ++j) {
        //        for (int k = 0; k < this->_faces.cols(); ++k) {
        //            if (this->_faces(idx, k) == this->_faces(i, j)) common_points++;
        //        }
        //    }
        //    if (common_points >= min_common_points) {
        //        result.emplace_back(i);
        //    }
        //}
        // return result;

        std::vector<uint32_t> neighboring_triangles;

        std::array<uint32_t, 3> indices;
        indices[0] = this->_faces(idx, 0);
        indices[1] = this->_faces(idx, 1);
        indices[2] = this->_faces(idx, 2);
        for (auto index : indices) {

            auto it = _std_faces.begin();

            while (it != _std_faces.end()) {
                it = std::find(it, _std_faces.end(), index);
                if (it != _std_faces.end()) {
                    neighboring_triangles.emplace_back(std::distance(_std_faces.begin(), it) / 3);
                    ++it;
                }
            }
        }
        //std::sort(neighboring_triangles.begin(), neighboring_triangles.end());
        //auto last = std::unique(neighboring_triangles.begin(), neighboring_triangles.end());
        //neighboring_triangles.erase(last, neighboring_triangles.end());
        std::sort(neighboring_triangles.begin(), neighboring_triangles.end());
        auto rmv = remove_unique(neighboring_triangles.begin(), neighboring_triangles.end());
        neighboring_triangles.erase(rmv, neighboring_triangles.end());
        return neighboring_triangles;
    }

    float getLongestEdgeLength(uint32_t idx) {
        glm::vec3 v0 = {_vertices(_faces(idx, 0), 0), _vertices(_faces(idx, 0), 1), _vertices(_faces(idx, 0), 2)};
        glm::vec3 v1 = {_vertices(_faces(idx, 1), 0), _vertices(_faces(idx, 1), 1), _vertices(_faces(idx, 1), 2)};
        glm::vec3 v2 = {_vertices(_faces(idx, 2), 0), _vertices(_faces(idx, 2), 1), _vertices(_faces(idx, 2), 2)};

        auto e0 = v0 - v1;
        auto e1 = v0 - v2;
        auto e2 = v1 - v2;

        return std::max(std::max(glm::length(e0), glm::length(e1)), glm::length(e2));
    }

    void UVMapping(const Eigen::MatrixXi& faces, const Eigen::MatrixXd& vertices, Eigen::MatrixXd& vertices_uv) {

        // Fix two points on the boundary
        Eigen::VectorXi bnd, b(2, 1);
        igl::boundary_loop(faces, bnd);
        b(0) = bnd(0);
        b(1) = bnd(round(bnd.size() / 2));
        Eigen::MatrixXd bc(2, 2);
        bc << 0, 0, 1, 0;
        igl::lscm(vertices, faces, b, bc, vertices_uv);
    }

    bool orthogonalProjection(uint32_t idx, const Eigen::MatrixXd& points, Eigen::MatrixXd& projected_points) {

        glm::vec3 u;
        glm::vec3 v;

        for (int i = 0; i < 3; ++i) {
            u[i] = _vertices(_faces(idx, 1), i) - _vertices(_faces(idx, 0), i);
            v[i] = _vertices(_faces(idx, 2), i) - _vertices(_faces(idx, 0), i);
        }

        projected_points.resizeLike(points);
        for (int i = 0; i < points.rows(); ++i) {
            glm::vec3 point;
            for (int j = 0; j < 3; ++j) {
                point[j] = points(i, j);
            }
            auto projection = (glm::dot(point, u) / glm::dot(u, u)) * u + (glm::dot(point, v) / glm::dot(v, v)) * v;
            for (int j = 0; j < 3; ++j) {
                projected_points(i, j) = projection[j];
            }
        }
        return true;
    }

    // Performs an orthonormal transformation into the basis of triangle idx
    // returns a 3D matrix where the n-dimension (3rd entry) can be omitted to get the 2D representation
    bool perform2Dprojection(uint32_t idx, const Eigen::MatrixXd& points, Eigen::MatrixXd& projected_points) {

        projected_points.resize(points.rows(), 3);

        // now solve the system to get the points in the new basis
        Eigen::Matrix3d A;
        A << _orthonormalBasis[idx][0].x, _orthonormalBasis[idx][1].x, _orthonormalBasis[idx][2].x,
            _orthonormalBasis[idx][0].y, _orthonormalBasis[idx][1].y, _orthonormalBasis[idx][2].y,
            _orthonormalBasis[idx][0].z, _orthonormalBasis[idx][1].z, _orthonormalBasis[idx][2].z;
        // A << u.x, u.y, u.z, v.x, v.y, v.z, n.x, n.y, n.z;
        for (int i = 0; i < points.rows(); ++i) {
            Eigen::Vector3d b;
            b << points(i, 0), points(i, 1), points(i, 2);
            auto proj_point = A.colPivHouseholderQr().solve(b);
            for (int j = 0; j < projected_points.cols(); ++j) {
                projected_points(i, j) = proj_point[j];
            }
        }

        return true;
    }

    bool performInverse2Dprojection(uint32_t idx, const Eigen::MatrixXd& points, Eigen::MatrixXd& projected_points) {

        projected_points.resize(points.rows(), 3);
#pragma omp parallel for
        for (int i = 0; i < points.rows(); ++i) {
            for (int j = 0; j < projected_points.cols(); ++j) {
                projected_points(i, j) = points(i, 0) * _orthonormalBasis[idx][0][j] +
                                         points(i, 1) * _orthonormalBasis[idx][1][j] +
                                         points(i, 2) * _orthonormalBasis[idx][2][j];
            }
        }

        return true;
    }

    std::vector<glm::vec3> seedPoints(const uint32_t idx, const int num_pts) {
        std::mt19937 rnd;
        rnd.seed(std::random_device()());
        // rnd.seed(666);
        std::uniform_real_distribution<float> fltdist(0, 1);

        // get triangle vertices
        auto v0_idx = this->_faces(idx, 0);
        auto v1_idx = this->_faces(idx, 1);
        auto v2_idx = this->_faces(idx, 2);

        std::vector<glm::vec3> seededPoints(num_pts);

        for (int i = 0; i < num_pts; ++i) {
            // $P = (1 - \sqrt{r_1}) A + (\sqrt{r_1}(1 - r_2)) B + (r_2 \sqrt{r_1})C$
            auto rnd_u = fltdist(rnd);
            auto rnd_v = fltdist(rnd);

            glm::vec3 point;

            point[0] = (1 - std::sqrt(rnd_u)) * this->_vertices(v0_idx, 0) +
                       (std::sqrt(rnd_u) * (1 - rnd_v)) * this->_vertices(v1_idx, 0) +
                       (rnd_v * std::sqrt(rnd_u)) * this->_vertices(v2_idx, 0);

            point[1] = (1 - std::sqrt(rnd_u)) * this->_vertices(v0_idx, 1) +
                       (std::sqrt(rnd_u) * (1 - rnd_v)) * this->_vertices(v1_idx, 1) +
                       (rnd_v * std::sqrt(rnd_u)) * this->_vertices(v2_idx, 1);

            point[2] = (1 - std::sqrt(rnd_u)) * this->_vertices(v0_idx, 2) +
                       (std::sqrt(rnd_u) * (1 - rnd_v)) * this->_vertices(v1_idx, 2) +
                       (rnd_v * std::sqrt(rnd_u)) * this->_vertices(v2_idx, 2);

            seededPoints[i] = point;
        }
        return seededPoints;
    }

    bool seedPoints(const uint32_t idx, const int num_pts, Eigen::MatrixXd& result) {
        std::mt19937 rnd;
        rnd.seed(std::random_device()());
        // rnd.seed(666);
        std::uniform_real_distribution<float> fltdist(0, 1);

        // get triangle vertices
        auto v0_idx = this->_faces(idx, 0);
        auto v1_idx = this->_faces(idx, 1);
        auto v2_idx = this->_faces(idx, 2);

        auto v0_0 = this->_vertices(v0_idx, 0);
        auto v0_1 = this->_vertices(v0_idx, 1);
        auto v0_2 = this->_vertices(v0_idx, 2);

        auto v1_0 = this->_vertices(v1_idx, 0);
        auto v1_1 = this->_vertices(v1_idx, 1);
        auto v1_2 = this->_vertices(v1_idx, 2);

        auto v2_0 = this->_vertices(v2_idx, 0);
        auto v2_1 = this->_vertices(v2_idx, 1);
        auto v2_2 = this->_vertices(v2_idx, 2);

        result.resize(num_pts, 3);

        for (int i = 0; i < num_pts; ++i) {
            // $P = (1 - \sqrt{r_1}) A + (\sqrt{r_1}(1 - r_2)) B + (r_2 \sqrt{r_1})C$
            auto rnd_u = fltdist(rnd);
            auto rnd_v = fltdist(rnd);

            glm::vec3 result_vec;
            for (int j = 0; j < result.cols(); ++j) {
                //result(i, j) = this->_vertices(v0_idx, j) +
                //               rnd_u * (this->_vertices(v1_idx, j) - this->_vertices(v0_idx, j)) + rnd_v *
                //               (this->_vertices(v2_idx, j) - this->_vertices(v0_idx, j));
                result(i, j) = (1 - std::sqrt(rnd_u)) * this->_vertices(v0_idx, j) +
                               (std::sqrt(rnd_u) * (1 - rnd_v)) * this->_vertices(v1_idx, j) +
                               (rnd_v * std::sqrt(rnd_u)) * this->_vertices(v2_idx, j);
                result_vec[j] = result(i, j);
            }
        }
        return true;
    }

    std::array<glm::vec3, 3> getTriangleVertices(const uint32_t idx) {

        std::array<glm::vec3, 3> verts;
        for (int i = 0; i < 3; ++i) {
            glm::vec3 vert;
            for (int j = 0; j < 3; ++j) {
                vert[j] = this->_vertices(this->_faces(idx, i), j);
            }
            verts[i] = vert;
        }
        return verts;
    }

    bool getTriangleVertices(const uint32_t idx, Eigen::Matrix3d& verts) {

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                verts(i, j) = this->_vertices(this->_faces(idx, i), j);
            }
        }
        return true;
    }

    bool pointInTriangle(const uint32_t idx, glm::vec3 p) {

        auto vertices = this->getTriangleVertices(idx);

        // Compute vectors
        auto v0 = vertices[2] - vertices[0];
        auto v1 = vertices[1] - vertices[0];
        auto v2 = p - vertices[0];

        // Compute dot products
        auto dot00 = glm::dot(v0, v0);
        auto dot01 = glm::dot(v0, v1);
        auto dot02 = glm::dot(v0, v2);
        auto dot11 = glm::dot(v1, v1);
        auto dot12 = glm::dot(v1, v2);

        // Compute barycentric coordinates
        auto invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
        auto u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        auto v = (dot00 * dot12 - dot01 * dot02) * invDenom;

        // Check if point is in triangle
        bool res = (u > 0) && (v > 0) && (u + v < 1);
        if (res)
            return true;
        return false;
    }

    bool pointInTriangle(const std::array<glm::vec3, 3>& vertices, glm::vec3 p) {

        // Compute vectors
        auto v0 = vertices[2] - vertices[0];
        auto v1 = vertices[1] - vertices[0];
        auto v2 = p - vertices[0];

        // Compute dot products
        auto dot00 = glm::dot(v0, v0);
        auto dot01 = glm::dot(v0, v1);
        auto dot02 = glm::dot(v0, v2);
        auto dot11 = glm::dot(v1, v1);
        auto dot12 = glm::dot(v1, v2);

        // Compute barycentric coordinates
        auto invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
        auto u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        auto v = (dot00 * dot12 - dot01 * dot02) * invDenom;

        // Check if point is in triangle
        bool res = (u > 0) && (v > 0) && (u + v < 1);
        if (res)
            return true;
        return false;
    }

    bool pointInTriangle(const std::array<glm::vec2, 3>& vertices, glm::vec2 p) {

        const auto signed_area =
            0.5 * (-vertices[1].y * vertices[2].x + vertices[0].y * (-vertices[1].x + vertices[2].x) +
                      vertices[0].x * (vertices[1].y - vertices[2].y) + vertices[1].x * vertices[2].y);

        const auto s = 1 / (2 * signed_area) *
                       (vertices[0].y * vertices[2].x - vertices[0].x * vertices[2].y +
                           (vertices[2].y - vertices[0].y) * p.x + (vertices[0].x - vertices[2].x) * p.y);
        const auto t = 1 / (2 * signed_area) *
                       (vertices[0].x * vertices[1].y - vertices[0].y * vertices[1].x +
                           (vertices[0].y - vertices[1].y) * p.x + (vertices[1].x - vertices[0].x) * p.y);

        return (s > 0 && t > 0 && 1 - s - t > 0);

        //const auto d1 = this->sign(p, vertices[0], vertices[1]);
        //const auto d2 = this->sign(p, vertices[1], vertices[2]);
        //const auto d3 = this->sign(p, vertices[2], vertices[0]);

        //const auto has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        //const auto has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

        //return !(has_neg && has_pos);
    }

    bool getPatch(uint32_t idx, Eigen::MatrixXd& out_verts, Eigen::MatrixXi& out_indices,
        const std::vector<uint32_t>& neighbors_ = std::vector<uint32_t>()) {

        std::vector<uint32_t> neighbors;
        // check for triangles with two common vertices
        if (neighbors_.empty()) {
            neighbors = this->getNeighboringTriangles(idx);
        } else {
            neighbors = neighbors_;
        }
        assert(neighbors.size() > 3);

        // auto idx_it = std::find(neighbors.begin(), neighbors.end(), idx);
        // neighbors.erase(idx_it);

        out_indices.resize(neighbors.size(), 3);

        std::vector<std::pair<uint32_t, uint32_t>> index_mapper(3);
        // put the main triangle in the index mapper
        index_mapper[0] = std::make_pair(0, this->_faces(idx, 0));
        index_mapper[1] = std::make_pair(1, this->_faces(idx, 1));
        index_mapper[2] = std::make_pair(2, this->_faces(idx, 2));
        // find the other indices
        for (int i = 1; i < neighbors.size(); ++i) {
            for (int j = 0; j < this->_faces.cols(); ++j) {
                std::vector<int> check_index;
                for (int k = 0; k < index_mapper.size(); ++k) {
                    if (this->_faces(neighbors[i], j) == index_mapper[k].second) {
                        check_index.emplace_back(k);
                    }
                }
                if (check_index.empty()) {
                    // make sure that there are no double indices
                    bool fill = true;
                    for (int n = 0; n < index_mapper.size(); ++n) {
                        if (this->_faces(neighbors[i], j) == index_mapper[n].second) {
                            fill = false;
                        }
                    }
                    if (fill) {
                        uint32_t id = index_mapper.size();
                        index_mapper.emplace_back(id, this->_faces(neighbors[i], j));
                    }
                }
            }
        }

        out_verts.resize(index_mapper.size(), 3);
        // fill the out vertices
        for (int i = 0; i < index_mapper.size(); ++i) {
            for (int j = 0; j < 3; ++j) {
                out_verts(index_mapper[i].first, j) = this->_vertices(index_mapper[i].second, j);
            }
        }

        // fill the out indices
        for (int i = 0; i < neighbors.size(); ++i) {
            for (int j = 0; j < this->_faces.cols(); ++j) {
                for (int k = 0; k < index_mapper.size(); ++k) {
                    if (this->_faces(neighbors[i], j) == index_mapper[k].second) {
                        out_indices(i, j) = static_cast<int>(index_mapper[k].first);
                    }
                }
            }
        }
        return true;
    }

    void fillMeshVertices(const Eigen::MatrixXd& in_vertices, std::vector<float>& out_vertices) {
        out_vertices.resize(in_vertices.rows() * 3);
#pragma omp parallel for
        for (int i = 0; i < in_vertices.rows(); ++i) {
            for (int j = 0; j < in_vertices.cols(); ++j) {
                out_vertices[3 * i + j] = static_cast<float>(in_vertices(i, j));
            }
            if (_vertices.cols() < 3) {
                out_vertices[3 * i + 2] = 0;
            }
        }
    }

    void fillMeshFaces(const Eigen::MatrixXi& in_faces, std::vector<uint32_t>& out_faces) {
        out_faces.resize(in_faces.rows() * in_faces.cols());
#pragma omp parallel for
        for (int i = 0; i < in_faces.rows(); ++i) {
            for (int j = 0; j < in_faces.cols(); ++j) {
                out_faces[in_faces.cols() * i + j] = static_cast<uint32_t>(in_faces(i, j));
            }
        }
    }

private:
    int nearestKSearch(const std::array<float, 3>& point, int k, std::vector<uint32_t>& k_indices,
        std::vector<float>& k_distances) const {

        k_indices.resize(k);
        k_distances.resize(k);

        std::vector<float> query(3);

        nanoflann::KNNResultSet<float, uint32_t, int> resultSet(k);
        resultSet.init(k_indices.data(), k_distances.data());
        this->_kd_tree->findNeighbors(resultSet, point.data(), nanoflann::SearchParams(10));

        return (k);
    }

    void buildKDTree(const mesh::MeshDataAccessCollection::VertexAttribute* pos_attr) {
        if (this->_mesh.attributes[_pos_attribute_idx].data == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[MeshUtility] Cannot construct KD Tree. No mesh set.");
            return;
        }

        this->_kd_tree = std::make_shared<NanoFlannIndex>(3, pos_attr, ::nanoflann::KDTreeSingleIndexAdaptorParams(15));

        this->_kd_tree->buildIndex();
    }

    template<typename T>
    void fillVertexMatrix(const T vert_data) {
#pragma omp parallel for
        for (int i = 0; i < _vertices.rows(); ++i) {
            _vertices(i, 0) =
                static_cast<double>(vert_data[this->_mesh.attributes[_pos_attribute_idx].component_cnt * i + 0]);
            _vertices(i, 1) =
                static_cast<double>(vert_data[this->_mesh.attributes[_pos_attribute_idx].component_cnt * i + 1]);
            _vertices(i, 2) =
                static_cast<double>(vert_data[this->_mesh.attributes[_pos_attribute_idx].component_cnt * i + 2]);
        }
    }

    template<typename T>
    void fillNormalMatrix(const T vert_data) {
#pragma omp parallel for
        for (int i = 0; i < _normals.rows(); ++i) {
            _normals(i, 0) =
                static_cast<double>(vert_data[this->_mesh.attributes[_normal_attribute_idx].component_cnt * i + 0]);
            _normals(i, 1) =
                static_cast<double>(vert_data[this->_mesh.attributes[_normal_attribute_idx].component_cnt * i + 1]);
            _normals(i, 2) =
                static_cast<double>(vert_data[this->_mesh.attributes[_normal_attribute_idx].component_cnt * i + 2]);
        }
    }

    template<typename T>
    void fillFaceMatrix(const T faces) {
#pragma omp parallel for
        for (int j = 0; j < _faces.rows(); ++j) {
            _faces(j, 0) = static_cast<int>(faces[3 * j + 0]);
            _faces(j, 1) = static_cast<int>(faces[3 * j + 1]);
            _faces(j, 2) = static_cast<int>(faces[3 * j + 2]);
        }
    }

    bool convertToEigenMatrices() {
        _vertices.resize(this->_mesh.attributes[_pos_attribute_idx].byte_size /
                             (this->_mesh.attributes[_pos_attribute_idx].component_cnt *
                                 mesh::MeshDataAccessCollection::getByteSize(
                                     this->_mesh.attributes[_pos_attribute_idx].component_type)),
            this->_mesh.attributes[_pos_attribute_idx].component_cnt);

        const auto indices = this->_mesh.indices;
        _faces.resize(indices.byte_size / (3 * mesh::MeshDataAccessCollection::getByteSize(indices.type)), 3);

        switch (indices.type) {
        case mesh::MeshDataAccessCollection::UNSIGNED_SHORT: {
            auto face_data = reinterpret_cast<unsigned short*>(indices.data);
            this->fillFaceMatrix(face_data);
        } break;
        case mesh::MeshDataAccessCollection::UNSIGNED_INT: {
            auto face_data = reinterpret_cast<uint32_t*>(indices.data);
            this->fillFaceMatrix(face_data);
        } break;
        case mesh::MeshDataAccessCollection::INT: {
            auto face_data = reinterpret_cast<int*>(indices.data);
            this->fillFaceMatrix(face_data);
        } break;
        }

        switch (this->_mesh.attributes[_pos_attribute_idx].component_type) {
        case mesh::MeshDataAccessCollection::FLOAT: {
            auto vert_data = reinterpret_cast<float*>(this->_mesh.attributes[_pos_attribute_idx].data);
            this->fillVertexMatrix(vert_data);
        } break;
        case mesh::MeshDataAccessCollection::DOUBLE: {
            auto vert_data = reinterpret_cast<double*>(this->_mesh.attributes[_pos_attribute_idx].data);
            this->fillVertexMatrix(vert_data);
        } break;
        }

        // faces also as stl vector for quick find
        // std::vector<int> tmp_faces(_faces.data(), _faces.data() + _faces.rows() * _faces.cols());
        _std_faces.resize(_faces.rows() * _faces.cols());
#pragma omp parallel for
        for (int i = 0; i < _faces.rows(); ++i) {
            for (int j = 0; j < _faces.cols(); ++j) {
                _std_faces[_faces.cols() * i + j] = _faces(i, j);
            }
        }

        // get normals if there are some
        if (this->_normal_attribute_idx != -1) {
            _normals.resize(this->_mesh.attributes[_normal_attribute_idx].byte_size /
                                (this->_mesh.attributes[_normal_attribute_idx].component_cnt *
                                    mesh::MeshDataAccessCollection::getByteSize(
                                        this->_mesh.attributes[_normal_attribute_idx].component_type)),
                this->_mesh.attributes[_normal_attribute_idx].component_cnt);

            switch (this->_mesh.attributes[_normal_attribute_idx].component_type) {
            case mesh::MeshDataAccessCollection::FLOAT: {
                auto normal_data = reinterpret_cast<float*>(this->_mesh.attributes[_pos_attribute_idx].data);
                this->fillNormalMatrix(normal_data);
            } break;
            case mesh::MeshDataAccessCollection::DOUBLE: {
                auto normal_data = reinterpret_cast<double*>(this->_mesh.attributes[_pos_attribute_idx].data);
                this->fillNormalMatrix(normal_data);
            } break;
            }
        }

        return true;
    }

    float sign(const glm::vec2 p1, const glm::vec2 p2, const glm::vec2 p3) {
        return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
    }

    bool createOrthonormalBasis() {

        _orthonormalBasis.resize(_faces.rows());

#pragma omp parallel for
        for (int idx = 0; idx < _faces.rows(); ++idx) {
            // create new orthonormal basis
            glm::vec3 u;
            glm::vec3 tmp_v;

            for (int i = 0; i < 3; ++i) {
                u[i] = _vertices(_faces(idx, 1), i) - _vertices(_faces(idx, 0), i);
                tmp_v[i] = _vertices(_faces(idx, 2), i) - _vertices(_faces(idx, 0), i);
            }

            auto n = glm::cross(u, tmp_v);
            auto v = glm::cross(u, n);

            _orthonormalBasis[idx][0] = glm::normalize(u);
            _orthonormalBasis[idx][1] = glm::normalize(v);
            _orthonormalBasis[idx][2] = glm::normalize(n);
        }
    }


    typedef MeshAdaptor<const mesh::MeshDataAccessCollection::VertexAttribute*> mesh_adaptor;
    typedef ::nanoflann::KDTreeSingleIndexAdaptor<::nanoflann::L2_Simple_Adaptor<float, mesh_adaptor, float>,
        mesh_adaptor, 3>
        NanoFlannIndex;

    std::vector<float> _mesh_vertices;
    std::vector<uint32_t> _mesh_faces;
    std::vector<mesh::MeshDataAccessCollection::VertexAttribute> _mesh_attribs;
    mesh::MeshDataAccessCollection::IndexData _mesh_indices;
    Eigen::MatrixXd _vertices;
    Eigen::MatrixXd _normals;
    Eigen::MatrixXi _faces;
    uint32_t _pos_attribute_idx;
    int _normal_attribute_idx = -1;
    std::vector<std::array<glm::vec3, 3>> _orthonormalBasis;

    std::vector<int> _std_faces;

    mesh::MeshDataAccessCollection::Mesh _mesh;
    std::shared_ptr<NanoFlannIndex> _kd_tree;
    const mesh::MeshDataAccessCollection::VertexAttribute* _va_ptr;
};


} // namespace probe
} // namespace megamol
