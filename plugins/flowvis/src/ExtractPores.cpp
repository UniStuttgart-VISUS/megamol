#include "stdafx.h"
#include "ExtractPores.h"

#include "mesh/MeshDataCall.h"
#include "mesh/TriangleMeshCall.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/utility/DataHash.h"

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_3.h>
#include <CGAL/Segment_3.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Surface_mesh.h>

#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup_extension.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

typedef K::Point_3 Point;
typedef K::Segment_3 Segment;

typedef CGAL::Surface_mesh<Point> Mesh;
typedef CGAL::Delaunay_triangulation_3<K> Delaunay;

typedef CGAL::AABB_face_graph_triangle_primitive<Mesh, CGAL::Default, CGAL::Tag_false> Primitive;
typedef CGAL::AABB_traits<K, Primitive> Traits;
typedef CGAL::AABB_tree<Traits> Tree;

typedef std::vector<std::size_t> CGAL_Polygon;

namespace PMP = CGAL::Polygon_mesh_processing;

megamol::flowvis::ExtractPores::ExtractPores()
        : mesh_lhs_slot("mesh_lhs_slot", "Separated surface meshes representing pores and throats.")
        , mesh_data_lhs_slot("mesh_data_lhs_slot", "Data associated with the meshes.")
        , mesh_rhs_slot("mesh_rhs_slot", "Input surface mesh of the fluid phase.")
        , pore_criterion("pore_criterion", "Criterion for classifying a tetrahedron as belonging to a pore.")
        , input_hash(ExtractPores::GUID()) {

    // Connect input slot
    this->mesh_rhs_slot.SetCompatibleCall<mesh::TriangleMeshCall::triangle_mesh_description>();
    this->MakeSlotAvailable(&this->mesh_rhs_slot);

    // Connect output slots
    this->mesh_lhs_slot.SetCallback(
        mesh::TriangleMeshCall::ClassName(), "get_data", &ExtractPores::getMeshDataCallback);
    this->mesh_lhs_slot.SetCallback(
        mesh::TriangleMeshCall::ClassName(), "get_extent", &ExtractPores::getMeshMetaDataCallback);
    this->MakeSlotAvailable(&this->mesh_lhs_slot);

    this->mesh_data_lhs_slot.SetCallback(
        mesh::MeshDataCall::ClassName(), "get_data", &ExtractPores::getMeshDataDataCallback);
    this->mesh_data_lhs_slot.SetCallback(
        mesh::MeshDataCall::ClassName(), "get_extent", &ExtractPores::getMeshDataMetaDataCallback);
    this->MakeSlotAvailable(&this->mesh_data_lhs_slot);

    // Create parameters
    this->pore_criterion << new core::param::EnumParam(0);
    this->pore_criterion.Param<core::param::EnumParam>()->SetTypePair(0, "< 1 shared segments");
    this->pore_criterion.Param<core::param::EnumParam>()->SetTypePair(1, "< 2 shared segments");
    this->MakeSlotAvailable(&this->pore_criterion);
}

megamol::flowvis::ExtractPores::~ExtractPores() {
    this->Release();
}

bool megamol::flowvis::ExtractPores::create() {
    return true;
}

void megamol::flowvis::ExtractPores::release() {}

bool megamol::flowvis::ExtractPores::getMeshDataCallback(core::Call& _call) {
    assert(dynamic_cast<mesh::TriangleMeshCall*>(&_call) != nullptr);

    auto& call = static_cast<mesh::TriangleMeshCall&>(_call);

    if (!compute()) {
        return false;
    }

    call.set_vertices(this->output.vertices);
    call.set_normals(this->output.normals);
    call.set_indices(this->output.indices);

    call.SetDataHash(core::utility::DataHash(
        ExtractPores::GUID(), this->input_hash, this->pore_criterion.Param<core::param::EnumParam>()->Value()));

    return true;
}

bool megamol::flowvis::ExtractPores::getMeshMetaDataCallback(core::Call& _call) {
    assert(dynamic_cast<mesh::TriangleMeshCall*>(&_call) != nullptr);

    auto& call = static_cast<mesh::TriangleMeshCall&>(_call);

    // Get input extent
    auto tmc_ptr = this->mesh_rhs_slot.CallAs<mesh::TriangleMeshCall>();

    if (tmc_ptr == nullptr || !(*tmc_ptr)(1)) {
        return false;
    }

    call.set_dimension(tmc_ptr->get_dimension());
    call.set_bounding_box(tmc_ptr->get_bounding_box());

    return true;
}

bool megamol::flowvis::ExtractPores::getMeshDataDataCallback(core::Call& _call) {
    assert(dynamic_cast<mesh::MeshDataCall*>(&_call) != nullptr);

    auto& call = static_cast<mesh::MeshDataCall&>(_call);

    if (!compute()) {
        return false;
    }

    call.set_data("id", this->output.datasets[0]);
    call.set_data("surface", this->output.datasets[1]);
    call.set_data("surface-to-volume ratio", this->output.datasets[2]);
    call.set_data("type", this->output.datasets[3]);
    call.set_data("volume", this->output.datasets[4]);

    call.SetDataHash(core::utility::DataHash(
        ExtractPores::GUID(), this->input_hash, this->pore_criterion.Param<core::param::EnumParam>()->Value()));

    return true;
}

bool megamol::flowvis::ExtractPores::getMeshDataMetaDataCallback(core::Call& _call) {
    assert(dynamic_cast<mesh::MeshDataCall*>(&_call) != nullptr);

    auto& call = static_cast<mesh::MeshDataCall&>(_call);

    // The following data is computed:
    // id                           - Unique ID of the pore or throat
    call.set_data("id");
    // surface                      - The surface of the pore or throat
    call.set_data("surface");
    // surface-to-volume ratio      - The surface-to-volume ratio of the pore or throat
    call.set_data("surface-to-volume ratio");
    // type                         - The type of the volume: pore (0), throat (1)
    call.set_data("type");
    // volume                       - The volume of the pore or throat
    call.set_data("volume");

    return true;
}

bool megamol::flowvis::ExtractPores::compute() {

    // Check input connection and get data
    auto tmc_ptr = this->mesh_rhs_slot.CallAs<mesh::TriangleMeshCall>();

    if (tmc_ptr == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Triangle mesh input is not connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);

        return false;
    }

    auto& tmc = *tmc_ptr;

    if (!tmc(0)) {
        if (tmc.DataHash() != this->input_hash) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Error getting triangle mesh. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);

            this->input_hash = tmc.DataHash();
        }

        return false;
    }

    bool input_changed = false;

    if (tmc.DataHash() != this->input_hash) {
        this->input.vertices = tmc.get_vertices();
        this->input.normals = tmc.get_normals();
        this->input.indices = tmc.get_indices();

        this->input_hash = tmc.DataHash();

        input_changed = true;
    }

    // Set empty output when encountering empty input
    if (this->input.vertices == nullptr) {
        this->output.vertices = nullptr;
        this->output.normals = nullptr;
        this->output.indices = nullptr;

        std::for_each(this->output.datasets.begin(), this->output.datasets.end(),
            [](std::shared_ptr<mesh::MeshDataCall::data_set>& dataset) { dataset = nullptr; });

        input_changed = false;

        return true;
    }

    // Perform computation
    if (input_changed || this->pore_criterion.IsDirty()) {
        this->output.vertices = std::make_shared<std::vector<float>>();
        this->output.normals = nullptr;
        this->output.indices = std::make_shared<std::vector<unsigned int>>();

        this->pore_criterion.ResetDirty();

        // Create surface mesh from input triangles and bounding box
        Mesh mesh;

        {
            std::vector<Point> points;
            std::vector<CGAL_Polygon> polygon_vec;

            // Add input mesh to polygon soup
            points.resize(this->input.vertices->size() / 3);
            polygon_vec.resize(this->input.indices->size() / 3);

            for (std::size_t i = 0; i < points.size(); ++i) {
                points[i] = Point((*this->input.vertices)[i * 3 + 0], (*this->input.vertices)[i * 3 + 1],
                    (*this->input.vertices)[i * 3 + 2]);
            }

            for (std::size_t i = 0; i < polygon_vec.size(); ++i) {
                polygon_vec[i] = {(*this->input.indices)[i * 3 + 0], (*this->input.indices)[i * 3 + 1],
                    (*this->input.indices)[i * 3 + 2]};
            }

            // Create CGAL surface mesh from polygon soup
            {
                PMP::remove_isolated_points_in_polygon_soup(points, polygon_vec);
                PMP::merge_duplicate_points_in_polygon_soup(points, polygon_vec);
                PMP::duplicate_non_manifold_edges_in_polygon_soup(points, polygon_vec);

                if (!PMP::orient_polygon_soup(points, polygon_vec)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Error orienting the triangles to form a surface mesh. [%s, %s, line %d]\n", __FILE__,
                        __FUNCTION__, __LINE__);

                    return false;
                }

                if (!PMP::is_polygon_soup_a_polygon_mesh(polygon_vec)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Cannot form a surface mesh from the input triangles. [%s, %s, line %d]\n", __FILE__,
                        __FUNCTION__, __LINE__);

                    return false;
                }

                PMP::polygon_soup_to_polygon_mesh(points, polygon_vec, mesh);
            }
        }

        // Tetrahedralize domain
        Delaunay dt;
        dt.insert(mesh.points().begin(), mesh.points().end());

        // Iterate over all cells and check which side its center is on
        std::size_t num_pores = 0;

        CGAL::Side_of_triangle_mesh<Mesh, K> orientation_test(mesh);

        Tree tree(CGAL::faces(mesh).first, CGAL::faces(mesh).second, mesh);
        tree.build();

        const auto err_threshold = 1e-6 * (static_cast<double>(tmc.get_bounding_box().Width()) +
                                              static_cast<double>(tmc.get_bounding_box().Height()) +
                                              static_cast<double>(tmc.get_bounding_box().Depth()));

        for (auto cell = dt.finite_cells_begin(); cell != dt.finite_cells_end(); ++cell) {
            const auto cell_center = CGAL::ORIGIN +
                (0.25 * ((cell->vertex(0)->point() - CGAL::ORIGIN) + (cell->vertex(1)->point() - CGAL::ORIGIN) +
                            (cell->vertex(2)->point() - CGAL::ORIGIN) + (cell->vertex(3)->point() - CGAL::ORIGIN)));

            if (orientation_test(cell_center) != CGAL::ON_BOUNDED_SIDE) {
                // Count number of edges shared with the surface mesh
                const auto& p1 = cell->vertex(0)->point();
                const auto& p2 = cell->vertex(1)->point();
                const auto& p3 = cell->vertex(2)->point();
                const auto& p4 = cell->vertex(3)->point();

                std::array<Segment, 6> edges{Segment(p1, p2), Segment(p1, p3), Segment(p1, p4), Segment(p2, p3),
                    Segment(p2, p4), Segment(p3, p4)};

                int num_shared_segments = 0;

                for (auto& edge : edges) {
                    auto edge_center = CGAL::midpoint(edge.start(), edge.target());

                    if (CGAL::to_double(tree.squared_distance(edge_center)) < err_threshold) {
                        ++num_shared_segments;
                    }
                }

                // If at most one edge is shared with the surface mesh, then this is a pore
                const auto max_num_shared_segments =
                    (this->pore_criterion.Param<core::param::EnumParam>()->Value() == 0) ? 0 : 1;

                if (num_shared_segments <= max_num_shared_segments) {
                    ++num_pores;

                    const auto index = this->output.vertices->size() / 3;

                    // Add tetrahedron vertices
                    this->output.vertices->push_back(static_cast<float>(CGAL::to_double(cell->vertex(0)->point().x())));
                    this->output.vertices->push_back(static_cast<float>(CGAL::to_double(cell->vertex(0)->point().y())));
                    this->output.vertices->push_back(static_cast<float>(CGAL::to_double(cell->vertex(0)->point().z())));

                    this->output.vertices->push_back(static_cast<float>(CGAL::to_double(cell->vertex(1)->point().x())));
                    this->output.vertices->push_back(static_cast<float>(CGAL::to_double(cell->vertex(1)->point().y())));
                    this->output.vertices->push_back(static_cast<float>(CGAL::to_double(cell->vertex(1)->point().z())));

                    this->output.vertices->push_back(static_cast<float>(CGAL::to_double(cell->vertex(2)->point().x())));
                    this->output.vertices->push_back(static_cast<float>(CGAL::to_double(cell->vertex(2)->point().y())));
                    this->output.vertices->push_back(static_cast<float>(CGAL::to_double(cell->vertex(2)->point().z())));

                    this->output.vertices->push_back(static_cast<float>(CGAL::to_double(cell->vertex(3)->point().x())));
                    this->output.vertices->push_back(static_cast<float>(CGAL::to_double(cell->vertex(3)->point().y())));
                    this->output.vertices->push_back(static_cast<float>(CGAL::to_double(cell->vertex(3)->point().z())));

                    // Set face indices
                    this->output.indices->push_back(index + 0);
                    this->output.indices->push_back(index + 1);
                    this->output.indices->push_back(index + 2);

                    this->output.indices->push_back(index + 0);
                    this->output.indices->push_back(index + 2);
                    this->output.indices->push_back(index + 3);

                    this->output.indices->push_back(index + 0);
                    this->output.indices->push_back(index + 1);
                    this->output.indices->push_back(index + 3);

                    this->output.indices->push_back(index + 1);
                    this->output.indices->push_back(index + 2);
                    this->output.indices->push_back(index + 3);
                }
            }
        }

        megamol::core::utility::log::Log::DefaultLog.WriteInfo("Number of pores found: %d\n", num_pores);
    }

    return true;
}
