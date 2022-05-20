#include "ExtractPores.h"

#include "mesh/MeshDataCall.h"
#include "mesh/TriangleMeshCall.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/utility/DataHash.h"

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_3.h>
#include <CGAL/Segment_3.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup_extension.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>

#include <algorithm>
#include <array>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

typedef K::Point_3 Point;
typedef K::Segment_3 Segment;

typedef CGAL::Surface_mesh<Point> Mesh;

enum class pore_type {
    none = -1,
    original,
    between
};

typedef CGAL::Triangulation_vertex_base_with_info_3<CGAL::SM_Vertex_index, K> Vb;
typedef CGAL::Triangulation_cell_base_with_info_3<std::pair<pore_type, int>, K> Cb;
typedef CGAL::Delaunay_triangulation_cell_base_3<K, Cb> Dcb;
typedef CGAL::Triangulation_data_structure_3<Vb, Dcb> Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds> Delaunay;

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
        , boundary_offset("boundary_offset", "Boundary width where extracted pores are ignored.")
        , neighborhood_size("neighborhood_size", "Neighborhood size for which vertices are checked to belong to the same mesh.")
        , tf_type("type::tf_type", "Transfer function for 'type' data.")
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

    this->boundary_offset << new core::param::FloatParam(0.0);
    this->MakeSlotAvailable(&this->boundary_offset);

    this->neighborhood_size << new core::param::IntParam(1, 1);
    this->MakeSlotAvailable(&this->neighborhood_size);

    this->tf_type << new core::param::TransferFunctionParam();
    this->MakeSlotAvailable(&this->tf_type);
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

    call.SetDataHash(core::utility::DataHash(ExtractPores::GUID(), this->input_hash,
        this->pore_criterion.Param<core::param::EnumParam>()->Value(),
        this->boundary_offset.Param<core::param::FloatParam>()->Value(),
        this->neighborhood_size.Param<core::param::IntParam>()->Value()));

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

    call.SetDataHash(core::utility::DataHash(ExtractPores::GUID(), this->input_hash,
        this->pore_criterion.Param<core::param::EnumParam>()->Value(),
        this->boundary_offset.Param<core::param::FloatParam>()->Value(),
        this->neighborhood_size.Param<core::param::IntParam>()->Value()));

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
    // type                         - The type of the pore
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
    if (input_changed || this->pore_criterion.IsDirty() || this->boundary_offset.IsDirty() ||
        this->neighborhood_size.IsDirty()) {

        this->output.vertices = std::make_shared<std::vector<float>>();
        this->output.normals = nullptr;
        this->output.indices = std::make_shared<std::vector<unsigned int>>();

        this->pore_criterion.ResetDirty();
        this->boundary_offset.ResetDirty();
        this->neighborhood_size.ResetDirty();

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
        std::vector<std::pair<Point, CGAL::SM_Vertex_index>> mesh_points_with_info;

        for (auto vert_it = mesh.vertices().begin(); vert_it != mesh.vertices().end(); ++vert_it) {
            mesh_points_with_info.push_back(std::make_pair(mesh.point(*vert_it), *vert_it));
        }

        Delaunay dt;
        dt.insert(mesh_points_with_info.begin(), mesh_points_with_info.end());

        for (auto cell = dt.finite_cells_begin(); cell != dt.finite_cells_end(); ++cell) {
            cell->info().first = pore_type::none;
            cell->info().second = 0;
        }

        // Compute "clip" box
        const auto& bb = tmc.get_bounding_box();

        const vislib::math::Cuboid<float> cp(
            bb.Left() + this->boundary_offset.Param<core::param::FloatParam>()->Value(),
            bb.Bottom() + this->boundary_offset.Param<core::param::FloatParam>()->Value(),
            bb.Back() + this->boundary_offset.Param<core::param::FloatParam>()->Value(),
            bb.Right() - this->boundary_offset.Param<core::param::FloatParam>()->Value(),
            bb.Top() - this->boundary_offset.Param<core::param::FloatParam>()->Value(),
            bb.Front() - this->boundary_offset.Param<core::param::FloatParam>()->Value());

        auto inside = [&cp](const Point& p) -> bool {
            return cp.Contains(vislib::math::Point<float, 3>(static_cast<float>(CGAL::to_double(p[0])),
                static_cast<float>(CGAL::to_double(p[1])), static_cast<float>(CGAL::to_double(p[2]))));
        };

        // Iterate over all cells and classify them as pores
        std::size_t num_pores = 0;

        CGAL::Side_of_triangle_mesh<Mesh, K> orientation_test(mesh);

        for (auto cell = dt.finite_cells_begin(); cell != dt.finite_cells_end(); ++cell) {
            // Ignore cells within the mesh (solid) or too close to the boundary of the domain (ill-shaped)
            const auto cell_center = CGAL::ORIGIN +
                (0.25 * ((cell->vertex(0)->point() - CGAL::ORIGIN) + (cell->vertex(1)->point() - CGAL::ORIGIN) +
                            (cell->vertex(2)->point() - CGAL::ORIGIN) + (cell->vertex(3)->point() - CGAL::ORIGIN)));

            if (orientation_test(cell_center) == CGAL::ON_BOUNDED_SIDE ||
                !(inside(cell->vertex(0)->point()) || inside(cell->vertex(1)->point()) ||
                    inside(cell->vertex(2)->point()) || inside(cell->vertex(3)->point()))) {

                continue;
            }

            // Count number of edges shared with the surface mesh
            int num_shared_segments = 0;

            for (int i = 0; i < 4; ++i) {
                // For each vertex of the tetrahedron, get neighboring vertices with given maximum edge distance
                std::unordered_set<CGAL::SM_Vertex_index> all_vertices;
                all_vertices.insert(cell->vertex(i)->info());

                for (int n = 0; n < neighborhood_size.Param<core::param::IntParam>()->Value(); ++n) {
                    std::vector<CGAL::SM_Vertex_index> vertices;

                    for (auto it = all_vertices.begin(); it != all_vertices.end(); ++it) {
                        auto vertex_range = mesh.vertices_around_target(mesh.halfedge(*it));

                        for (auto vertex_it = vertex_range.begin(); vertex_it != vertex_range.end(); ++vertex_it) {
                            vertices.push_back(*vertex_it);
                        }
                    }

                    all_vertices.insert(vertices.begin(), vertices.end());
                }

                // Check how many of the other vertices are in the neighborhood
                for (int j = 0; j < 4; ++j) {
                    if (i != j) {
                        if (all_vertices.find(cell->vertex(j)->info()) != all_vertices.end()) {
                            ++num_shared_segments;
                        }
                    }
                }
            }

            // If [no | at most one] edge is shared with the surface mesh, then this is a pore
            const auto max_num_shared_segments =
                (this->pore_criterion.Param<core::param::EnumParam>()->Value() == 0) ? 0 : 1;

            if (num_shared_segments <= max_num_shared_segments) {
                cell->info().first = pore_type::original;
            }
        }

        // Iteratively classify additional pores lying in between pore tetrahedra
        bool change = false;

        do {
            change = false;

            for (auto cell = dt.finite_cells_begin(); cell != dt.finite_cells_end(); ++cell) {
                cell->info().second = 0;
            }

            // Count for each tetrahedron the number of face-connected pore tetrahedra
            for (auto cell = dt.finite_cells_begin(); cell != dt.finite_cells_end(); ++cell) {
                if (cell->info().first != pore_type::none) {
                    for (int i = 0; i < 4; ++i) {
                        if (dt.is_cell(cell->neighbor(i)) && !dt.is_infinite(cell->neighbor(i))) {
                            ++cell->neighbor(i)->info().second;
                        }
                    }
                }
            }

            // Classify tetrahedra as pore when connected to at least two pore tetrahedra
            for (auto cell = dt.finite_cells_begin(); cell != dt.finite_cells_end(); ++cell) {
                if (cell->info().first == pore_type::none && cell->info().second > 1) {
                    cell->info().first = pore_type::between;

                    change = true;
                }
            }
        } while (change);

        // Add pore tetrahedra to output
        this->output.datasets[3] = std::make_shared<mesh::MeshDataCall::data_set>();
        this->output.datasets[3]->min_value = 0;
        this->output.datasets[3]->max_value = 1;
        this->output.datasets[3]->data = std::make_shared<std::vector<float>>();

        for (auto cell = dt.finite_cells_begin(); cell != dt.finite_cells_end(); ++cell) {
            if (cell->info().first != pore_type::none) {
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

                // Set pore information
                this->output.datasets[3]->data->push_back(static_cast<float>(cell->info().first));
                this->output.datasets[3]->data->push_back(static_cast<float>(cell->info().first));
                this->output.datasets[3]->data->push_back(static_cast<float>(cell->info().first));
                this->output.datasets[3]->data->push_back(static_cast<float>(cell->info().first));
            }
        }

        megamol::core::utility::log::Log::DefaultLog.WriteInfo("Number of pores found: %d\n", num_pores);
    }

    // Set transfer function
    if ((this->tf_type.IsDirty() || input_changed) && this->output.datasets[3] != nullptr) {
        this->tf_type.ResetDirty();

        this->output.datasets[3]->transfer_function =
            this->tf_type.Param<core::param::TransferFunctionParam>()->Value();
        this->output.datasets[3]->transfer_function_dirty = true;
    }

    return true;
}
