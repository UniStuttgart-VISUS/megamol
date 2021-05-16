#include "stdafx.h"
#include "ExtractPores.h"

#include "mesh/MeshDataCall.h"
#include "mesh/TriangleMeshCall.h"

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>

//#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
//#include <CGAL/Mesh_criteria_3.h>
//#include <CGAL/Mesh_triangulation_3.h>
//#include <CGAL/Polyhedral_complex_mesh_domain_3.h>
//#include <CGAL/make_mesh_3.h>

#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup_extension.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>

#include <algorithm>
#include <array>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

typedef K::Point_3 Point;

typedef CGAL::Surface_mesh<Point> Mesh;
typedef CGAL::Delaunay_triangulation_3<K> Delaunay;

//typedef CGAL::Polyhedral_complex_mesh_domain_3<K, Mesh> Mesh_domain;
//typedef CGAL::Mesh_triangulation_3<Mesh_domain, K, CGAL::Sequential_tag>::type Tr;
//typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr, Mesh_domain::Corner_index, Mesh_domain::Curve_index> C3t3;
//typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;

typedef std::vector<std::size_t> CGAL_Polygon;

namespace PMP = CGAL::Polygon_mesh_processing;

megamol::flowvis::ExtractPores::ExtractPores()
        : mesh_lhs_slot("mesh_lhs_slot", "Separated surface meshes representing pores and throats.")
        , mesh_data_lhs_slot("mesh_data_lhs_slot", "Data associated with the meshes.")
        , mesh_rhs_slot("mesh_rhs_slot", "Input surface mesh of the fluid phase.")
        , input_hash(9834752) {

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

    call.SetDataHash(this->input_hash);

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

    call.SetDataHash(this->input_hash);

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
    if (input_changed) {

        // Create surface mesh from input triangles and bounding box
        std::vector<Point> points;
        Mesh mesh;

        {
            std::vector<Point> points;
            std::vector<CGAL_Polygon> polygon_vec;

            // Create initial CGAL polygon soup from bounding box
            {
                const auto& bbox = tmc.get_bounding_box();
                const auto x_min = bbox.GetLeft();
                const auto x_max = bbox.GetRight();
                const auto y_min = bbox.GetBottom();
                const auto y_max = bbox.GetTop();
                const auto z_min = bbox.GetBack();
                const auto z_max = bbox.GetFront();

                const Point lbb(x_min, y_min, z_min); // 0
                const Point rbb(x_max, y_min, z_min); // 1
                const Point ltb(x_min, y_max, z_min); // 2
                const Point rtb(x_max, y_max, z_min); // 3
                const Point lbf(x_min, y_min, z_max); // 4
                const Point rbf(x_max, y_min, z_max); // 5
                const Point ltf(x_min, y_max, z_max); // 6
                const Point rtf(x_max, y_max, z_max); // 7

                points.insert(points.end(), {lbb, rbb, ltb, rtb, lbf, rbf, ltf, rtf});

                polygon_vec.push_back({1, 0, 2});
                polygon_vec.push_back({1, 2, 3}); // back

                polygon_vec.push_back({4, 5, 7});
                polygon_vec.push_back({4, 7, 6}); // front

                polygon_vec.push_back({0, 1, 5});
                polygon_vec.push_back({0, 5, 4}); // bottom

                polygon_vec.push_back({6, 7, 3});
                polygon_vec.push_back({6, 3, 2}); // top

                polygon_vec.push_back({0, 4, 6});
                polygon_vec.push_back({0, 6, 2}); // left

                polygon_vec.push_back({5, 1, 3});
                polygon_vec.push_back({5, 3, 7}); // right
            }

            // Add input mesh with inverted normals to polygon soup


            // Create CGAL surface mesh from polygon soup
            {
                PMP::remove_isolated_points_in_polygon_soup(points, polygon_vec);
                PMP::merge_duplicate_points_in_polygon_soup(points, polygon_vec);

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

        // Tetrahedralize volume enclosed by surface mesh
        //        const std::array<std::reference_wrapper<Mesh>, 1> mesh_list{mesh};
        //        const std::array<std::pair<int, int>, 1> incidence{std::make_pair(0, 1)};
        //
        //        Mesh_domain domain(mesh_list.begin(), mesh_list.end(), incidence.begin(), incidence.end());
        //        domain.detect_features();
        //
        //        Mesh_criteria criteria /*(CGAL::parameters::edge_size = 8, CGAL::parameters::facet_angle = 25,
        //             CGAL::parameters::facet_size = 8, CGAL::parameters::facet_distance = 0.2,
        //             CGAL::parameters::cell_radius_edge_ratio = 3, CGAL::parameters::cell_size = 10)*/
        //            ;
        //
        //        C3t3 triangulation =
        //            CGAL::make_mesh_3<C3t3>(domain, criteria, CGAL::parameters::manifold(),
        //            CGAL::parameters::no_perturb());
        //
        //        if (!triangulation.is_valid()) {
        //            megamol::core::utility::log::Log::DefaultLog.WriteError(
        //                "Triangulation not successfull. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        //
        //            return false;
        //        }

        // Tetrahedralize domain
        Delaunay dt;
        dt.insert(points.begin(), points.end());

        // Classify pores by number of faces shared with the surface mesh
        

        //        for (auto cell = triangulation.triangulation().finite_cells_begin();
        //             cell != triangulation.triangulation().finite_cells_end(); ++cell) {
        //
        //            bool has_surface_facet = false;
        //
        //            for (int i = 0; i < 4; ++i) {
        //                has_surface_facet |= cell->is_facet_on_surface(i);
        //            }
        //
        //            std::cout << (has_surface_facet ? "surface" : "nope") << std::endl;
        //        }

        // Separate face-connected tetrahedra using the classification

        // Calculate properties, such as volume, surface, and surface-to-volume ratio

        // Create mesh for each pore and throat


        this->output.vertices = this->input.vertices;
        this->output.normals = this->input.normals;
        this->output.indices = this->input.indices;
    }

    return true;
}
