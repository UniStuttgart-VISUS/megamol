#include "SimplifyMesh.h"

#ifdef WITH_CGAL
#include "mesh/TriangleMeshCall.h"

#include "mmcore/param/FloatParam.h"

#include "mmcore/utility/DataHash.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_3.h>
#include <CGAL/Segment_3.h>
#include <CGAL/Surface_mesh.h>

#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup_extension.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>

#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>

#include <memory>
#include <vector>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

typedef K::Point_3 Point;
typedef K::Segment_3 Segment;

typedef CGAL::Surface_mesh<Point> Mesh;

typedef std::vector<std::size_t> CGAL_Polygon;

namespace PMP = CGAL::Polygon_mesh_processing;
namespace SMS = CGAL::Surface_mesh_simplification;

megamol::mesh::SimplifyMesh::SimplifyMesh()
        : mesh_lhs_slot("mesh_lhs_slot", "Simplified mesh.")
        , mesh_rhs_slot("mesh_rhs_slot", "Input surface mesh.")
        , stop_ratio("stop_ratio", "Ratio defining the number of resulting faces compared to the original mesh.")
        , input_hash(SimplifyMesh::GUID()) {

    // Connect input slot
    this->mesh_rhs_slot.SetCompatibleCall<mesh::TriangleMeshCall::triangle_mesh_description>();
    this->MakeSlotAvailable(&this->mesh_rhs_slot);

    // Connect output slots
    this->mesh_lhs_slot.SetCallback(
        mesh::TriangleMeshCall::ClassName(), "get_data", &SimplifyMesh::getMeshDataCallback);
    this->mesh_lhs_slot.SetCallback(
        mesh::TriangleMeshCall::ClassName(), "get_extent", &SimplifyMesh::getMeshMetaDataCallback);
    this->MakeSlotAvailable(&this->mesh_lhs_slot);

    // Initialize parameter slots
    this->stop_ratio << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->stop_ratio);
}

megamol::mesh::SimplifyMesh::~SimplifyMesh() {
    this->Release();
}

bool megamol::mesh::SimplifyMesh::create() {
    return true;
}

void megamol::mesh::SimplifyMesh::release() {}

bool megamol::mesh::SimplifyMesh::getMeshDataCallback(core::Call& _call) {
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

bool megamol::mesh::SimplifyMesh::getMeshMetaDataCallback(core::Call& _call) {
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

bool megamol::mesh::SimplifyMesh::compute() {

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

    if (compute_hash(tmc.DataHash()) != this->input_hash) {
        this->input.vertices = tmc.get_vertices();
        this->input.normals = tmc.get_normals();
        this->input.indices = tmc.get_indices();

        this->input_hash = compute_hash(tmc.DataHash());

        input_changed = true;
    }

    // Set empty output when encountering empty input
    if (this->input.vertices == nullptr) {
        this->output.vertices = nullptr;
        this->output.normals = nullptr;
        this->output.indices = nullptr;

        input_changed = false;

        return true;
    }

    // Perform computation
    if (input_changed) {
        this->output.vertices = std::make_shared<std::vector<float>>();
        this->output.normals = std::make_shared<std::vector<float>>();
        this->output.indices = std::make_shared<std::vector<unsigned int>>();

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

        // Simplify mesh
        const auto stop_ratio = this->stop_ratio.Param<core::param::FloatParam>()->Value();

        if (stop_ratio < 1.0f) {
            SMS::Count_ratio_stop_predicate<Mesh> stop(stop_ratio);
            SMS::edge_collapse(mesh, stop);
        }

        // Create output
        for (auto point_it = mesh.points().begin(); point_it != mesh.points().end(); ++point_it) {
            this->output.vertices->push_back(static_cast<float>(point_it->cartesian(0)));
            this->output.vertices->push_back(static_cast<float>(point_it->cartesian(1)));
            this->output.vertices->push_back(static_cast<float>(point_it->cartesian(2)));
        }

        for (auto face_it = mesh.faces_begin(); face_it != mesh.faces_end(); ++face_it) {
            const auto range = mesh.vertices_around_face(mesh.halfedge(*face_it));

            for (auto vert_it = range.begin(); vert_it != range.end(); ++vert_it) {
                this->output.indices->push_back(vert_it->idx());
            }
        }

        this->output.normals = nullptr;
    }

    return true;
}

SIZE_T megamol::mesh::SimplifyMesh::compute_hash(const SIZE_T data_hash) const {
    return core::utility::DataHash(
        SimplifyMesh::GUID(), data_hash, this->stop_ratio.Param<core::param::FloatParam>()->Value());
}
#else
bool megamol::mesh::SimplifyMesh::create() {
    return false;
}

void megamol::mesh::SimplifyMesh::release() {}
#endif
