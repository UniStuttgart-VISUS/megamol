#include "ParticleSurfaceRefinement.h"

#include "mmcore/param/FloatParam.h"

#include "CGAL/Exact_predicates_inexact_constructions_kernel.h"

#include "CGAL/Polygon_mesh_processing/border.h"
#include "CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h"
#include "CGAL/Polygon_mesh_processing/repair.h"
#include "CGAL/Polygon_mesh_processing/triangulate_hole.h"
#include "CGAL/Surface_mesh.h"

#include "CGAL/Kernel_traits.h"
#include "CGAL/boost/graph/properties.h"

#include "boost/cstdint.hpp"
#include "boost/graph/graph_traits.hpp"


megamol::thermodyn::ParticleSurfaceRefinement::ParticleSurfaceRefinement()
        : data_out_slot_("dataOut", "")
        , data_in_slot_("dataIn", "")
        , volume_thres_slot_("volume threshold", "")
        , area_thres_slot_("area threshold", "") {
    data_out_slot_.SetCallback<mesh::CallMesh, 0>(&ParticleSurfaceRefinement::get_data_cb);
    data_out_slot_.SetCallback<mesh::CallMesh, 1>(&ParticleSurfaceRefinement::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&data_in_slot_);

    volume_thres_slot_ << new core::param::FloatParam(0.01f, 0.0f, 1.0f);
    MakeSlotAvailable(&volume_thres_slot_);

    area_thres_slot_ << new core::param::FloatParam(0.01f, 0.0f, 1.0f);
    MakeSlotAvailable(&area_thres_slot_);
}


megamol::thermodyn::ParticleSurfaceRefinement::~ParticleSurfaceRefinement() {
    this->Release();
}


bool megamol::thermodyn::ParticleSurfaceRefinement::create() {
    return true;
}


void megamol::thermodyn::ParticleSurfaceRefinement::release() {}


bool megamol::thermodyn::ParticleSurfaceRefinement::get_data_cb(core::Call& c) {
    auto data_out = dynamic_cast<mesh::CallMesh*>(&c);
    if (data_out == nullptr)
        return false;

    auto data_in = data_in_slot_.CallAs<mesh::CallMesh>();
    if (data_in == nullptr)
        return false;

    auto meta = data_out->getMetaData();
    auto in_meta = data_in->getMetaData();
    in_meta.m_frame_ID = meta.m_frame_ID;
    data_in->setMetaData(in_meta);
    if (!(*data_in)(0))
        return false;

    if (data_in->hasUpdate() || is_dirty()) {
        if (!assert_data(*data_in))
            return false;

        reset_dirty();
        ++version_;
    }

    data_out->setData(mesh_col_, version_);

    return true;
}


bool megamol::thermodyn::ParticleSurfaceRefinement::get_extent_cb(core::Call& c) {
    auto data_out = dynamic_cast<mesh::CallMesh*>(&c);
    if (data_out == nullptr)
        return false;

    auto data_in = data_in_slot_.CallAs<mesh::CallMesh>();
    if (data_in == nullptr)
        return false;

    auto meta = data_out->getMetaData();
    auto in_meta = data_in->getMetaData();
    in_meta.m_frame_ID = meta.m_frame_ID;
    data_in->setMetaData(in_meta);
    if (!(*data_in)(1))
        return false;

    data_out->setMetaData(data_in->getMetaData());

    return true;
}


bool megamol::thermodyn::ParticleSurfaceRefinement::assert_data(mesh::CallMesh& meshes) {
    auto const data = meshes.getData()->accessMeshes();

    auto const mesh_count = data.size();

    mesh_col_ = std::make_shared<mesh::MeshDataAccessCollection>();

    indices_.resize(mesh_count);
    vertices_.resize(mesh_count);
    normals_.resize(mesh_count);

    auto data_idx = 0;
    for (auto const& mesh_el : data) {
        auto const& mesh = mesh_el.second;

        auto& indices = indices_[data_idx];
        auto& vertices = vertices_[data_idx];
        auto& normals = normals_[data_idx];

        auto const idx_ptr = reinterpret_cast<glm::uvec3*>(mesh.indices.data);
        auto const idx_count = mesh.indices.byte_size / sizeof(glm::uvec3);

        auto const fit = std::find_if(mesh.attributes.begin(), mesh.attributes.end(), [](auto const& el) {
            return el.semantic == mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION;
        });

        if (fit == mesh.attributes.end())
            return false;

        auto const pos_ptr = reinterpret_cast<glm::vec3*>(fit->data + fit->offset);
        auto const pos_count = fit->byte_size / fit->stride;

        using Gt = CGAL::Exact_predicates_inexact_constructions_kernel;
        using Point3 = CGAL::Point_3<Gt>;

        std::vector<std::vector<std::size_t>> pindices(idx_count);
        for (uint64_t i = 0; i < idx_count; ++i) {
            pindices[i] = {idx_ptr[i].x, idx_ptr[i].y, idx_ptr[i].z};
        }

        std::vector<Point3> ppoints(pos_count);
        for (uint64_t i = 0; i < pos_count; ++i) {
            ppoints[i] = Point3(pos_ptr[i].x, pos_ptr[i].y, pos_ptr[i].z);
        }

        auto is_mesh = CGAL::Polygon_mesh_processing::is_polygon_soup_a_polygon_mesh(pindices);

        if (!is_mesh) {
            CGAL::Polygon_mesh_processing::orient_polygon_soup(ppoints, pindices);
        }

        using TriangleMesh = CGAL::Surface_mesh<Point3>;

        TriangleMesh poly;
        CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(ppoints, pindices, poly);

        CGAL::Polygon_mesh_processing::remove_connected_components_of_negligible_size(
            poly, CGAL::parameters::volume_threshold(2000.0f).area_threshold(3000.0f));

        using vertex_descriptor = boost::graph_traits<CGAL::Surface_mesh<Point3>>::vertex_descriptor;
        using halfedge_descriptor = boost::graph_traits<CGAL::Surface_mesh<Point3>>::halfedge_descriptor;
        using face_descriptor = boost::graph_traits<CGAL::Surface_mesh<Point3>>::face_descriptor;

        std::list<halfedge_descriptor> border_cycles;

        CGAL::Polygon_mesh_processing::extract_boundary_cycles(poly, std::back_inserter(border_cycles));

        for (auto const& h : border_cycles) {
            std::list<face_descriptor> patch_facets;
            std::list<vertex_descriptor> patch_vertices;
            CGAL::Polygon_mesh_processing::triangulate_refine_and_fair_hole(
                poly, h, std::back_inserter(patch_facets), std::back_inserter(patch_vertices));
        }

        indices.clear();
        indices.reserve(poly.number_of_faces());
        vertices.clear();
        vertices.reserve(poly.number_of_vertices());
        normals.clear();
        normals.reserve(poly.number_of_vertices());
        {
            using namespace CGAL;
            typedef typename boost::graph_traits<TriangleMesh>::face_descriptor face_descriptor;
            typedef typename boost::graph_traits<TriangleMesh>::halfedge_descriptor halfedge_descriptor;
            typedef typename boost::property_map<TriangleMesh, boost::vertex_point_t>::const_type Vpm;
            typedef typename boost::property_traits<Vpm>::reference Point_3_ref;
            typedef typename boost::property_traits<Vpm>::value_type Point_3;
            typedef typename Kernel_traits<Point_3>::Kernel::Vector_3 Vector_3;

            Vpm vpm = get(boost::vertex_point, poly);

            uint32_t v_idx = 0;
            for (face_descriptor f : faces(poly)) {
                halfedge_descriptor h = halfedge(f, poly);
                Point_3_ref p = get(vpm, target(h, poly));
                Point_3_ref q = get(vpm, target(next(h, poly), poly));
                Point_3_ref r = get(vpm, source(h, poly));

                Vector_3 n = collinear(p, q, r) ? Vector_3(1, 0, 0) : unit_normal(p, q, r);

                vertices.push_back(glm::vec3(p.x(), p.y(), p.z()));
                vertices.push_back(glm::vec3(q.x(), q.y(), q.z()));
                vertices.push_back(glm::vec3(r.x(), r.y(), r.z()));
                normals.push_back(glm::vec3(n.x(), n.y(), n.z()));
                normals.push_back(glm::vec3(n.x(), n.y(), n.z()));
                normals.push_back(glm::vec3(n.x(), n.y(), n.z()));
                indices.push_back(v_idx++);
                indices.push_back(v_idx++);
                indices.push_back(v_idx++);
            }
        }

        mesh::MeshDataAccessCollection::IndexData index_data;
        index_data.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
        index_data.byte_size = indices.size() * sizeof(std::decay_t<decltype(indices)>::value_type);
        index_data.data = reinterpret_cast<decltype(index_data.data)>(indices.data());

        std::vector<mesh::MeshDataAccessCollection::VertexAttribute> attributes = {
            {reinterpret_cast<decltype(mesh::MeshDataAccessCollection::VertexAttribute::data)>(vertices.data()),
                vertices.size() * sizeof(std::decay_t<decltype(vertices)>::value_type), 3,
                mesh::MeshDataAccessCollection::ValueType::FLOAT, sizeof(std::decay_t<decltype(vertices)>::value_type),
                0, mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION},
            {reinterpret_cast<decltype(mesh::MeshDataAccessCollection::VertexAttribute::data)>(normals.data()),
                normals.size() * sizeof(std::decay_t<decltype(normals)>::value_type), 3,
                mesh::MeshDataAccessCollection::ValueType::FLOAT, sizeof(std::decay_t<decltype(normals)>::value_type),
                0, mesh::MeshDataAccessCollection::AttributeSemanticType::NORMAL}};

        mesh_col_->addMesh(mesh_el.first, attributes, index_data);
    }

    return true;
}
