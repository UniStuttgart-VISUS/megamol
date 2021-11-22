/*
 * ExtractSkeleton.cpp
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ExtractSkeleton.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmadios/CallADIOSData.h"
#include "mmcore/utility/log/Log.h"
#include "probe/CallKDTree.h"
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>
#include <CGAL/convex_hull_3_to_face_graph.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <limits>


namespace megamol {
namespace probe {
typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Surface_mesh<Point> Surface_mesh;
typedef boost::graph_traits<Surface_mesh>::vertex_descriptor vertex_descriptor;
typedef CGAL::Mean_curvature_flow_skeletonization<Surface_mesh> Skeletonization;
typedef Skeletonization::Skeleton Skeleton;
typedef Skeleton::vertex_descriptor Skeleton_vertex;
typedef Skeleton::edge_descriptor Skeleton_edge;

typedef CGAL::Delaunay_triangulation_3<Kernel> Delaunay;


ExtractSkeleton::ExtractSkeleton() : Module(), _getDataCall("getData", ""), _deployLineCall("deployCenterline", "") {

    this->_deployLineCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &ExtractSkeleton::getData);
    this->_deployLineCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &ExtractSkeleton::getMetaData);
    this->MakeSlotAvailable(&this->_deployLineCall);


    this->_getDataCall.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->_getDataCall);
}

ExtractSkeleton::~ExtractSkeleton() {
    this->Release();
}

bool ExtractSkeleton::create() {
    return true;
}

void ExtractSkeleton::release() {}

bool ExtractSkeleton::InterfaceIsDirty() {
    return false;
}

bool ExtractSkeleton::compute(float* vertices, uint32_t num_vertices, uint32_t num_components) {


    Surface_mesh sm;
    std::list<Point> points_for_triangulation;

    // We need a valid bounding box
    if (!this->_bbox.IsBoundingBoxValid()) {
        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::min();
        float min_y = std::numeric_limits<float>::max();
        float max_y = std::numeric_limits<float>::min();
        float min_z = std::numeric_limits<float>::max();
        float max_z = std::numeric_limits<float>::min();
        for (uint32_t i = 0; i < num_vertices; i++) {
            min_x = std::min(min_x, vertices[num_components * i + 0]);
            max_x = std::max(max_x, vertices[num_components * i + 0]);
            min_y = std::min(min_y, vertices[num_components * i + 1]);
            max_y = std::max(max_y, vertices[num_components * i + 1]);
            min_z = std::min(min_z, vertices[num_components * i + 2]);
            max_z = std::max(max_z, vertices[num_components * i + 2]);
            points_for_triangulation.emplace_back(Point(
                vertices[num_components * i + 0], vertices[num_components * i + 1], vertices[num_components * i + 2]));
        }
        this->_bbox.SetBoundingBox(min_x, min_y, max_z, max_x, max_y, min_z);
    }

    Delaunay T(points_for_triangulation.begin(), points_for_triangulation.end());
    CGAL::convex_hull_3_to_face_graph(T, sm);
    // checking preconditions
    for (auto he : sm.halfedges()) {
        if (sm.is_border(he)) {
            core::utility::log::Log::DefaultLog.WriteError(
                "[ExtractSkeleton] Precondition violation. Mesh has boarder.");
        }
    }


    Skeleton skeleton;
    CGAL::extract_mean_curvature_flow_skeleton(sm, skeleton);


    _cl_indices_per_slice.clear();
    _centerline.clear();


    return true;
}

bool ExtractSkeleton::getMetaData(core::Call& call) {

    auto cl = dynamic_cast<mesh::CallMesh*>(&call);
    if (cl == nullptr)
        return false;

    auto cm = this->_getDataCall.CallAs<mesh::CallMesh>();
    if (cm == nullptr)
        return false;

    auto line_meta_data = cl->getMetaData();
    auto mesh_meta_data = cm->getMetaData();

    // get metadata from adios
    mesh_meta_data.m_frame_ID = line_meta_data.m_frame_ID;
    cm->setMetaData(mesh_meta_data);
    if (!(*cm)(1))
        return false;
    mesh_meta_data = cm->getMetaData();

    // put metadata in line call
    _bbox = mesh_meta_data.m_bboxs;
    line_meta_data.m_bboxs = mesh_meta_data.m_bboxs;
    line_meta_data.m_frame_cnt = mesh_meta_data.m_frame_cnt;
    cl->setMetaData(line_meta_data);

    return true;
}

bool ExtractSkeleton::getData(core::Call& call) {
    auto cl = dynamic_cast<mesh::CallMesh*>(&call);
    if (cl == nullptr)
        return false;

    auto cm = this->_getDataCall.CallAs<mesh::CallMesh>();
    if (cm == nullptr)
        return false;
    if (!(*cm)(0))
        return false;

    auto meta_data = cm->getMetaData();


    if (cm->hasUpdate()) {
        ++_version;

        auto data = cm->getData();

        if (data->accessMeshes().size() > 1 || data->accessMeshes().empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("[ExtractSkeleton] Cannot handle mesh");
            return false;
        }

        float* vertices = nullptr;
        uint32_t num_vertices = 0;
        uint32_t num_components = 0;
        for (auto& attrib : data->accessMeshes().begin()->second.attributes) {
            if (attrib.semantic == mesh::MeshDataAccessCollection::POSITION) {
                if (attrib.component_type != mesh::MeshDataAccessCollection::FLOAT) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[ExtractSkeleton] Cannot handle data type");
                    return false;
                }
                vertices = reinterpret_cast<float*>(attrib.data);
                num_vertices = attrib.byte_size / (mesh::MeshDataAccessCollection::getByteSize(attrib.component_type) *
                                                      attrib.component_cnt);
                num_components = attrib.component_cnt;
            }
        }

        assert(vertices != nullptr || num_vertices != 0 || num_components != 0);

        this->compute(vertices, num_vertices, num_components);

        _line_attribs.resize(1);
        _line_attribs[0].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
        _line_attribs[0].byte_size = _centerline.size() * sizeof(std::array<float, 4>);
        _line_attribs[0].component_cnt = 3;
        _line_attribs[0].stride = sizeof(std::array<float, 4>);
        _line_attribs[0].data = reinterpret_cast<uint8_t*>(_centerline.data());
        _line_attribs[0].semantic = mesh::MeshDataAccessCollection::POSITION;

        _cl_indices.resize(_centerline.size() - 1);
        std::generate(_cl_indices.begin(), _cl_indices.end(), [n = 0]() mutable { return n++; });

        _line_indices.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
        _line_indices.byte_size = _cl_indices.size() * sizeof(uint32_t);
        _line_indices.data = reinterpret_cast<uint8_t*>(_cl_indices.data());
    }

    // put data in line
    mesh::MeshDataAccessCollection line;
    std::string identifier = std::string(FullName()) + "_line";
    line.addMesh(identifier, _line_attribs, _line_indices, mesh::MeshDataAccessCollection::LINES);
    cl->setData(std::make_shared<mesh::MeshDataAccessCollection>(std::move(line)), _version);

    auto line_meta_data = cl->getMetaData();
    line_meta_data.m_bboxs = this->_bbox;
    cl->setMetaData(line_meta_data);

    return true;
}

} // namespace probe
} // namespace megamol
