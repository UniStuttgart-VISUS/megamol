/*
 * PlaceProbes.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "PlaceProbes.h"
#include <random>
#include "ProbeCalls.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"

megamol::probe::PlaceProbes::PlaceProbes()
    : Module()
    , m_version(0)
    , m_mesh_slot("getMesh", "")
    , m_probe_slot("deployProbes", "")
    , m_centerline_slot("getCenterLine", "")
    , m_method_slot("method", "")
    , m_probes_per_unit_slot("Probes_per_unit", "Sets the average probe count per unit area") {

    this->m_probe_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(0), &PlaceProbes::getData);
    this->m_probe_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(1), &PlaceProbes::getMetaData);
    this->MakeSlotAvailable(&this->m_probe_slot);

    this->m_mesh_slot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->m_mesh_slot);

    this->m_centerline_slot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->m_centerline_slot);

    core::param::EnumParam* ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "vertices");
    ep->SetTypePair(1, "dart_throwing");
    ep->SetTypePair(2, "force_directed");
    ep->SetTypePair(3, "simple_wruschd_sampling");
    ep->SetTypePair(4, "vertices+normals");
    this->m_method_slot << ep;
    this->MakeSlotAvailable(&this->m_method_slot);

    this->m_probes_per_unit_slot << new core::param::IntParam(1,0);

    /* Feasibility test */
    m_probes = std::make_shared<ProbeCollection>();
    m_probes->addProbe(FloatProbe());

    auto retrieved_probe = m_probes->getProbe<FloatProbe>(0);

    float data;
    retrieved_probe.probe(&data);

    auto result = retrieved_probe.getSamplingResult();
}

megamol::probe::PlaceProbes::~PlaceProbes() { this->Release(); }

bool megamol::probe::PlaceProbes::create() { return true; }

void megamol::probe::PlaceProbes::release() {}

bool megamol::probe::PlaceProbes::getData(core::Call& call) {

    auto* pc = dynamic_cast<CallProbes*>(&call);
    mesh::CallMesh* cm = this->m_mesh_slot.CallAs<mesh::CallMesh>();
    mesh::CallMesh* ccl = this->m_centerline_slot.CallAs<mesh::CallMesh>();

    if (cm == nullptr || ccl == nullptr) return false;

    if (!(*cm)(0)) return false;
    if (!(*ccl)(0)) return false;

    bool something_changed = cm->hasUpdate() || ccl->hasUpdate();

    auto mesh_meta_data = cm->getMetaData();
    auto probe_meta_data = pc->getMetaData();
    auto centerline_meta_data = ccl->getMetaData();

    probe_meta_data.m_bboxs = mesh_meta_data.m_bboxs;

    m_mesh = cm->getData();
    m_centerline = ccl->getData();

    // here something really happens
    if (something_changed) {
        ++m_version;
        
        if (mesh_meta_data.m_bboxs.IsBoundingBoxValid()) {
            m_whd = {mesh_meta_data.m_bboxs.BoundingBox().Width(), mesh_meta_data.m_bboxs.BoundingBox().Height(),
            mesh_meta_data.m_bboxs.BoundingBox().Depth()};
        } else if (centerline_meta_data.m_bboxs.IsBoundingBoxValid()) {
            m_whd = {centerline_meta_data.m_bboxs.BoundingBox().Width(), centerline_meta_data.m_bboxs.BoundingBox().Height(),
                centerline_meta_data.m_bboxs.BoundingBox().Depth()};
        }
        const auto longest_edge_index = std::distance(m_whd.begin(), std::max_element(m_whd.begin(), m_whd.end()));

        this->placeProbes(longest_edge_index);
    }

    pc->setData(this->m_probes,m_version);

    pc->setMetaData(probe_meta_data);
    return true;
}

bool megamol::probe::PlaceProbes::getMetaData(core::Call& call) {

    auto* pc = dynamic_cast<CallProbes*>(&call);
    mesh::CallMesh* cm = this->m_mesh_slot.CallAs<mesh::CallMesh>();
    mesh::CallMesh* ccl = this->m_centerline_slot.CallAs<mesh::CallMesh>();

    if (cm == nullptr || ccl == nullptr) return false;

    // set frame id before callback
    auto mesh_meta_data = cm->getMetaData();
    auto probe_meta_data = pc->getMetaData();
    auto centerline_meta_data = ccl->getMetaData();

    mesh_meta_data.m_frame_ID = probe_meta_data.m_frame_ID;
    centerline_meta_data.m_frame_ID = probe_meta_data.m_frame_ID;

    cm->setMetaData(mesh_meta_data);
    ccl->setMetaData(centerline_meta_data);

    if (!(*cm)(1)) return false;
    if (!(*ccl)(1)) return false;
    
    mesh_meta_data = cm->getMetaData();
    centerline_meta_data = ccl->getMetaData();

    probe_meta_data.m_frame_cnt = mesh_meta_data.m_frame_cnt;
    probe_meta_data.m_bboxs = mesh_meta_data.m_bboxs; // normally not available here
    
    pc->setMetaData(probe_meta_data);

    return true;
}

void megamol::probe::PlaceProbes::dartSampling(mesh::MeshDataAccessCollection::VertexAttribute& vertices,
    std::vector<std::array<float, 4>>& output, mesh::MeshDataAccessCollection::IndexData indexData,
    float distanceIndicator) {

    uint32_t num_triangles = indexData.byte_size / (mesh::MeshDataAccessCollection::getByteSize(indexData.type) * 3);
    auto indexDataAccessor = reinterpret_cast<uint32_t*>(indexData.data);
    auto vertexAccessor = reinterpret_cast<float*>(vertices.data);
    auto num_verts = vertices.byte_size / vertices.stride;

    std::mt19937 rnd;
    rnd.seed(std::random_device()());
    //rnd.seed(666);
    std::uniform_real_distribution<float> fltdist(0, 1);
    std::uniform_int_distribution<uint32_t> dist(0, num_triangles - 1);

    uint32_t num_probes = 4000;
    output.reserve(num_probes);

    uint32_t indx = 0;
    uint32_t error_index = 0;
    while (indx != (num_probes - 1) && error_index < num_probes){//&& error_index < (num_triangles - indx)) {

        uint32_t triangle = dist(rnd);
        std::array<float, 3> vert0;
        vert0[0] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 0] + 0];
        vert0[1] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 0] + 1];
        vert0[2] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 0] + 2];

        std::array<float, 3> vert1;
        vert1[0] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 1] + 0];
        vert1[1] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 1] + 1];
        vert1[2] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 1] + 2];

        std::array<float, 3> vert2;
        vert2[0] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 2] + 0];
        vert2[1] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 2] + 1];
        vert2[2] = vertexAccessor[3 * indexDataAccessor[3 * triangle + 2] + 2];


        auto rnd1 = fltdist(rnd);
        auto rnd2 = fltdist(rnd);
        rnd2 *= (1 - rnd1);
        float rnd3 = 1 - (rnd1 + rnd2);

        std::array<float, 3> triangle_middle;
        triangle_middle[0] = (rnd1*vert0[0] + rnd2*vert1[0] + rnd3*vert2[0]) ;
        triangle_middle[1] = (rnd1*vert0[1] + rnd2*vert1[1] + rnd3*vert2[1]) ;
        triangle_middle[2] = (rnd1*vert0[2] + rnd2*vert1[2] + rnd3*vert2[2]) ;

        bool do_placement = true;
        for (uint32_t j = 0; j < indx; j++) {
            std::array<float, 3> dist_vec;
            dist_vec[0] = triangle_middle[0] - output[j][0];
            dist_vec[1] = triangle_middle[1] - output[j][1];
            dist_vec[2] = triangle_middle[2] - output[j][2];

            const float distance =
                std::sqrt(dist_vec[0] * dist_vec[0] + dist_vec[1] * dist_vec[1] + dist_vec[2] * dist_vec[2]);

            if (distance <= distanceIndicator) {
                do_placement = false;
                break;
            }
        }

        if (do_placement) {
            output.emplace_back(std::array<float,4>({triangle_middle[0], triangle_middle[1], triangle_middle[2], 1.0f}));
            indx++;
            error_index = 0;
        } else {
            error_index++;
        }
    }
    output.shrink_to_fit();
}

void megamol::probe::PlaceProbes::forceDirectedSampling(
    mesh::MeshDataAccessCollection::VertexAttribute& vertices, std::vector<std::array<float, 4>>& output) {






}


void megamol::probe::PlaceProbes::vertexSampling(
    mesh::MeshDataAccessCollection::VertexAttribute& vertices, std::vector<std::array<float, 4>>& output) {


    uint32_t probe_count = vertices.byte_size / vertices.stride;
    output.resize(probe_count);

    auto vertex_accessor = reinterpret_cast<float*>(vertices.data);
    auto vertex_step = vertices.stride / sizeof(float);

#pragma omp parallel for
    for (int i = 0; i < probe_count; i++) {
        output[i][0] = vertex_accessor[vertex_step * i + 0];
        output[i][1] = vertex_accessor[vertex_step * i + 1];
        output[i][2] = vertex_accessor[vertex_step * i + 2];
        output[i][3] = 1.0f;
    }
}

void megamol::probe::PlaceProbes::vertexNormalSampling(
    mesh::MeshDataAccessCollection::VertexAttribute& vertices,
    mesh::MeshDataAccessCollection::VertexAttribute& normals)
{

    uint32_t probe_count = vertices.byte_size / vertices.stride;

    auto vertex_accessor = reinterpret_cast<float*>(vertices.data);
    auto vertex_step = vertices.stride / sizeof(float);

    auto normal_accessor = reinterpret_cast<float*>(normals.data);
    auto normal_step = normals.stride / sizeof(float);

//#pragma omp parallel for
    for (int i = 0; i < probe_count; i++) {

        BaseProbe probe;

        probe.m_position = {
            vertex_accessor[vertex_step * i + 0],
            vertex_accessor[vertex_step * i + 1],
            vertex_accessor[vertex_step * i + 2]};
        probe.m_direction = {
            normal_accessor[normal_step * i + 0],
            normal_accessor[normal_step * i + 1],
            normal_accessor[normal_step * i + 2]};
        probe.m_begin = -2.0;
        probe.m_end = 50.0;

        this->m_probes->addProbe(std::move(probe));
    }
}

bool megamol::probe::PlaceProbes::placeProbes(uint32_t lei) {


    m_probes = std::make_shared<ProbeCollection>();


    assert(m_mesh->accessMesh().size() == 1);
    assert(m_centerline->accessMesh().size() == 1);


    mesh::MeshDataAccessCollection::VertexAttribute vertices;
    mesh::MeshDataAccessCollection::VertexAttribute centerline;

    for (auto& attribute : m_mesh->accessMesh()[0].attributes) {
        if (attribute.semantic == mesh::MeshDataAccessCollection::POSITION) {
            vertices = attribute;
        }
    }

    for (auto& attribute : m_centerline->accessMesh()[0].attributes) {
        if (attribute.semantic == mesh::MeshDataAccessCollection::POSITION) {
            centerline = attribute;
        }
    }


    std::vector<std::array<float, 4>> probePositions;

    const auto smallest_edge_index = std::distance(m_whd.begin(), std::min_element(m_whd.begin(), m_whd.end()));
    const float distanceIndicator = m_whd[smallest_edge_index] / 20;

    if (this->m_method_slot.Param<core::param::EnumParam>()->Value() == 0) {
        this->vertexSampling(vertices, probePositions);
    } else if (this->m_method_slot.Param<core::param::EnumParam>()->Value() == 1) {
        this->dartSampling(vertices, probePositions, m_mesh->accessMesh()[0].indices, distanceIndicator);
    } else if (this->m_method_slot.Param<core::param::EnumParam>()->Value() == 2) {
        this->forceDirectedSampling(vertices, probePositions);
    }

    if (this->m_method_slot.Param<core::param::EnumParam>()->Value() == 4) {

        mesh::MeshDataAccessCollection::VertexAttribute normals;
        for (auto& attribute : m_mesh->accessMesh()[0].attributes) {
            if (attribute.semantic == mesh::MeshDataAccessCollection::NORMAL) {
                normals = attribute;
            }
        }

        this->vertexNormalSampling(vertices, normals);
    }
    else
    {
        this->placeByCenterline(lei, probePositions, centerline);
    }

    return true;
}

bool megamol::probe::PlaceProbes::placeByCenterline(uint32_t lei, std::vector<std::array<float, 4>>& probePositions,
    mesh::MeshDataAccessCollection::VertexAttribute& centerline) {


    uint32_t probe_count = probePositions.size();
    // uint32_t probe_count = 1;
    uint32_t centerline_vert_count = centerline.byte_size / centerline.stride;

    auto vertex_accessor = reinterpret_cast<float*>(probePositions.data()->data());
    auto centerline_accessor = reinterpret_cast<float*>(centerline.data);

    auto vertex_step = 4;
    auto centerline_step = centerline.stride / sizeof(centerline.component_type);
    
    for (uint32_t i = 0; i < probe_count; i++) {
        BaseProbe probe;

        std::vector<float> distances(centerline_vert_count);
        for (uint32_t j = 0; j < centerline_vert_count; j++) {
            // std::array<float, 3> diffvec = {
            //    vertex_accessor[vertex_step * i + 0] - centerline_accessor[centerline_step * j + 0],
            //    vertex_accessor[vertex_step * i + 1] - centerline_accessor[centerline_step * j + 1],
            //    vertex_accessor[vertex_step * i + 2] - centerline_accessor[centerline_step * j + 2]};
            // distances[j] = std::sqrt(diffvec[0] * diffvec[0] + diffvec[1] * diffvec[1] + diffvec[2] * diffvec[2]);
            distances[j] =
                std::abs(vertex_accessor[vertex_step * i + lei] - centerline_accessor[centerline_step * j + lei]);
        }

        auto min_iter = std::min_element(distances.begin(), distances.end());
        auto min_index = std::distance(distances.begin(), min_iter);
        distances[min_index] = std::numeric_limits<float>::max();

        auto second_min_iter = std::min_element(distances.begin(), distances.end());
        auto second_min_index = std::distance(distances.begin(), second_min_iter);

        // calc normal in plane between vert, min and second_min
        std::array<float, 3> along_centerline;
        along_centerline[0] = centerline_accessor[centerline_step * min_index + 0] -
                              centerline_accessor[centerline_step * second_min_index + 0];
        along_centerline[1] = centerline_accessor[centerline_step * min_index + 1] -
                              centerline_accessor[centerline_step * second_min_index + 1];
        along_centerline[2] = centerline_accessor[centerline_step * min_index + 2] -
                              centerline_accessor[centerline_step * second_min_index + 2];

        std::array<float, 3> min_to_vert;
        min_to_vert[0] = vertex_accessor[vertex_step * i + 0] - centerline_accessor[centerline_step * min_index + 0];
        min_to_vert[1] = vertex_accessor[vertex_step * i + 1] - centerline_accessor[centerline_step * min_index + 1];
        min_to_vert[2] = vertex_accessor[vertex_step * i + 2] - centerline_accessor[centerline_step * min_index + 2];

        std::array<float, 3> bitangente;
        bitangente[0] = along_centerline[1] * min_to_vert[2] - along_centerline[2] * min_to_vert[1];
        bitangente[1] = along_centerline[2] * min_to_vert[0] - along_centerline[0] * min_to_vert[2];
        bitangente[2] = along_centerline[0] * min_to_vert[1] - along_centerline[1] * min_to_vert[0];

        std::array<float, 3> normal;
        normal[0] = along_centerline[1] * bitangente[2] - along_centerline[2] * bitangente[1];
        normal[1] = along_centerline[2] * bitangente[0] - along_centerline[0] * bitangente[2];
        normal[2] = along_centerline[0] * bitangente[1] - along_centerline[1] * bitangente[0];

        // normalize normal
        float normal_length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        normal[0] /= normal_length;
        normal[1] /= normal_length;
        normal[2] /= normal_length;

        // do the projection
        float final_dist = normal[0] * min_to_vert[0] + normal[1] * min_to_vert[1] + normal[2] * min_to_vert[2];

        // flip normal to point inwards
        if (final_dist > 0) {
            normal[0] *= -1;
            normal[1] *= -1;
            normal[2] *= -1;
        } else {
            final_dist *= -1;
        }

        // Do clamping if end pos is not between center line points
        std::array<float, 3> end_pos;
        end_pos[0] = vertex_accessor[vertex_step * i + 0] + final_dist * normal[0];
        end_pos[1] = vertex_accessor[vertex_step * i + 1] + final_dist * normal[1];
        end_pos[2] = vertex_accessor[vertex_step * i + 2] + final_dist * normal[2];

        std::array<float, 3> end_pos_to_min;
        end_pos_to_min[0] = centerline_accessor[centerline_step * min_index + 0] - end_pos[0];
        end_pos_to_min[1] = centerline_accessor[centerline_step * min_index + 1] - end_pos[1];
        end_pos_to_min[2] = centerline_accessor[centerline_step * min_index + 2] - end_pos[2];

        std::array<float, 3> end_pos_to_second_min;
        end_pos_to_second_min[0] = centerline_accessor[centerline_step * second_min_index + 0] - end_pos[0];
        end_pos_to_second_min[1] = centerline_accessor[centerline_step * second_min_index + 1] - end_pos[1];
        end_pos_to_second_min[2] = centerline_accessor[centerline_step * second_min_index + 2] - end_pos[2];

        const float between_check = end_pos_to_min[0] * end_pos_to_second_min[0] +
                                    end_pos_to_min[1] * end_pos_to_second_min[1] +
                                    end_pos_to_min[2] * end_pos_to_second_min[2];

        if (between_check > 0) {
            normal[0] = centerline_accessor[centerline_step * min_index + 0] - vertex_accessor[vertex_step * i + 0];
            normal[1] = centerline_accessor[centerline_step * min_index + 1] - vertex_accessor[vertex_step * i + 1];
            normal[2] = centerline_accessor[centerline_step * min_index + 2] - vertex_accessor[vertex_step * i + 2];

            // normalize normal
            normal_length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
            normal[0] /= normal_length;
            normal[1] /= normal_length;
            normal[2] /= normal_length;
        }

        probe.m_position = {vertex_accessor[vertex_step * i + 0], vertex_accessor[vertex_step * i + 1],
            vertex_accessor[vertex_step * i + 2]};
        probe.m_direction = normal;
        probe.m_begin = -0.1 * final_dist;
        probe.m_end = final_dist;

        this->m_probes->addProbe(std::move(probe));
    }


    return true;
}
