#include "PlaceProbes.h"
#include "ProbeCalls.h"

megamol::probe::PlaceProbes::PlaceProbes()
    : Module(), m_mesh_slot("getMesh", ""), m_probe_slot("deployProbes", ""), m_centerline_slot("getCenterLine", "") {

    this->m_probe_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(0), &PlaceProbes::getData);
    this->m_probe_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(1), &PlaceProbes::getMetaData);
    this->MakeSlotAvailable(&this->m_probe_slot);

    this->m_mesh_slot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->m_mesh_slot);

    this->m_centerline_slot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->m_centerline_slot);

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

    auto mesh_meta_data = cm->getMetaData();
    auto probe_meta_data = pc->getMetaData();
    auto centerline_meta_data = pc->getMetaData();

    if (mesh_meta_data.m_data_hash == m_mesh_cached_hash && probe_meta_data.m_data_hash == m_probe_cached_hash &&
        centerline_meta_data.m_data_hash == m_centerline_cached_hash)
        return true;

    if (!(*cm)(0)) return false;
    if (!(*ccl)(0)) return false;

    mesh_meta_data = cm->getMetaData();
    probe_meta_data = pc->getMetaData();

    probe_meta_data.m_bboxs = mesh_meta_data.m_bboxs;

    m_mesh = cm->getData();
    m_centerline = ccl->getData();

    // here something really happens
    this->placeProbes();

    pc->setData(this->m_probes);

    m_mesh_cached_hash = mesh_meta_data.m_data_hash;
    m_centerline_cached_hash = centerline_meta_data.m_data_hash;
    m_probe_cached_hash++;
    probe_meta_data.m_data_hash = m_probe_cached_hash;
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
    auto centerline_meta_data = pc->getMetaData();

    mesh_meta_data.m_frame_ID = probe_meta_data.m_frame_ID;
    centerline_meta_data.m_frame_ID = probe_meta_data.m_frame_ID;
    cm->setMetaData(mesh_meta_data);
    ccl->setMetaData(centerline_meta_data);

    if (!(*cm)(1)) return false;
    if (!(*ccl)(1)) return false;

    mesh_meta_data = cm->getMetaData();
    probe_meta_data.m_frame_cnt = mesh_meta_data.m_frame_cnt;
    probe_meta_data.m_bboxs = mesh_meta_data.m_bboxs; // normally not available here

    pc->setMetaData(probe_meta_data);

    return true;
}



bool megamol::probe::PlaceProbes::placeProbes() {

    m_probes = std::make_shared<ProbeCollection>();


    assert(m_mesh->accessMesh().size() == 1);
    assert(m_centerline->accessMesh().size() == 1);

    std::shared_ptr<mesh::MeshDataAccessCollection::VertexAttribute> vertices;
    std::shared_ptr<mesh::MeshDataAccessCollection::VertexAttribute> centerline;

    for (auto& attribute : m_mesh->accessMesh()[0].attributes) {
        if (attribute.semantic == mesh::MeshDataAccessCollection::POSITION) {
            vertices = std::make_shared<mesh::MeshDataAccessCollection::VertexAttribute>(attribute);
        }
    }

    for (auto& attribute : m_centerline->accessMesh()[0].attributes) {
        if (attribute.semantic == mesh::MeshDataAccessCollection::POSITION) {
            centerline = std::make_shared<mesh::MeshDataAccessCollection::VertexAttribute>(attribute);
        }
    }

    //uint32_t probe_count = vertices->byte_size / vertices->stride;
    uint32_t probe_count = 100; 
    uint32_t centerline_vert_count =
        centerline->byte_size / centerline->stride;

    auto vertex_accessor = reinterpret_cast<float*>(vertices->data);
    auto centerline_accessor = reinterpret_cast<float*>(centerline->data);

    auto vertex_step = vertices->stride / sizeof(vertices->component_type);
    auto centerline_step = centerline->stride / sizeof(centerline->component_type);

    for (uint32_t i = 0; i < probe_count; i++) {
        FloatProbe probe;

        std::vector<float> distances(centerline_vert_count);
        for (uint32_t j = 0; j < centerline_vert_count; j++) {
            distances[j] =
                std::sqrt(vertex_accessor[vertex_step * i + 0] * centerline_accessor[centerline_step * j + 0] +
                          vertex_accessor[vertex_step * i + 1] * centerline_accessor[centerline_step * j + 1] +
                          vertex_accessor[vertex_step * i + 2] * centerline_accessor[centerline_step * j + 2]);
        }

        auto min_iter = std::min_element(distances.begin(), distances.end());
        auto min_index = std::distance(distances.begin(), min_iter);
        distances[min_index] = std::numeric_limits<float>::max();

        auto second_min_iter = std::min_element(distances.begin(), distances.end());
        auto second_min_index = std::distance(distances.begin(), second_min_iter);

        // calc normal in plane between vert, min and second_min
        std::array<float, 3> along_centerline;
        along_centerline[0] = centerline_accessor[3 * min_index + 0] - centerline_accessor[3 * second_min_index + 0];
        along_centerline[1] = centerline_accessor[3 * min_index + 1] - centerline_accessor[3 * second_min_index + 1];
        along_centerline[2] = centerline_accessor[3 * min_index + 2] - centerline_accessor[3 * second_min_index + 2];

        std::array<float, 3> min_to_vert;
        min_to_vert[0] = vertex_accessor[3 * i + 0] - centerline_accessor[3 * min_index + 0];
        min_to_vert[1] = vertex_accessor[3 * i + 1] - centerline_accessor[3 * min_index + 1];
        min_to_vert[2] = vertex_accessor[3 * i + 2] - centerline_accessor[3 * min_index + 2];

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

        probe.m_position = {vertex_accessor[3 * i + 0], vertex_accessor[3 * i + 1], vertex_accessor[3 * i + 2]};
        probe.m_direction = normal;
        probe.m_begin = -0.2;
        probe.m_end = final_dist;

        this->m_probes->addProbe(std::move(probe));
    }


    return true;
}
