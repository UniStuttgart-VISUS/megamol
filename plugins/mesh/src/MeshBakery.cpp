#include "MeshBakery.h"

#include <glm/glm.hpp>

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

megamol::mesh::MeshBakery::MeshBakery()
        : AbstractMeshDataSource()
        , m_version(0)
        , m_geometry_type("GeometryType", "...")
        , m_icosphere_radius("IcosphereRadius","...")
        , m_icosphere_subdivs("IcosphereSubdivs", "")
{
    m_geometry_type << new megamol::core::param::EnumParam(2);
    m_geometry_type.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Quad");
    m_geometry_type.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "Cone");
    m_geometry_type.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "Icosphere");
    MakeSlotAvailable(&m_geometry_type);

    m_icosphere_radius << new megamol::core::param::FloatParam(1.0f,0.0f);
    MakeSlotAvailable(&m_icosphere_radius);

    m_icosphere_subdivs << new megamol::core::param::IntParam(0,0);
    MakeSlotAvailable(&m_icosphere_subdivs);
}

megamol::mesh::MeshBakery::~MeshBakery() {
    Release();
}

bool megamol::mesh::MeshBakery::create(void) {
    AbstractMeshDataSource::create();
    return true;
}

void megamol::mesh::MeshBakery::release() {}

bool megamol::mesh::MeshBakery::getMeshDataCallback(core::Call& caller) {
    CallMesh* lhs_mesh_call = dynamic_cast<CallMesh*>(&caller);
    CallMesh* rhs_mesh_call = m_mesh_rhs_slot.CallAs<CallMesh>();

    if (lhs_mesh_call == nullptr) {
        return false;
    }

    auto mesh_access_collections = std::make_shared<MeshDataAccessCollection>();

    if (rhs_mesh_call != nullptr) {
        if (!(*rhs_mesh_call)(0)) {
            return false;
        }
        if (rhs_mesh_call->hasUpdate()) {
            ++m_version;
        }
        auto mesh_access_collections = rhs_mesh_call->getData();
    }

    bool something_has_changed =
        m_geometry_type.IsDirty() || m_icosphere_radius.IsDirty() || m_icosphere_subdivs.IsDirty();

    if (m_vertex_positions.empty() || something_has_changed) {
        //TODO bad move, but necessary right now. chaining is broken with this
        clearMeshAccessCollection();

        m_geometry_type.ResetDirty();
        m_icosphere_radius.ResetDirty();
        m_icosphere_subdivs.ResetDirty();

        ++m_version;

        std::array<float, 6> bbox;

        // init default bounding box
        bbox[0] = -1.0f;
        bbox[1] = -1.0f;
        bbox[2] = -1.0f;
        bbox[3] = 1.0f;
        bbox[4] = 1.0f;
        bbox[5] = 1.0f;

        auto geometry_type = m_geometry_type.Param<core::param::EnumParam>()->Value();

        //TODO call geometry generating functions
        switch (geometry_type) {
        case 0:
            m_icosphere_radius.Param<megamol::core::param::FloatParam>()->SetGUIVisible(false);
            m_icosphere_subdivs.Param<megamol::core::param::IntParam>()->SetGUIVisible(false);
            break;
        case 1:
            m_icosphere_radius.Param<megamol::core::param::FloatParam>()->SetGUIVisible(false);
            m_icosphere_subdivs.Param<megamol::core::param::IntParam>()->SetGUIVisible(false);
            createConeGeometry();
            break;
        case 2:
            m_icosphere_radius.Param<megamol::core::param::FloatParam>()->SetGUIVisible(true);
            m_icosphere_subdivs.Param<megamol::core::param::IntParam>()->SetGUIVisible(true);
            createIcosphereGeometry(m_icosphere_radius.Param<megamol::core::param::FloatParam>()->Value(),
                static_cast<unsigned int>(m_icosphere_subdivs.Param<megamol::core::param::IntParam>()->Value()));
            for (auto& val : bbox) {
                val *= m_icosphere_radius.Param<megamol::core::param::FloatParam>()->Value();
            }
            break;
        default:
            break;
        }

        std::vector<MeshDataAccessCollection::VertexAttribute> mesh_attributes;
        MeshDataAccessCollection::IndexData mesh_indices;

        mesh_indices.byte_size = m_indices.size() * sizeof(uint32_t);
        mesh_indices.data = reinterpret_cast<uint8_t*>(m_indices.data());
        mesh_indices.type = MeshDataAccessCollection::UNSIGNED_INT;

        mesh_attributes.emplace_back(MeshDataAccessCollection::VertexAttribute{
            reinterpret_cast<uint8_t*>(m_vertex_normals.data()), m_vertex_normals.size() * sizeof(float), 3,
            MeshDataAccessCollection::FLOAT, 12, 0, MeshDataAccessCollection::NORMAL});

        mesh_attributes.emplace_back(MeshDataAccessCollection::VertexAttribute{
            reinterpret_cast<uint8_t*>(m_vertex_positions.data()), m_vertex_positions.size() * sizeof(float), 3,
            MeshDataAccessCollection::FLOAT, 12, 0, MeshDataAccessCollection::POSITION});

        mesh_attributes.emplace_back(MeshDataAccessCollection::VertexAttribute{
            reinterpret_cast<uint8_t*>(m_vertex_tangents.data()), m_vertex_tangents.size() * sizeof(float), 3,
            MeshDataAccessCollection::FLOAT, 12, 0, MeshDataAccessCollection::TANGENT});

        mesh_attributes.emplace_back(MeshDataAccessCollection::VertexAttribute{
            reinterpret_cast<uint8_t*>(m_vertex_uvs.data()), m_vertex_uvs.size() * sizeof(float), 2,
            MeshDataAccessCollection::FLOAT, 8, 0, MeshDataAccessCollection::TEXCOORD});

        auto identifier = std::string(m_geometry_type.Param<core::param::EnumParam>()->ValueString());
        m_mesh_access_collection.first->addMesh(identifier, mesh_attributes, mesh_indices);
        m_mesh_access_collection.second.push_back(identifier);

        auto meta_data = lhs_mesh_call->getMetaData();
        meta_data.m_bboxs.SetBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        meta_data.m_bboxs.SetClipBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        lhs_mesh_call->setMetaData(meta_data);
    }

    mesh_access_collections->append(*m_mesh_access_collection.first);
    lhs_mesh_call->setData(mesh_access_collections, m_version);

    return true;
}

bool megamol::mesh::MeshBakery::getMeshMetaDataCallback(core::Call& caller) {
    return AbstractMeshDataSource::getMeshMetaDataCallback(caller);
}

void megamol::mesh::MeshBakery::createTriangleGeometry() {}

void megamol::mesh::MeshBakery::createConeGeometry() {

    m_vertex_positions.clear();
    m_vertex_normals.clear();
    m_vertex_tangents.clear();
    m_vertex_uvs.clear();
    m_vertex_colors.clear();
    m_indices.clear();

    int segments = 8;
    float radius = 0.0025f;
    float height = 1.0f;
    float alpha = std::atan(radius / height);

    //TODO create cone base


    //TODO create cone side
    for (int i = 0; i < segments; ++i) {
        float x = 0.0f, y = 0.0f, z = 0.0f;

        x = radius * std::sin(static_cast<float>(i) / static_cast<float>(segments) * 2.0f * 3.14159265359f);
        y = radius * std::cos(static_cast<float>(i) / static_cast<float>(segments) * 2.0f * 3.14159265359f);

        float nx = x, ny = y;
        float nz = std::tan(alpha) * radius;
        float nl = std::sqrt(nx * nx + ny * ny + nz * nz);

        m_vertex_positions.push_back(x);
        m_vertex_positions.push_back(y);
        m_vertex_positions.push_back(z);

        m_vertex_normals.push_back(nx / nl);
        m_vertex_normals.push_back(ny / nl);
        m_vertex_normals.push_back(nz / nl);

        m_vertex_tangents.push_back(y / radius);
        m_vertex_tangents.push_back(x / radius);
        m_vertex_tangents.push_back(z / radius);

        m_vertex_uvs.push_back(0.0f);
        m_vertex_uvs.push_back(0.0f);
    }

    //TODO cone tip vertices
    for (int i = 0; i < segments; ++i) {
        float x = 0.0f, y = 0.0f, z = height;

        float nx = (m_vertex_normals[i * 3 + 0] + m_vertex_normals[((i + 1) % segments) * 3 + 0]) / 2.0f;
        float ny = (m_vertex_normals[i * 3 + 1] + m_vertex_normals[((i + 1) % segments) * 3 + 1]) / 2.0f;
        float nz = (m_vertex_normals[i * 3 + 2] + m_vertex_normals[((i + 1) % segments) * 3 + 2]) / 2.0f;

        float tx = (m_vertex_tangents[i * 3 + 0] + m_vertex_tangents[((i + 1) % segments) * 3 + 0]) / 2.0f;
        float ty = (m_vertex_tangents[i * 3 + 1] + m_vertex_tangents[((i + 1) % segments) * 3 + 1]) / 2.0f;
        float tz = (m_vertex_tangents[i * 3 + 2] + m_vertex_tangents[((i + 1) % segments) * 3 + 2]) / 2.0f;

        m_vertex_positions.push_back(x);
        m_vertex_positions.push_back(y);
        m_vertex_positions.push_back(z);

        m_vertex_normals.push_back(nx);
        m_vertex_normals.push_back(ny);
        m_vertex_normals.push_back(nz);

        m_vertex_tangents.push_back(tx);
        m_vertex_tangents.push_back(tx);
        m_vertex_tangents.push_back(tx);

        m_vertex_uvs.push_back(0.0f);
        m_vertex_uvs.push_back(0.0f);
    }

    for (uint32_t i = 0; i < segments; ++i) {
        m_indices.push_back(i);
        m_indices.push_back((i + 1) % segments);
        m_indices.push_back(i + segments);
    }
}

void megamol::mesh::MeshBakery::createIcosphereGeometry(float radius, unsigned int subdivisions) {

    m_vertex_positions.clear();
    m_vertex_normals.clear();
    m_vertex_tangents.clear();
    m_vertex_uvs.clear();
    m_vertex_colors.clear();
    m_indices.clear();

    // Create intial icosahedron
    float x = 0.525731112119133606f * radius;
    float z = 0.850650808352039932f * radius;
    float nx = 0.525731112119133606f;
    float nz = 0.850650808352039932f;

    m_vertex_positions = {-x, 0.0f, z, x, 0.0f, z, -x, 0.0f, -z, x, 0.0f, -z, 0.0f, z, x, 0.0f, z, -x, 0.0f, -z, x,
        0.0f, -z, -x, z, x, 0.0f, -z, x, 0.0f, z, -x, 0.0f, -z, -x, 0.0f};
    m_vertex_normals = {-nx, 0.0f, nz, nx, 0.0f, nz, -nx, 0.0f, -nz, nx, 0.0f, -nz, 0.0f, nz, nx, 0.0f, nz, -nx, 0.0f,
        -nz, nx, 0.0f, -nz, -nx, nz, nx, 0.0f, -nz, nx, 0.0f, nz, -nx, 0.0f, -nz, -nx, 0.0f};
    m_indices = {0, 4, 1, 0, 9, 4, 9, 5, 4, 4, 5, 8, 4, 8, 1, 8, 10, 1, 8, 3, 10, 5, 3, 8, 5, 2, 3, 2, 7, 3, 7, 10, 3,
        7, 6, 10, 7, 11, 6, 11, 0, 6, 0, 1, 6, 6, 1, 10, 9, 0, 11, 9, 11, 2, 9, 2, 5, 7, 2, 11};

    // Subdivide icosahedron
    for (unsigned int subdivs = 0; subdivs < subdivisions; subdivs++) {
        std::vector<uint32_t> refined_indices;
        refined_indices.reserve(m_indices.size() * 3);

        for (int i = 0; i < m_indices.size(); i = i + 3) {
            unsigned int idx1 = m_indices[i];
            unsigned int idx2 = m_indices[i + 1];
            unsigned int idx3 = m_indices[i + 2];

            glm::vec3 newVtx1((m_vertex_positions[idx1 * 3 + 0] + m_vertex_positions[idx2 * 3 + 0]),
                (m_vertex_positions[idx1 * 3 + 1] + m_vertex_positions[idx2 * 3 + 1]),
                (m_vertex_positions[idx1 * 3 + 2] + m_vertex_positions[idx2 * 3 + 2]));
            newVtx1 = glm::normalize(newVtx1);
            newVtx1 *= radius;

            glm::vec3 newVtx2((m_vertex_positions[idx2 * 3 + 0] + m_vertex_positions[idx3 * 3 + 0]),
                (m_vertex_positions[idx2 * 3 + 1] + m_vertex_positions[idx3 * 3 + 1]),
                (m_vertex_positions[idx2 * 3 + 2] + m_vertex_positions[idx3 * 3 + 2]));
            newVtx2 = glm::normalize(newVtx2);
            newVtx2 *= radius;

            glm::vec3 newVtx3((m_vertex_positions[idx3 * 3 + 0] + m_vertex_positions[idx1 * 3 + 0]),
                (m_vertex_positions[idx3 * 3 + 1] + m_vertex_positions[idx1 * 3 + 1]),
                (m_vertex_positions[idx3 * 3 + 2] + m_vertex_positions[idx1 * 3 + 2]));
            newVtx3 = glm::normalize(newVtx3);
            newVtx3 *= radius;

            glm::vec3 newVtxNrml1((m_vertex_normals[idx1 * 3 + 0] + m_vertex_normals[idx2 * 3 + 0]),
                (m_vertex_normals[idx1 * 3 + 1] + m_vertex_normals[idx2 * 3 + 1]),
                (m_vertex_normals[idx1 * 3 + 2] + m_vertex_normals[idx2 * 3 + 2]));
            newVtxNrml1 = glm::normalize(newVtx1);

            glm::vec3 newVtxNrml2((m_vertex_normals[idx2 * 3 + 0] + m_vertex_normals[idx3 * 3 + 0]),
                (m_vertex_normals[idx2 * 3 + 1] + m_vertex_normals[idx3 * 3 + 1]),
                (m_vertex_normals[idx2 * 3 + 2] + m_vertex_normals[idx3 * 3 + 2]));
            newVtxNrml2 = glm::normalize(newVtx2);

            glm::vec3 newVtxNrml3((m_vertex_normals[idx3 * 3 + 0] + m_vertex_normals[idx1 * 3 + 0]),
                (m_vertex_normals[idx3 * 3 + 1] + m_vertex_normals[idx1 * 3 + 1]),
                (m_vertex_normals[idx3 * 3 + 2] + m_vertex_normals[idx1 * 3 + 2]));
            newVtxNrml3 = glm::normalize(newVtx3);

            unsigned int newIdx1 = static_cast<unsigned int>(m_vertex_positions.size() / 3);
            m_vertex_positions.insert(m_vertex_positions.end(), {newVtx1.x, newVtx1.y, newVtx1.z});
            m_vertex_normals.insert(m_vertex_normals.end(), {newVtxNrml1.x, newVtxNrml1.y, newVtxNrml1.z});

            unsigned int newIdx2 = newIdx1 + 1;
            m_vertex_positions.insert(m_vertex_positions.end(), {newVtx2.x, newVtx2.y, newVtx2.z});
            m_vertex_normals.insert(m_vertex_normals.end(), {newVtxNrml2.x, newVtxNrml2.y, newVtxNrml2.z});

            unsigned int newIdx3 = newIdx2 + 1;
            m_vertex_positions.insert(m_vertex_positions.end(), {newVtx3.x, newVtx3.y, newVtx3.z});
            m_vertex_normals.insert(m_vertex_normals.end(), {newVtxNrml3.x, newVtxNrml3.y, newVtxNrml3.z});

            refined_indices.insert(refined_indices.end(),
                {idx1, newIdx1, newIdx3, newIdx1, idx2, newIdx2, newIdx3, newIdx1, newIdx2, newIdx3, newIdx2, idx3});
        }

        m_indices.clear();
        m_indices = refined_indices;
    }

    // fill unused attributes with zeros for now
    m_vertex_tangents.resize(m_vertex_normals.size());
    m_vertex_uvs.resize((m_vertex_normals.size() / 3) * 2);
}
