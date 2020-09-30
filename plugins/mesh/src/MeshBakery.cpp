#include "MeshBakery.h"

#include "mmcore/param/EnumParam.h"

megamol::mesh::MeshBakery::MeshBakery() 
    : core::Module()
    , m_version(0)
    , m_mesh_access_collection(nullptr)
    , m_geometry_type("GeometryType", "...")
    , m_mesh_lhs_slot("deployMeshAccess", "...")
    , m_mesh_rhs_slot("chainMeshAccess", "...") 
{
    this->m_geometry_type << new megamol::core::param::EnumParam(0);
    this->m_geometry_type.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Quad");
    this->m_geometry_type.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "Cone");
    this->MakeSlotAvailable(&this->m_geometry_type);

    this->m_mesh_lhs_slot.SetCallback(CallMesh::ClassName(), "GetData", &MeshBakery::getDataCallback);
    this->m_mesh_lhs_slot.SetCallback(CallMesh::ClassName(), "GetMetaData", &MeshBakery::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_mesh_lhs_slot);

    this->m_mesh_rhs_slot.SetCompatibleCall<CallMeshDescription>();
    this->MakeSlotAvailable(&this->m_mesh_rhs_slot);
}

megamol::mesh::MeshBakery::~MeshBakery() { this->Release(); }

bool megamol::mesh::MeshBakery::create(void) {

    this->m_mesh_access_collection = std::make_shared<MeshDataAccessCollection>();

    return true;
}

void megamol::mesh::MeshBakery::release() {}

bool megamol::mesh::MeshBakery::getDataCallback(core::Call& caller) {

    CallMesh* lhs_mesh_call = dynamic_cast<CallMesh*>(&caller);
    CallMesh* rhs_mesh_call = m_mesh_rhs_slot.CallAs<CallMesh>();

    if (lhs_mesh_call == NULL) return false;

    std::shared_ptr<MeshDataAccessCollection> mesh_access_collection(nullptr);

    if (lhs_mesh_call->getData() == nullptr) {
        // no incoming mesh -> use your own mesh storage
        mesh_access_collection = this->m_mesh_access_collection;
    } else {
        // incoming mesh -> use it (delete local?)
        mesh_access_collection = lhs_mesh_call->getData();
    }

    //TODO check chain

    if (m_geometry_type.IsDirty()) {
        m_geometry_type.ResetDirty();

        ++m_version;

        //TODO call geometry generating functions
        createConeGeometry();

        std::array<float, 6> bbox;

        bbox[0] = std::numeric_limits<float>::max();
        bbox[1] = std::numeric_limits<float>::max();
        bbox[2] = std::numeric_limits<float>::max();
        bbox[3] = std::numeric_limits<float>::min();
        bbox[4] = std::numeric_limits<float>::min();
        bbox[5] = std::numeric_limits<float>::min();

        bbox[0] = -1.0f;
        bbox[1] = -1.0f;
        bbox[2] = -1.0f;
        bbox[3] = 1.0f;
        bbox[4] = 1.0f;
        bbox[5] = 1.0f;

        std::vector<MeshDataAccessCollection::VertexAttribute> mesh_attributes;
        MeshDataAccessCollection::IndexData mesh_indices;

        mesh_indices.byte_size = m_indices.size() * sizeof(uint32_t);
        mesh_indices.data = reinterpret_cast<uint8_t*>(m_indices.data());
        mesh_indices.type = MeshDataAccessCollection::UNSIGNED_INT;

        mesh_attributes.emplace_back(MeshDataAccessCollection::VertexAttribute{
            reinterpret_cast<uint8_t*>(m_vertex_normals.data()), m_vertex_normals.size() * sizeof(float), 3,
            MeshDataAccessCollection::FLOAT, 0, 0, MeshDataAccessCollection::NORMAL});

        mesh_attributes.emplace_back(MeshDataAccessCollection::VertexAttribute{
            reinterpret_cast<uint8_t*>(m_vertex_positions.data()), m_vertex_positions.size() * sizeof(float), 3,
            MeshDataAccessCollection::FLOAT, 0, 0, MeshDataAccessCollection::POSITION});

        mesh_attributes.emplace_back(MeshDataAccessCollection::VertexAttribute{
            reinterpret_cast<uint8_t*>(m_vertex_tangents.data()), m_vertex_tangents.size() * sizeof(float), 3,
            MeshDataAccessCollection::FLOAT, 0, 0, MeshDataAccessCollection::TANGENT});

        mesh_attributes.emplace_back(MeshDataAccessCollection::VertexAttribute{
            reinterpret_cast<uint8_t*>(m_vertex_uvs.data()), m_vertex_uvs.size() * sizeof(float), 2,
            MeshDataAccessCollection::FLOAT, 0, 0, MeshDataAccessCollection::TEXCOORD});


        mesh_access_collection->addMesh(mesh_attributes, mesh_indices);


        auto meta_data = lhs_mesh_call->getMetaData();
        meta_data.m_bboxs.SetBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        meta_data.m_bboxs.SetClipBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        lhs_mesh_call->setMetaData(meta_data);
    }

    lhs_mesh_call->setData(mesh_access_collection, m_version);

    return true; 
}

bool megamol::mesh::MeshBakery::getMetaDataCallback(core::Call& caller) {
    CallMesh* lhs_mesh_call = dynamic_cast<CallMesh*>(&caller);
    CallMesh* rhs_mesh_call = m_mesh_rhs_slot.CallAs<CallMesh>();

    if (lhs_mesh_call == NULL) return false;
    auto lhs_meta_data = lhs_mesh_call->getMetaData();

    bool something_has_changed = false; // something has changed in the neath...
    unsigned int frame_cnt = std::numeric_limits<unsigned int>::max();
    auto bbox = lhs_meta_data.m_bboxs.BoundingBox();
    auto cbbox = lhs_meta_data.m_bboxs.ClipBox();

    if (rhs_mesh_call != NULL) {
        auto rhs_meta_data = rhs_mesh_call->getMetaData();
        rhs_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
        rhs_mesh_call->setMetaData(rhs_meta_data);
        if (!(*rhs_mesh_call)(1)) return false;
        rhs_meta_data = rhs_mesh_call->getMetaData();

        something_has_changed = rhs_mesh_call->hasUpdate();

        frame_cnt = std::min(rhs_meta_data.m_frame_cnt, frame_cnt);

        bbox.Union(rhs_meta_data.m_bboxs.BoundingBox());
        cbbox.Union(rhs_meta_data.m_bboxs.ClipBox());
    }

    lhs_meta_data.m_frame_cnt = frame_cnt;
    lhs_meta_data.m_bboxs.SetBoundingBox(bbox);
    lhs_meta_data.m_bboxs.SetClipBox(cbbox);

    lhs_mesh_call->setMetaData(lhs_meta_data);

    return true; 
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
    float radius = 0.05f;
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

        m_vertex_normals.push_back(nx/nl);
        m_vertex_normals.push_back(ny/nl);
        m_vertex_normals.push_back(nz/nl);

        m_vertex_tangents.push_back(y/radius);
        m_vertex_tangents.push_back(x/radius);
        m_vertex_tangents.push_back(z/radius);

        m_vertex_uvs.push_back(0.0f);
        m_vertex_uvs.push_back(0.0f);
    }

    //TODO cone tip vertices
    for (int i = 0; i < segments; ++i) {
        float x = 0.0f, y = 0.0f, z = height;

        float nx = (m_vertex_normals[i * 3 + 0] + m_vertex_normals[((i+1) % segments) * 3 + 0]) / 2.0f;
        float ny = (m_vertex_normals[i * 3 + 1] + m_vertex_normals[((i+1) % segments) * 3 + 1]) / 2.0f;
        float nz = (m_vertex_normals[i * 3 + 2] + m_vertex_normals[((i+1) % segments) * 3 + 2]) / 2.0f;

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
        m_indices.push_back( (i+1) % segments );
        m_indices.push_back(i + segments);
    }
}
