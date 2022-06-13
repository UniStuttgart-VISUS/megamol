
#include "mesh/AbstractMeshDataSource.h"


megamol::mesh::AbstractMeshDataSource::AbstractMeshDataSource()
        : core::Module()
        , m_mesh_access_collection({nullptr, {}})
        , m_mesh_lhs_slot("meshes", "The slot publishing the loaded data")
        , m_mesh_rhs_slot("chainMeshes", "The slot for chaining mesh data sources") {
    this->m_mesh_lhs_slot.SetCallback(CallMesh::ClassName(), "GetData", &AbstractMeshDataSource::getMeshDataCallback);
    this->m_mesh_lhs_slot.SetCallback(
        CallMesh::ClassName(), "GetMetaData", &AbstractMeshDataSource::getMeshMetaDataCallback);
    this->MakeSlotAvailable(&this->m_mesh_lhs_slot);

    this->m_mesh_rhs_slot.SetCompatibleCall<CallMeshDescription>();
    this->MakeSlotAvailable(&this->m_mesh_rhs_slot);
}

megamol::mesh::AbstractMeshDataSource::~AbstractMeshDataSource() {
    this->Release();
}

bool megamol::mesh::AbstractMeshDataSource::create(void) {
    // default empty collection
    m_mesh_access_collection.first = std::make_shared<MeshDataAccessCollection>();
    return true;
}

bool megamol::mesh::AbstractMeshDataSource::getMeshMetaDataCallback(core::Call& caller) {
    CallMesh* lhs_mesh_call = dynamic_cast<CallMesh*>(&caller);
    CallMesh* rhs_mesh_call = m_mesh_rhs_slot.CallAs<CallMesh>();

    if (lhs_mesh_call == NULL)
        return false;
    auto lhs_meta_data = lhs_mesh_call->getMetaData();

    unsigned int frame_cnt = 1;
    auto bbox = lhs_meta_data.m_bboxs.BoundingBox();
    auto cbbox = lhs_meta_data.m_bboxs.ClipBox();

    if (rhs_mesh_call != NULL) {
        auto rhs_meta_data = rhs_mesh_call->getMetaData();
        rhs_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
        rhs_mesh_call->setMetaData(rhs_meta_data);
        if (!(*rhs_mesh_call)(1))
            return false;
        rhs_meta_data = rhs_mesh_call->getMetaData();

        frame_cnt = std::max(rhs_meta_data.m_frame_cnt, frame_cnt);

        bbox.Union(rhs_meta_data.m_bboxs.BoundingBox());
        cbbox.Union(rhs_meta_data.m_bboxs.ClipBox());
    }

    lhs_meta_data.m_frame_cnt = frame_cnt;
    lhs_meta_data.m_bboxs.SetBoundingBox(bbox);
    lhs_meta_data.m_bboxs.SetClipBox(cbbox);

    lhs_mesh_call->setMetaData(lhs_meta_data);

    return true;
}

void megamol::mesh::AbstractMeshDataSource::release() {}

void megamol::mesh::AbstractMeshDataSource::syncMeshAccessCollection(CallMesh* lhs_call, CallMesh* rhs_call) {
    if (lhs_call->getData() == nullptr) {
        // no incoming mesh -> use your own mesh access collection, i.e. share to left
        lhs_call->setData(m_mesh_access_collection.first, lhs_call->version());
    } else {
        // incoming material -> use it, copy material from last used collection if needed
        if (lhs_call->getData() != m_mesh_access_collection.first) {

            std::pair<std::shared_ptr<MeshDataAccessCollection>, std::vector<std::string>> mesh_access_collection = {
                lhs_call->getData(), {}};

            for (auto const& identifier : m_mesh_access_collection.second) {
                MeshDataAccessCollection::Mesh mesh = m_mesh_access_collection.first->accessMesh(identifier);
                mesh_access_collection.first->addMesh(identifier, mesh.attributes, mesh.indices, mesh.primitive_type);
                m_mesh_access_collection.first->deleteMesh(identifier);
            }

            m_mesh_access_collection = mesh_access_collection;
        }
    }

    if (rhs_call != nullptr) {
        rhs_call->setData(m_mesh_access_collection.first, rhs_call->version());
    }
}

void megamol::mesh::AbstractMeshDataSource::clearMeshAccessCollection() {
    for (auto& identifier : m_mesh_access_collection.second) {
        m_mesh_access_collection.first->deleteMesh(identifier);
    }
    m_mesh_access_collection.second.clear();
}
