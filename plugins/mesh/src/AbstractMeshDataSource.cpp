#include "stdafx.h"

#include "mesh/AbstractMeshDataSource.h"


megamol::mesh::AbstractMeshDataSource::AbstractMeshDataSource()
    : core::Module()
    , m_mesh_access_collection({nullptr, {}})
    , m_mesh_lhs_slot("meshes", "The slot publishing the loaded data")
    , m_mesh_rhs_slot("chainMeshes", "The slot for chaining mesh data sources") 
{
    this->m_mesh_lhs_slot.SetCallback(
        CallMesh::ClassName(), "GetData", &AbstractMeshDataSource::getMeshDataCallback);
    this->m_mesh_lhs_slot.SetCallback(
        CallMesh::ClassName(), "GetMetaData", &AbstractMeshDataSource::getMeshMetaDataCallback);
    this->MakeSlotAvailable(&this->m_mesh_lhs_slot);

    this->m_mesh_rhs_slot.SetCompatibleCall<CallMeshDescription>();
    this->MakeSlotAvailable(&this->m_mesh_rhs_slot);
}

megamol::mesh::AbstractMeshDataSource::~AbstractMeshDataSource() { this->Release(); }

bool megamol::mesh::AbstractMeshDataSource::create(void) {
    return true;
}

void megamol::mesh::AbstractMeshDataSource::release() {}

void megamol::mesh::AbstractMeshDataSource::syncMeshAccessCollection(CallMesh* lhs_call, CallMesh* rhs_call) {
    if (lhs_call->getData() == nullptr) {
        // no incoming mesh -> use your own mesh access collection
        if (m_mesh_access_collection.first == nullptr) {
            m_mesh_access_collection.first = std::make_shared<MeshDataAccessCollection>();
        }
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
