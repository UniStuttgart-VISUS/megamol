#include "stdafx.h"

#include "mesh/AbstractMeshDataSource.h"


megamol::mesh::AbstractMeshDataSource::AbstractMeshDataSource()
    : core::Module()
    , m_mesh_collection({nullptr, {}})
    , m_mesh_lhs_slot("getData", "The slot publishing the loaded data")
    , m_mesh_rhs_slot("getMesh", "The slot for chaining material data sources") 
{
    this->m_mesh_lhs_slot.SetCallback(
        CallGPUMeshData::ClassName(), "GetData", &AbstractMeshDataSource::getDataCallback);
    this->m_mesh_lhs_slot.SetCallback(
        CallGPUMeshData::ClassName(), "GetMetaData", &AbstractMeshDataSource::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_mesh_lhs_slot);

    this->m_mesh_rhs_slot.SetCompatibleCall<CallMeshDescription>();
    this->MakeSlotAvailable(&this->m_mesh_rhs_slot);
}

megamol::mesh::AbstractMeshDataSource::~AbstractMeshDataSource() { this->Release(); }

bool megamol::mesh::AbstractMeshDataSource::create(void) {
    return true;
}

void megamol::mesh::AbstractMeshDataSource::release() {}

void megamol::mesh::AbstractMeshDataSource::syncMeshAccessCollection(CallMesh* lhs_call) {
    //if (lhs_call->getData() == nullptr) {
    //    // no incoming material -> use your own material storage
    //    if (m_mesh_collection.first == nullptr) {
    //        m_mesh_collection.first = std::make_shared<GPUMeshCollection>();
    //    }
    //} else {
    //    // incoming material -> use it, copy material from last used collection if needed
    //    if (lhs_call->getData() != m_mesh_collection.first) {
    //        std::pair<std::shared_ptr<GPUMeshCollection>, std::vector<std::string>> mesh_collection = {
    //            lhs_call->getData(), {}};
    //        for (auto const& identifier : m_mesh_collection.second) {
    //            //mtl_collection.first->addMesh(m_mesh_collection.first->getMeshes()[idx]);
    //            auto const& submesh = m_mesh_collection.first->getSubMesh(identifier);
    //            mesh_collection.first->addMesh(identifier, submesh.mesh->mesh, submesh);
    //            m_mesh_collection.first->deleteSubMesh(identifier);
    //        }
    //        m_mesh_collection = mesh_collection;
    //    }
    //}
}
