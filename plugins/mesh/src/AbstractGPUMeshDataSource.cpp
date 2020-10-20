/*
 * AbstractGPUMeshDataSource.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */
#include "stdafx.h"

#include "mesh/AbstractGPUMeshDataSource.h"


megamol::mesh::AbstractGPUMeshDataSource::AbstractGPUMeshDataSource()
    : core::Module()
    , m_mesh_collection({nullptr, {}})
    , m_mesh_lhs_slot("gpuMeshes", "The slot publishing the loaded data")
    , m_mesh_rhs_slot("chainGpuMeshes", "The slot for chaining material data sources") 
{
    this->m_mesh_lhs_slot.SetCallback(
        CallGPUMeshData::ClassName(), "GetData", &AbstractGPUMeshDataSource::getDataCallback);
    this->m_mesh_lhs_slot.SetCallback(
        CallGPUMeshData::ClassName(), "GetMetaData", &AbstractGPUMeshDataSource::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_mesh_lhs_slot);

    this->m_mesh_rhs_slot.SetCompatibleCall<CallGPUMeshDataDescription>();
    this->MakeSlotAvailable(&this->m_mesh_rhs_slot);
}

megamol::mesh::AbstractGPUMeshDataSource::~AbstractGPUMeshDataSource() { this->Release(); }

bool megamol::mesh::AbstractGPUMeshDataSource::create(void) {
    return true;
}

void megamol::mesh::AbstractGPUMeshDataSource::release() {}

void megamol::mesh::AbstractGPUMeshDataSource::syncMeshCollection(CallGPUMeshData* lhs_call) {
    if (lhs_call->getData() == nullptr) {
        // no incoming material -> use your own material storage
        if (m_mesh_collection.first == nullptr) {
            m_mesh_collection.first = std::make_shared<GPUMeshCollection>();
        }
    } else {
        // incoming material -> use it, copy material from last used collection if needed
        if (lhs_call->getData() != m_mesh_collection.first) {
            std::pair<std::shared_ptr<GPUMeshCollection>, std::vector<std::string>> mesh_collection = {
                lhs_call->getData(), {}};
            for (auto const& identifier : m_mesh_collection.second) {
                //mtl_collection.first->addMesh(m_mesh_collection.first->getMeshes()[idx]);
                auto const& submesh = m_mesh_collection.first->getSubMesh(identifier);
                mesh_collection.first->addMesh(identifier, submesh.mesh->mesh, submesh);
                m_mesh_collection.first->deleteSubMesh(identifier);
            }
            m_mesh_collection = mesh_collection;
        }
    }
}
