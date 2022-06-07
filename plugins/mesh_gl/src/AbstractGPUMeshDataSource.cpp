/*
 * AbstractGPUMeshDataSource.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */
#include "mesh_gl/AbstractGPUMeshDataSource.h"
#include "mesh_gl/MeshCalls_gl.h"


megamol::mesh_gl::AbstractGPUMeshDataSource::AbstractGPUMeshDataSource()
        : core::Module()
        , m_mesh_collection({nullptr, {}})
        , m_mesh_lhs_slot("gpuMeshes", "The slot publishing the loaded data")
        , m_mesh_rhs_slot("chainGpuMeshes", "The slot for chaining material data sources") {
    this->m_mesh_lhs_slot.SetCallback(
        CallGPUMeshData::ClassName(), "GetData", &AbstractGPUMeshDataSource::getDataCallback);
    this->m_mesh_lhs_slot.SetCallback(
        CallGPUMeshData::ClassName(), "GetMetaData", &AbstractGPUMeshDataSource::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_mesh_lhs_slot);

    this->m_mesh_rhs_slot.SetCompatibleCall<CallGPUMeshDataDescription>();
    this->MakeSlotAvailable(&this->m_mesh_rhs_slot);
}

megamol::mesh_gl::AbstractGPUMeshDataSource::~AbstractGPUMeshDataSource() {
    this->Release();
}

bool megamol::mesh_gl::AbstractGPUMeshDataSource::create(void) {
    // default empty collection
    m_mesh_collection.first = std::make_shared<GPUMeshCollection>();
    return true;
}

void megamol::mesh_gl::AbstractGPUMeshDataSource::release() {}

void megamol::mesh_gl::AbstractGPUMeshDataSource::clearMeshCollection() {
    for (auto& identifier : m_mesh_collection.second) {
        m_mesh_collection.first->deleteSubMesh(identifier);
    }
    m_mesh_collection.second.clear();
}
