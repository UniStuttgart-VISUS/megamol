/*
 * AbstractGPUMeshDataSource.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */
#include "stdafx.h"

#include "mesh/AbstractGPUMeshDataSource.h"
#include "mesh/MeshCalls.h"


megamol::mesh::AbstractGPUMeshDataSource::AbstractGPUMeshDataSource()
    : core::Module()
    , m_mesh_lhs_slot("getData", "The slot publishing the loaded data")
    , m_mesh_lhs_cached_hash(0)
    , m_mesh_rhs_slot("getMesh", "The slot for chaining material data sources") 
    , m_mesh_rhs_cached_hash(0) 
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
    m_gpu_meshes = std::make_shared<GPUMeshCollection>();

    return true;
}

void megamol::mesh::AbstractGPUMeshDataSource::release() {}
