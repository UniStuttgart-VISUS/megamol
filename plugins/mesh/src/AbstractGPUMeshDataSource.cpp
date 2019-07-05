/*
 * AbstractGPUMeshDataSource.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */
#include "stdafx.h"

#include "mesh/AbstractGPUMeshDataSource.h"
#include "mesh/CallGPUMeshData.h"


megamol::mesh::AbstractGPUMeshDataSource::AbstractGPUMeshDataSource()
    : core::Module()
    , m_getData_slot("getData", "The slot publishing the loaded data")
    , m_mesh_callerSlot("getMesh", "The slot for chaining material data sources") {
    this->m_getData_slot.SetCallback(
        CallGPUMeshData::ClassName(), "GetData", &AbstractGPUMeshDataSource::getDataCallback);
    this->m_getData_slot.SetCallback(
        CallGPUMeshData::ClassName(), "GetExtent", &AbstractGPUMeshDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->m_getData_slot);

    this->m_mesh_callerSlot.SetCompatibleCall<CallGPUMeshDataDescription>();
    this->MakeSlotAvailable(&this->m_mesh_callerSlot);
}

megamol::mesh::AbstractGPUMeshDataSource::~AbstractGPUMeshDataSource() { this->Release(); }

bool megamol::mesh::AbstractGPUMeshDataSource::create(void) {
    m_gpu_meshes = std::make_shared<GPUMeshCollection>();

    return true;
}

bool megamol::mesh::AbstractGPUMeshDataSource::getExtentCallback(core::Call& caller) {
    CallGPUMeshData* mc = dynamic_cast<CallGPUMeshData*>(&caller);
    if (mc == NULL) return false;

    mc->SetExtent(
        1, this->m_bbox[0], this->m_bbox[1], this->m_bbox[2], this->m_bbox[3], this->m_bbox[4], this->m_bbox[5]);

    return true;
}

void megamol::mesh::AbstractGPUMeshDataSource::release() {}
