/*
 * AbstractGPUMaterialsDataSource.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#include "mesh/AbstractGPUMaterialDataSource.h"
#include "mesh/MeshCalls.h"

megamol::mesh::AbstractGPUMaterialDataSource::AbstractGPUMaterialDataSource()
    : core::Module()
    , m_gpu_materials(std::make_shared<GPUMaterialCollecton>())
    , m_getData_slot("getData", "The slot publishing the loaded data")
    , m_mtl_callerSlot("getMaterial", "The slot for chaining material data sources") {
    this->m_getData_slot.SetCallback(
        CallGPUMaterialData::ClassName(), "GetData", &AbstractGPUMaterialDataSource::getDataCallback);
    this->m_getData_slot.SetCallback(
        CallGPUMaterialData::ClassName(), "GetMetaData", &AbstractGPUMaterialDataSource::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_getData_slot);

    this->m_mtl_callerSlot.SetCompatibleCall<CallGPUMaterialDataDescription>();
    this->MakeSlotAvailable(&this->m_mtl_callerSlot);
}

megamol::mesh::AbstractGPUMaterialDataSource::~AbstractGPUMaterialDataSource() { this->Release(); }

bool megamol::mesh::AbstractGPUMaterialDataSource::create(void) {
    // intentionally empty ?
    return true;
}

void megamol::mesh::AbstractGPUMaterialDataSource::release() {
    // intentionally empty ?
}
