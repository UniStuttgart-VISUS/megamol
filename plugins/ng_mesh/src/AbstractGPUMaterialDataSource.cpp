/*
 * AbstractGPUMaterialsDataSource.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#include "ng_mesh/AbstractGPUMaterialDataSource.h"
#include "ng_mesh/GPUMaterialDataCall.h"

megamol::ngmesh::AbstractGPUMaterialDataSource::AbstractGPUMaterialDataSource()
    : core::Module()
    , m_gpu_materials(std::make_shared<GPUMaterialCollecton>())
    , m_getData_slot("getData", "The slot publishing the loaded data")
    , m_mtl_callerSlot("getMaterial", "The slot for chaining material data sources") {
    this->m_getData_slot.SetCallback(
        GPUMaterialDataCall::ClassName(), "GetData", &AbstractGPUMaterialDataSource::getDataCallback);
    this->MakeSlotAvailable(&this->m_getData_slot);

    this->m_mtl_callerSlot.SetCompatibleCall<GPUMaterialDataCallDescription>();
    this->MakeSlotAvailable(&this->m_mtl_callerSlot);
}

megamol::ngmesh::AbstractGPUMaterialDataSource::~AbstractGPUMaterialDataSource() { this->Release(); }

bool megamol::ngmesh::AbstractGPUMaterialDataSource::create(void) {
    // intentionally empty ?
    return true;
}

void megamol::ngmesh::AbstractGPUMaterialDataSource::release() {
    // intentionally empty ?
}
