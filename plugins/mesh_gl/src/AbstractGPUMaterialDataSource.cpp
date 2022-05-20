/*
 * AbstractGPUMaterialsDataSource.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#include "mesh_gl/AbstractGPUMaterialDataSource.h"

megamol::mesh_gl::AbstractGPUMaterialDataSource::AbstractGPUMaterialDataSource()
        : core::Module()
        , m_material_collection({nullptr, {}})
        , m_getData_slot("gpuMaterials", "The slot publishing the loaded data")
        , m_mtl_callerSlot("chainGpuMaterials", "The slot for chaining material data sources") {
    this->m_getData_slot.SetCallback(
        CallGPUMaterialData::ClassName(), "GetData", &AbstractGPUMaterialDataSource::getDataCallback);
    this->m_getData_slot.SetCallback(
        CallGPUMaterialData::ClassName(), "GetMetaData", &AbstractGPUMaterialDataSource::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_getData_slot);

    this->m_mtl_callerSlot.SetCompatibleCall<CallGPUMaterialDataDescription>();
    this->MakeSlotAvailable(&this->m_mtl_callerSlot);
}

megamol::mesh_gl::AbstractGPUMaterialDataSource::~AbstractGPUMaterialDataSource() {
    this->Release();
}

bool megamol::mesh_gl::AbstractGPUMaterialDataSource::create(void) {
    // default empty collection
    m_material_collection.first = std::make_shared<GPUMaterialCollection>();
    return true;
}

void megamol::mesh_gl::AbstractGPUMaterialDataSource::release() {
    // intentionally empty ?
}


void megamol::mesh_gl::AbstractGPUMaterialDataSource::clearMaterialCollection() {
    m_material_collection.first->clear();
    m_material_collection.second.clear();
}
