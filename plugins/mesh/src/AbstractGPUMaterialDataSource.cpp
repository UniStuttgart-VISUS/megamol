/*
 * AbstractGPUMaterialsDataSource.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#include "mesh/AbstractGPUMaterialDataSource.h"

megamol::mesh::AbstractGPUMaterialDataSource::AbstractGPUMaterialDataSource()
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

megamol::mesh::AbstractGPUMaterialDataSource::~AbstractGPUMaterialDataSource() {
    this->Release();
}

bool megamol::mesh::AbstractGPUMaterialDataSource::create(void) {
    // default empty collection
    m_material_collection.first = std::make_shared<GPUMaterialCollection>();
    return true;
}

void megamol::mesh::AbstractGPUMaterialDataSource::release() {
    // intentionally empty ?
}

void megamol::mesh::AbstractGPUMaterialDataSource::syncMaterialCollection(
    megamol::mesh::CallGPUMaterialData* lhs_call, CallGPUMaterialData* rhs_call) {
    if (lhs_call->getData() == nullptr) {
        // no incoming material -> use your own material storage, i.e share to left
        lhs_call->setData(m_material_collection.first, lhs_call->version());
    } else {
        // incoming material -> use it, copy material from last used collection if needed
        if (lhs_call->getData() != m_material_collection.first) {
            std::pair<std::shared_ptr<GPUMaterialCollection>, std::vector<std::string>> mtl_collection = {
                lhs_call->getData(), {}};
            for (auto& identifier : m_material_collection.second) {
                mtl_collection.first->addMaterial(identifier, m_material_collection.first->getMaterial(identifier));
                mtl_collection.second.push_back(identifier);
                m_material_collection.first->deleteMaterial(identifier);
            }
            m_material_collection = mtl_collection;
        }
    }

    if (rhs_call != nullptr) {
        rhs_call->setData(m_material_collection.first, rhs_call->version());
    }
}

void megamol::mesh::AbstractGPUMaterialDataSource::clearMaterialCollection() {
    for (auto& identifier : m_material_collection.second) {
        m_material_collection.first->deleteMaterial(identifier);
    }
    m_material_collection.second.clear();
}
