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
    , m_mesh_rhs_slot("getMesh", "The slot for chaining material data sources") {
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

bool megamol::mesh::AbstractGPUMeshDataSource::getMetaDataCallback(core::Call& caller) {
    CallGPUMeshData* mc = dynamic_cast<CallGPUMeshData*>(&caller);
    if (mc == NULL) return false;

    megamol::core::BoundingBoxes bboxs;
    bboxs.SetObjectSpaceBBox(
        this->m_bbox[0], this->m_bbox[1], this->m_bbox[2], this->m_bbox[3], this->m_bbox[4], this->m_bbox[5]);
    bboxs.SetObjectSpaceClipBox(
        this->m_bbox[0], this->m_bbox[1], this->m_bbox[2], this->m_bbox[3], this->m_bbox[4], this->m_bbox[5]);

    mc->setMetaData({0, 1, 0, bboxs});

    return true;
}

void megamol::mesh::AbstractGPUMeshDataSource::release() {}
