/*
 * AbstractGPURenderTaskDataSource.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#include "mesh/AbstractGPURenderTaskDataSource.h"
#include "mesh/CallGPUMaterialData.h"
#include "mesh/CallGPUMeshData.h"
#include "mesh/CallGPURenderTaskData.h"

megamol::mesh::AbstractGPURenderTaskDataSource::AbstractGPURenderTaskDataSource()
    : core::Module()
    , m_getData_slot("getData", "The slot publishing the loaded data")
    , m_renderTask_callerSlot("getRenderTasks", "The slot for chaining render task data sources.")
    , m_material_callerSlot("getMaterialData", "Connects to a material data source")
    , m_mesh_callerSlot("getMeshData", "Connects to a mesh data source") {
    this->m_getData_slot.SetCallback(
        CallGPURenderTaskData::ClassName(), "GetData", &AbstractGPURenderTaskDataSource::getDataCallback);
    this->m_getData_slot.SetCallback(
        CallGPURenderTaskData::ClassName(), "GetExtent", &AbstractGPURenderTaskDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->m_getData_slot);

    this->m_renderTask_callerSlot.SetCompatibleCall<GPURenderTasksDataCallDescription>();
    this->MakeSlotAvailable(&this->m_renderTask_callerSlot);

    this->m_material_callerSlot.SetCompatibleCall<CallGPUMaterialDataDescription>();
    this->MakeSlotAvailable(&this->m_material_callerSlot);

    this->m_mesh_callerSlot.SetCompatibleCall<CallGPUMeshDataDescription>();
    this->MakeSlotAvailable(&this->m_mesh_callerSlot);
}

megamol::mesh::AbstractGPURenderTaskDataSource::~AbstractGPURenderTaskDataSource() { this->Release(); }

bool megamol::mesh::AbstractGPURenderTaskDataSource::create(void) {
    // intentionally empty ?

    m_gpu_render_tasks = std::make_shared<GPURenderTaskCollection>();

    return true;
}

bool megamol::mesh::AbstractGPURenderTaskDataSource::getExtentCallback(core::Call& caller) {
    CallGPURenderTaskData* lhs_rtc = dynamic_cast<CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL) return false;

    unsigned int frame_cnt;
    megamol::core::BoundingBoxes bbox;

    CallGPUMeshData* mc = this->m_mesh_callerSlot.CallAs<CallGPUMeshData>();
    if (mc == NULL) return false;

    if (!(*mc)(1)) return false;

    frame_cnt = mc->FrameCount();
    bbox = mc->GetBoundingBoxes();

    CallGPURenderTaskData* rhs_rtc = m_renderTask_callerSlot.CallAs<CallGPURenderTaskData>();
    if (rhs_rtc != NULL) {
        if (!(*rhs_rtc)(1)) return false;

        frame_cnt = std::min(frame_cnt, rhs_rtc->FrameCount());
        auto osbbox = bbox.ObjectSpaceBBox();
        osbbox.Union(rhs_rtc->AccessBoundingBoxes().ObjectSpaceBBox());
        bbox.SetObjectSpaceBBox(osbbox);
    }

    lhs_rtc->SetFrameCount(frame_cnt);
    lhs_rtc->AccessBoundingBoxes() = bbox;

    return true;
}

void megamol::mesh::AbstractGPURenderTaskDataSource::release() {
    // intentionally empty ?
}
