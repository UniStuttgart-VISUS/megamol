/*
 * AbstractGPURenderTaskDataSource.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#include "mesh/AbstractGPURenderTaskDataSource.h"
#include "mesh/MeshCalls.h"

megamol::mesh::AbstractGPURenderTaskDataSource::AbstractGPURenderTaskDataSource()
    : core::Module()
    , m_getData_slot("getData", "The slot publishing the loaded data")
    , m_renderTask_callerSlot("getRenderTasks", "The slot for chaining render task data sources.")
    , m_material_callerSlot("getMaterialData", "Connects to a material data source")
    , m_mesh_callerSlot("getMeshData", "Connects to a mesh data source")
    , m_light_slot("lights", "Lights are retrieved over this slot. If no light is connected, a default camera light is used")
{
    this->m_getData_slot.SetCallback(
        CallGPURenderTaskData::ClassName(), "GetData", &AbstractGPURenderTaskDataSource::getDataCallback);
    this->m_getData_slot.SetCallback(
        CallGPURenderTaskData::ClassName(), "GetMetaData", &AbstractGPURenderTaskDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->m_getData_slot);

    this->m_renderTask_callerSlot.SetCompatibleCall<GPURenderTasksDataCallDescription>();
    this->MakeSlotAvailable(&this->m_renderTask_callerSlot);

    this->m_material_callerSlot.SetCompatibleCall<CallGPUMaterialDataDescription>();
    this->MakeSlotAvailable(&this->m_material_callerSlot);

    this->m_mesh_callerSlot.SetCompatibleCall<CallGPUMeshDataDescription>();
    this->MakeSlotAvailable(&this->m_mesh_callerSlot);

    this->m_light_slot.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->MakeSlotAvailable(&this->m_light_slot);
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

    CallGPUMeshData* mc = this->m_mesh_callerSlot.CallAs<CallGPUMeshData>();
    if (mc == NULL) return false;

    if (!(*mc)(1)) return false;

    auto meta_data = mc->getMetaData();

    CallGPURenderTaskData* rhs_rtc = m_renderTask_callerSlot.CallAs<CallGPURenderTaskData>();
    if (rhs_rtc != NULL) {
        if (!(*rhs_rtc)(1)) return false;

        auto rhs_meta_data = rhs_rtc->getMetaData();

        meta_data.m_frame_cnt = std::min(meta_data.m_frame_cnt, rhs_meta_data.m_frame_cnt);
        auto osbbox = meta_data.m_bboxs.ObjectSpaceBBox();
        osbbox.Union(rhs_meta_data.m_bboxs.ObjectSpaceBBox());
        meta_data.m_bboxs.SetObjectSpaceBBox(osbbox);
    }

    lhs_rtc->setMetaData(meta_data);

    return true;
}

bool megamol::mesh::AbstractGPURenderTaskDataSource::GetLights(void) {
    core::view::light::CallLight* cl = this->m_light_slot.CallAs<core::view::light::CallLight>();
    if (cl == nullptr) {
        // TODO add local light
        return false;
    }
    cl->setLightMap(&this->lightMap);
    cl->fillLightMap();
    bool lightDirty = false;
    for (const auto element : this->lightMap) {
        auto light = element.second;
        if (light.dataChanged) {
            lightDirty = true;
        }
    }
    return lightDirty;
}

void megamol::mesh::AbstractGPURenderTaskDataSource::release() {
    // intentionally empty ?
}
