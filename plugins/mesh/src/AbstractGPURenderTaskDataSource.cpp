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
    , m_renderTask_lhs_slot("getData", "The slot publishing the loaded data")
    , m_renderTask_lhs_cached_hash(0)
    , m_renderTask_rhs_slot("getRenderTasks", "The slot for chaining render task data sources.")
    , m_renderTask_rhs_cached_hash(0)
    , m_material_slot("getMaterialData", "Connects to a material data source")
    , m_material_cached_hash(0)
    , m_mesh_slot("getMeshData", "Connects to a mesh data source")
    , m_mesh_cached_hash(0)
    , m_light_slot("lights", "Lights are retrieved over this slot. If no light is connected, a default camera light is used")
    , m_light_cached_hash(0)
{
    this->m_renderTask_lhs_slot.SetCallback(
        CallGPURenderTaskData::ClassName(), "GetData", &AbstractGPURenderTaskDataSource::getDataCallback);
    this->m_renderTask_lhs_slot.SetCallback(
        CallGPURenderTaskData::ClassName(), "GetMetaData", &AbstractGPURenderTaskDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->m_renderTask_lhs_slot);

    this->m_renderTask_rhs_slot.SetCompatibleCall<GPURenderTasksDataCallDescription>();
    this->MakeSlotAvailable(&this->m_renderTask_rhs_slot);

    this->m_material_slot.SetCompatibleCall<CallGPUMaterialDataDescription>();
    this->MakeSlotAvailable(&this->m_material_slot);

    this->m_mesh_slot.SetCompatibleCall<CallGPUMeshDataDescription>();
    this->MakeSlotAvailable(&this->m_mesh_slot);

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
    // get lhs call and meta data
    CallGPURenderTaskData* lhs_rtc = dynamic_cast<CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL) return false;
    auto lhs_meta_data = lhs_rtc->getMetaData();
    //TODO do I need to use the cached hash for lhs here?

    // get mesh call, set frame id, do meta data callback
    CallGPUMeshData* mc = this->m_mesh_slot.CallAs<CallGPUMeshData>();
    if (mc == NULL) return false;
    auto mesh_meta_data = mc->getMetaData();
    mesh_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
    if (!(*mc)(1)) return false;
    mesh_meta_data = mc->getMetaData();
    
    CallGPURenderTaskData* rhs_rtc = m_renderTask_rhs_slot.CallAs<CallGPURenderTaskData>();
    if (rhs_rtc != NULL) {
        auto rhs_meta_data = rhs_rtc->getMetaData();
        rhs_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
        if (!(*rhs_rtc)(1)) return false;
        rhs_meta_data = rhs_rtc->getMetaData();

        mesh_meta_data.m_frame_cnt = std::min(mesh_meta_data.m_frame_cnt, rhs_meta_data.m_frame_cnt);
        auto osbbox = mesh_meta_data.m_bboxs.ObjectSpaceBBox();
        osbbox.Union(rhs_meta_data.m_bboxs.ObjectSpaceBBox());
        mesh_meta_data.m_bboxs.SetObjectSpaceBBox(osbbox);
    }

    // finish by updating lhs meta data with union of mesh and rhs meta data
    lhs_rtc->setMetaData(mesh_meta_data);

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
