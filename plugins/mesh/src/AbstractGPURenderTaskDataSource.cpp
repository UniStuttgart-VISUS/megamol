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
        CallGPURenderTaskData::ClassName(), "GetMetaData", &AbstractGPURenderTaskDataSource::getMetaDataCallback);
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

    m_gpu_render_tasks = std::make_shared<GPURenderTaskCollection>();

    return true;
}

bool megamol::mesh::AbstractGPURenderTaskDataSource::getMetaDataCallback(core::Call& caller) {
    

    CallGPURenderTaskData* lhs_rt_call = dynamic_cast<CallGPURenderTaskData*>(&caller);
    CallGPURenderTaskData* rhs_rt_call = m_renderTask_rhs_slot.CallAs<CallGPURenderTaskData>();
    CallGPUMaterialData* material_call = m_material_slot.CallAs<CallGPUMaterialData>();
    CallGPUMeshData* mesh_call = this->m_mesh_slot.CallAs<CallGPUMeshData>();

    if (lhs_rt_call == NULL) return false;
    if (material_call == NULL) return false;
    if (mesh_call == NULL) return false;

    auto lhs_meta_data = lhs_rt_call->getMetaData();
    auto mtl_meta_data = material_call->getMetaData();
    auto mesh_meta_data = mesh_call->getMetaData();
    core::Spatial3DMetaData rhs_meta_data;
    
    mesh_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
    mesh_call->setMetaData(mesh_meta_data);
    if (!(*mesh_call)(1)) return false;
    mesh_meta_data = mesh_call->getMetaData();

    if (rhs_rt_call != NULL) {
        rhs_meta_data = rhs_rt_call->getMetaData();
        rhs_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
        rhs_rt_call->setMetaData(rhs_meta_data);
        if (!(*rhs_rt_call)(1)) return false;
        rhs_meta_data = rhs_rt_call->getMetaData();

        if (rhs_meta_data.m_data_hash > m_renderTask_rhs_cached_hash) {
            m_renderTask_lhs_cached_hash++;
        }
    } else {
        rhs_meta_data.m_frame_cnt = 1;
    }

    if (mtl_meta_data.m_data_hash > m_material_cached_hash) {
        m_renderTask_lhs_cached_hash++;
    }
    if (mesh_meta_data.m_data_hash > m_mesh_cached_hash) {
        m_renderTask_lhs_cached_hash++;
    }

    lhs_meta_data.m_data_hash = m_renderTask_lhs_cached_hash;
    lhs_meta_data.m_frame_cnt = std::min(mesh_meta_data.m_frame_cnt, rhs_meta_data.m_frame_cnt);

    auto bbox = mesh_meta_data.m_bboxs.BoundingBox();
    bbox.Union(rhs_meta_data.m_bboxs.BoundingBox());
    lhs_meta_data.m_bboxs.SetBoundingBox(bbox);

    auto cbbox = mesh_meta_data.m_bboxs.ClipBox();
    cbbox.Union(rhs_meta_data.m_bboxs.ClipBox());
    lhs_meta_data.m_bboxs.SetClipBox(cbbox);

    lhs_rt_call->setMetaData(lhs_meta_data);

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
