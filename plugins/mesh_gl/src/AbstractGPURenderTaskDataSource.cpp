/*
 * AbstractGPURenderTaskDataSource.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "mesh_gl/AbstractGPURenderTaskDataSource.h"
#include "mesh_gl/MeshCalls_gl.h"

megamol::mesh_gl::AbstractGPURenderTaskDataSource::AbstractGPURenderTaskDataSource()
        : core::Module()
        , m_rendertask_collection({nullptr, {}})
        , m_renderTask_lhs_slot("renderTasks", "The slot publishing the loaded data")
        , m_renderTask_rhs_slot("chainRenderTasks", "The slot for chaining render task data sources.")
        , m_mesh_slot("gpuMeshes", "Connects to a mesh data source") {
    this->m_renderTask_lhs_slot.SetCallback(
        CallGPURenderTaskData::ClassName(), "GetData", &AbstractGPURenderTaskDataSource::getDataCallback);
    this->m_renderTask_lhs_slot.SetCallback(
        CallGPURenderTaskData::ClassName(), "GetMetaData", &AbstractGPURenderTaskDataSource::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_renderTask_lhs_slot);

    this->m_renderTask_rhs_slot.SetCompatibleCall<GPURenderTasksDataCallDescription>();
    this->MakeSlotAvailable(&this->m_renderTask_rhs_slot);

    this->m_mesh_slot.SetCompatibleCall<CallGPUMeshDataDescription>();
    this->MakeSlotAvailable(&this->m_mesh_slot);
}

megamol::mesh_gl::AbstractGPURenderTaskDataSource::~AbstractGPURenderTaskDataSource() {
    this->Release();
}

bool megamol::mesh_gl::AbstractGPURenderTaskDataSource::create(void) {
    // default empty collection
    m_rendertask_collection.first = std::make_shared<GPURenderTaskCollection>();
    return true;
}

bool megamol::mesh_gl::AbstractGPURenderTaskDataSource::getMetaDataCallback(core::Call& caller) {

    CallGPURenderTaskData* lhs_rt_call = dynamic_cast<CallGPURenderTaskData*>(&caller);
    CallGPURenderTaskData* rhs_rt_call = m_renderTask_rhs_slot.CallAs<CallGPURenderTaskData>();
    CallGPUMeshData* mesh_call = this->m_mesh_slot.CallAs<CallGPUMeshData>();

    if (lhs_rt_call == NULL)
        return false;
    auto lhs_meta_data = lhs_rt_call->getMetaData();

    unsigned int frame_cnt = std::numeric_limits<unsigned int>::max();
    auto bbox = lhs_meta_data.m_bboxs.BoundingBox();
    auto cbbox = lhs_meta_data.m_bboxs.ClipBox();


    if (rhs_rt_call != NULL) {
        auto rhs_meta_data = rhs_rt_call->getMetaData();
        rhs_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
        rhs_rt_call->setMetaData(rhs_meta_data);
        if (!(*rhs_rt_call)(1))
            return false;
        rhs_meta_data = rhs_rt_call->getMetaData();

        frame_cnt = std::min(rhs_meta_data.m_frame_cnt, frame_cnt);

        if (rhs_meta_data.m_bboxs.IsBoundingBoxValid()) {
            bbox.Union(rhs_meta_data.m_bboxs.BoundingBox());
        }
        if (rhs_meta_data.m_bboxs.IsClipBoxValid()) {
            cbbox.Union(rhs_meta_data.m_bboxs.ClipBox());
        }
    }

    if (mesh_call != NULL) {
        auto mesh_meta_data = mesh_call->getMetaData();
        mesh_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
        mesh_call->setMetaData(mesh_meta_data);
        if (!(*mesh_call)(1))
            return false;
        mesh_meta_data = mesh_call->getMetaData();

        frame_cnt = std::min(mesh_meta_data.m_frame_cnt, frame_cnt);

        if (mesh_meta_data.m_bboxs.IsBoundingBoxValid()) {
            bbox.Union(mesh_meta_data.m_bboxs.BoundingBox());
        }
        if (mesh_meta_data.m_bboxs.IsClipBoxValid()) {
            cbbox.Union(mesh_meta_data.m_bboxs.ClipBox());
        }
    }

    lhs_meta_data.m_frame_cnt = frame_cnt;
    lhs_meta_data.m_bboxs.SetBoundingBox(bbox);
    lhs_meta_data.m_bboxs.SetClipBox(cbbox);

    lhs_rt_call->setMetaData(lhs_meta_data);

    return true;
}

void megamol::mesh_gl::AbstractGPURenderTaskDataSource::clearRenderTaskCollection() {
    m_rendertask_collection.first->clear();
    m_rendertask_collection.second.clear();
}

void megamol::mesh_gl::AbstractGPURenderTaskDataSource::release() {
    // intentionally empty ?
}
