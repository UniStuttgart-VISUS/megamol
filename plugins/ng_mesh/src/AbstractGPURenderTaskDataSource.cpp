/*
* AbstractGPURenderTaskDataSource.cpp
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "stdafx.h"

#include "ng_mesh/AbstractGPURenderTaskDataSource.h"
#include "ng_mesh/GPURenderTaskDataCall.h"
#include "ng_mesh/GPUMeshDataCall.h"
#include "ng_mesh/GPUMaterialDataCall.h"

megamol::ngmesh::AbstractGPURenderTaskDataSource::AbstractGPURenderTaskDataSource()
	: core::Module(),
    m_getData_slot("getData", "The slot publishing the loaded data"),
	m_renderTask_callerSlot("getRenderTasks", "The slot for chaining render task data sources."),
	m_material_callerSlot("getMaterialData", "Connects to a material data source"),
    m_mesh_callerSlot("getMeshData", "Connects to a mesh data source")
{
	this->m_getData_slot.SetCallback(GPURenderTaskDataCall::ClassName(), "GetData", &AbstractGPURenderTaskDataSource::getDataCallback);
	this->m_getData_slot.SetCallback(GPURenderTaskDataCall::ClassName(), "GetExtent", &AbstractGPURenderTaskDataSource::getExtentCallback);
	this->MakeSlotAvailable(&this->m_getData_slot);

    this->m_renderTask_callerSlot.SetCompatibleCall<GPURenderTasksDataCallDescription>();
    this->MakeSlotAvailable(&this->m_renderTask_callerSlot);

	this->m_material_callerSlot.SetCompatibleCall<GPUMaterialDataCallDescription>();
	this->MakeSlotAvailable(&this->m_material_callerSlot);

    this->m_mesh_callerSlot.SetCompatibleCall<GPUMeshDataCallDescription>();
    this->MakeSlotAvailable(&this->m_mesh_callerSlot);
}

megamol::ngmesh::AbstractGPURenderTaskDataSource::~AbstractGPURenderTaskDataSource()
{
	this->Release();
}

bool megamol::ngmesh::AbstractGPURenderTaskDataSource::create(void)
{
	// intentionally empty ?

	m_gpu_render_tasks = std::make_shared<GPURenderTaskCollection>();

	return true;
}

bool megamol::ngmesh::AbstractGPURenderTaskDataSource::getExtentCallback(core::Call & caller)
{
	GPURenderTaskDataCall* lhs_rtc = dynamic_cast<GPURenderTaskDataCall*>(&caller);
    if (lhs_rtc == NULL)
		return false;

    unsigned int frame_cnt;
    megamol::core::BoundingBoxes bbox;

	GPUMeshDataCall* mc = this->m_mesh_callerSlot.CallAs<GPUMeshDataCall>();
	if (mc == NULL)
		return false;

	if (!(*mc)(1))
		return false;

    frame_cnt = mc->FrameCount();
    bbox = mc->GetBoundingBoxes();

    GPURenderTaskDataCall* rhs_rtc = m_renderTask_callerSlot.CallAs<GPURenderTaskDataCall>();
    if (rhs_rtc != NULL)
    {
        if (!(*rhs_rtc)(1))
            return false;

        frame_cnt = std::min(frame_cnt, rhs_rtc->FrameCount());
        auto osbbox = bbox.ObjectSpaceBBox();
        osbbox.Union(rhs_rtc->AccessBoundingBoxes().ObjectSpaceBBox());
        bbox.SetObjectSpaceBBox(osbbox);
    }

	lhs_rtc->SetFrameCount(frame_cnt);
    lhs_rtc->AccessBoundingBoxes() = bbox;

	return true;
}

void megamol::ngmesh::AbstractGPURenderTaskDataSource::release()
{
	// intentionally empty ?
}
