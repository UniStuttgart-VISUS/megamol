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
	: core::Module(), m_getData_slot("getData", "The slot publishing the loaded data"),
	m_mesh_callerSlot("getMeshData", "Connects the renderer with a mesh data source"),
	m_material_callerSlot("getMaterialData", "Connects the renderer with a material data source")
{
	this->m_getData_slot.SetCallback(GPURenderTaskDataCall::ClassName(), "GetData", &AbstractGPURenderTaskDataSource::getDataCallback);
	this->m_getData_slot.SetCallback(GPURenderTaskDataCall::ClassName(), "GetExtent", &AbstractGPURenderTaskDataSource::getExtentCallback);
	this->MakeSlotAvailable(&this->m_getData_slot);

	this->m_mesh_callerSlot.SetCompatibleCall<GPUMeshDataCallDescription>();
	this->MakeSlotAvailable(&this->m_mesh_callerSlot);

	this->m_material_callerSlot.SetCompatibleCall<GPUMaterialDataCallDescription>();
	this->MakeSlotAvailable(&this->m_material_callerSlot);
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
	GPURenderTaskDataCall* rtc = dynamic_cast<GPURenderTaskDataCall*>(&caller);
	if (rtc == NULL)
		return false;

	GPUMeshDataCall* mc = this->m_mesh_callerSlot.CallAs<GPUMeshDataCall>();
	if (mc == NULL)
		return false;

	if (!(*mc)(1))
		return false;

	rtc->SetFrameCount(mc->FrameCount());
	rtc->AccessBoundingBoxes() = mc->GetBoundingBoxes();

	return true;
}

void megamol::ngmesh::AbstractGPURenderTaskDataSource::release()
{
	// intentionally empty ?
}
