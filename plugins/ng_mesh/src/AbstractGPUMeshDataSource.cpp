/*
* AbstractGPUMeshDataSource.cpp
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/
#include "stdafx.h"

#include "ng_mesh/AbstractGPUMeshDataSource.h"
#include "ng_mesh/GPUMeshDataCall.h"


megamol::ngmesh::AbstractGPUMeshDataSource::AbstractGPUMeshDataSource()
	: core::Module(), m_getData_slot("getData", "The slot publishing the loaded data")
{
	this->m_getData_slot.SetCallback(GPUMeshDataCall::ClassName(), "GetData", &AbstractGPUMeshDataSource::getDataCallback);
	this->m_getData_slot.SetCallback(GPUMeshDataCall::ClassName(), "GetExtent", &AbstractGPUMeshDataSource::getExtentCallback);
	this->MakeSlotAvailable(&this->m_getData_slot);
}

megamol::ngmesh::AbstractGPUMeshDataSource::~AbstractGPUMeshDataSource()
{
	this->Release();
}

bool megamol::ngmesh::AbstractGPUMeshDataSource::create(void)
{
	m_gpu_meshes = std::make_shared<GPUMeshCollection>();

	return true;
}

bool megamol::ngmesh::AbstractGPUMeshDataSource::getExtentCallback(core::Call & caller)
{
	GPUMeshDataCall* mc = dynamic_cast<GPUMeshDataCall*>(&caller);
	if (mc == NULL)
		return false;

	mc->SetExtent(
		1,
		this->m_bbox[0],
		this->m_bbox[1],
		this->m_bbox[2],
		this->m_bbox[3],
		this->m_bbox[4],
		this->m_bbox[5]);

	return true;
}

void megamol::ngmesh::AbstractGPUMeshDataSource::release()
{
}