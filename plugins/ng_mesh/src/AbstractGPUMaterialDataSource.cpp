/*
* AbstractGPUMaterialsDataSource.cpp
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "stdafx.h"

#include "ng_mesh/AbstractGPUMaterialDataSource.h"
#include "ng_mesh/GPUMaterialDataCall.h"

megamol::ngmesh::AbstractGPUMaterialDataSource::AbstractGPUMaterialDataSource()
	: core::Module(),
	m_gpu_materials(std::make_shared<GPUMaterialDataStorage>()),
	m_getData_slot("getData", "The slot publishing the loaded data")
{
	this->m_getData_slot.SetCallback(GPUMaterialDataCall::ClassName(), "GetData", &AbstractGPUMaterialDataSource::getDataCallback);
	this->MakeSlotAvailable(&this->m_getData_slot);
}

megamol::ngmesh::AbstractGPUMaterialDataSource::~AbstractGPUMaterialDataSource()
{
	this->Release();
}

bool megamol::ngmesh::AbstractGPUMaterialDataSource::create(void)
{
	// intentionally empty ?
	return true;
}

void megamol::ngmesh::AbstractGPUMaterialDataSource::release()
{
	// intentionally empty ?
}


