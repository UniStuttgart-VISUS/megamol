/*
* AbstractGPURenderTaskDataSource.cpp
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "stdafx.h"

#include "ng_mesh/AbstractGPURenderTaskDataSource.h"
#include "ng_mesh/GPURenderTaskDataCall.h"

megamol::ngmesh::AbstractGPURenderTaskDataSource::AbstractGPURenderTaskDataSource()
	: core::Module(), m_getData_slot("getData", "The slot publishing the loaded data")
{
	this->m_getData_slot.SetCallback(GPURenderTaskDataCall::ClassName(), "GetData", &AbstractGPURenderTaskDataSource::getDataCallback);
	this->MakeSlotAvailable(&this->m_getData_slot);
}

megamol::ngmesh::AbstractGPURenderTaskDataSource::~AbstractGPURenderTaskDataSource()
{
	this->Release();
}

bool megamol::ngmesh::AbstractGPURenderTaskDataSource::create(void)
{
	// intentionally empty ?
	return true;
}

void megamol::ngmesh::AbstractGPURenderTaskDataSource::release()
{
	// intentionally empty ?
}
