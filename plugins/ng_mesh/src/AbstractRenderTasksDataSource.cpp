/*
* AbstractRenderTasksDataSource.cpp
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "stdafx.h"

#include "ng_mesh/AbstractRenderTasksDataSource.h"
#include "ng_mesh/RenderTasksDataCall.h"

megamol::ngmesh::AbstractRenderTasksDataSource::AbstractRenderTasksDataSource()
	: core::Module(), m_getData_slot("getData", "The slot publishing the loaded data")
{
	this->m_getData_slot.SetCallback(RenderTasksDataCall::ClassName(), "GetData", &AbstractRenderTasksDataSource::getDataCallback);
	this->MakeSlotAvailable(&this->m_getData_slot);
}

megamol::ngmesh::AbstractRenderTasksDataSource::~AbstractRenderTasksDataSource()
{
	this->Release();
}

bool megamol::ngmesh::AbstractRenderTasksDataSource::create(void)
{
	// intentionally empty ?
	return true;
}

void megamol::ngmesh::AbstractRenderTasksDataSource::release()
{
	// intentionally empty ?
}
