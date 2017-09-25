/*
* AbstractNGMeshDataSource.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "stdafx.h"

#include "ng_mesh/AbstractNGMeshDataSource.h"
#include "ng_mesh/CallNGMeshRenderBatches.h"

using namespace megamol;
using namespace megamol::ngmesh;

AbstractNGMeshDataSource::AbstractNGMeshDataSource()
	: core::Module(),
	m_getData_slot("getData","The slot publishing the loaded data")
{
	this->m_getData_slot.SetCallback(CallNGMeshRenderBatches::ClassName(), "GetData", &AbstractNGMeshDataSource::getDataCallback);
	this->m_getData_slot.SetCallback(CallNGMeshRenderBatches::ClassName(), "GetExtent", &AbstractNGMeshDataSource::getExtentCallback);
	this->MakeSlotAvailable(&this->m_getData_slot);
}

AbstractNGMeshDataSource::~AbstractNGMeshDataSource()
{
	this->Release();
}

bool AbstractNGMeshDataSource::create(void)
{
	// intentionally empty ?
	return true;
}

bool AbstractNGMeshDataSource::getExtentCallback(core::Call& caller)
{
	CallNGMeshRenderBatches* render_batches_call = dynamic_cast<CallNGMeshRenderBatches*>(&caller);
	if (render_batches_call == NULL)
		return false;

	render_batches_call->SetExtent(
		1,
		this->m_bbox.Left(),
		this->m_bbox.Bottom(),
		this->m_bbox.Back(),
		this->m_bbox.Right(),
		this->m_bbox.Top(),
		this->m_bbox.Front());
}

void AbstractNGMeshDataSource::release()
{
	// intentionally empty...render batch data should destruct properly on its own
}