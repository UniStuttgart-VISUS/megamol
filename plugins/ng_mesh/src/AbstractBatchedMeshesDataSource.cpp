/*
* AbstractBatchedMeshesDataSource.cpp
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "stdafx.h"

#include "ng_mesh/AbstractBatchedMeshesDataSource.h"
#include "ng_mesh/BatchedMeshesDataCall.h"

megamol::ngmesh::AbstractBatchedMeshesDataSource::AbstractBatchedMeshesDataSource()
	: core::Module(), m_getData_slot("getData","The slot publishing the loaded data")
{
	this->m_getData_slot.SetCallback(BatchedMeshesDataCall::ClassName(), "GetData", &AbstractBatchedMeshesDataSource::getDataCallback);
	this->m_getData_slot.SetCallback(BatchedMeshesDataCall::ClassName(), "GetExtent", &AbstractBatchedMeshesDataSource::getExtentCallback);
	this->MakeSlotAvailable(&this->m_getData_slot);
}

megamol::ngmesh::AbstractBatchedMeshesDataSource::~AbstractBatchedMeshesDataSource()
{
	this->Release();
}

bool megamol::ngmesh::AbstractBatchedMeshesDataSource::create(void)
{
	// intentionally empty ?
	return true;
}

bool megamol::ngmesh::AbstractBatchedMeshesDataSource::getExtentCallback(core::Call & caller)
{
	BatchedMeshesDataCall* render_batches_call = dynamic_cast<BatchedMeshesDataCall*>(&caller);
	if (render_batches_call == NULL)
		return false;

	render_batches_call->SetExtent(
		1,
		this->m_bbox[0],
		this->m_bbox[1],
		this->m_bbox[2],
		this->m_bbox[3],
		this->m_bbox[4],
		this->m_bbox[5]);
}

void megamol::ngmesh::AbstractBatchedMeshesDataSource::release()
{
}
