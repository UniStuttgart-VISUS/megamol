/*
* AbstractNGMeshDataSource.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "stdafx.h"

#include "mmcore/param/FilePathParam.h"

#include "ng_mesh/AbstractNGMeshDataSource.h"
#include "ng_mesh/CallNGMeshRenderBatches.h"

using namespace megamol;
using namespace megamol::ngmesh;

AbstractNGMeshDataSource::AbstractNGMeshDataSource()
	: core::Module(),
	getDataSlot("getData","The slot publishing the loaded data"),
	filenameSlot("filename", "The path to the file to load")
{
	this->getDataSlot.SetCallback(CallNGMeshRenderBatches::ClassName(), "GetData", &AbstractNGMeshDataSource::getDataCallback);
	this->getDataSlot.SetCallback(CallNGMeshRenderBatches::ClassName(), "GetExtent", &AbstractNGMeshDataSource::getExtentCallback);
	this->MakeSlotAvailable(&this->getDataSlot);

	this->filenameSlot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->filenameSlot);
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

bool AbstractNGMeshDataSource::getDataCallback(core::Call& caller)
{
	CallNGMeshRenderBatches* render_batches_call = dynamic_cast<CallNGMeshRenderBatches*>(&caller);
	if (render_batches_call == NULL)
		return false;

	if (this->filenameSlot.IsDirty())
	{
		this->filenameSlot.ResetDirty();

		// Clear render batches TODO: add explicit clear function?
		CallNGMeshRenderBatches::RenderBatchesData empty_render_batches;
		m_render_batches = empty_render_batches;

		m_bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);

		auto vislib_filename = filenameSlot.Param<core::param::FilePathParam>()->Value();
		std::string filename(vislib_filename.PeekBuffer());
	
		// TODO try/catch?
		load(filename);
	}

	render_batches_call->setRenderBatches(&m_render_batches);

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