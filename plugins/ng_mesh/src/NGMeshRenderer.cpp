/*
* NGMeshRenderer.cpp
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "NGMeshRenderer.h"

megamol::ngmesh::NGMeshRenderer::NGMeshRenderer()
	: Renderer3DModule(), m_renderBatches_callerSlot("getData", "Connects the mesh renderer with a mesh data source")
{
	this->m_renderBatches_callerSlot.SetCompatibleCall<CallNGMeshRenderBatchesDescription>();
	this->MakeSlotAvailable(&this->m_renderBatches_callerSlot);
}

megamol::ngmesh::NGMeshRenderer::~NGMeshRenderer()
{

}

bool megamol::ngmesh::NGMeshRenderer::create()
{
	//TODO
	return false;
}

void megamol::ngmesh::NGMeshRenderer::release()
{

}

bool megamol::ngmesh::NGMeshRenderer::GetCapabilities(megamol::core::Call& call)
{
	//TODO
	return false;
}

bool megamol::ngmesh::NGMeshRenderer::GetExtents(megamol::core::Call& call)
{
	//TODO
	return false;
}

bool megamol::ngmesh::NGMeshRenderer::Render(megamol::core::Call& call)
{
	//TODO
	return false;
}