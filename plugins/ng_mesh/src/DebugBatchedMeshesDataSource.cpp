/*
* DebugBatchedMeshesDataSource.cpp
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "stdafx.h"

#include <vector>

#include "DebugBatchedMeshesDataSource.h"

#include "ng_mesh/BatchedMeshesDataCall.h"

megamol::ngmesh::DebugBatchedMeshesDataSource::DebugBatchedMeshesDataSource()
{
	// no slots to load for this debug data source
}

megamol::ngmesh::DebugBatchedMeshesDataSource::~DebugBatchedMeshesDataSource()
{
}

bool megamol::ngmesh::DebugBatchedMeshesDataSource::getDataCallback(core::Call & caller)
{
	BatchedMeshesDataCall* mesh_call = dynamic_cast<BatchedMeshesDataCall*>(&caller);
	if (mesh_call == NULL)
		return false;

	load();

	mesh_call->setBatchedMeshesDataAccessor(&m_mesh_data_accessor);

	return true;
}

bool megamol::ngmesh::DebugBatchedMeshesDataSource::load()
{
	// Create std-container for holding vertex data
	std::vector<std::vector<float>> vbs = { {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f},// normal data buffer
		{-0.5f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f} }; // position data buffer
	// Create std-container holding vertex attribute descriptions
	std::vector<VertexLayout::Attribute> attribs = {
		VertexLayout::Attribute(3,GL_FLOAT,GL_FALSE,0),
		VertexLayout::Attribute(3,GL_FLOAT,GL_FALSE,0) };

	VertexLayout vertex_descriptor(0,attribs);

	// Create std-container holding index data
	std::vector<uint32_t> indices = { 0,1,2 };

	std::vector<std::pair< std::vector<float>::iterator, std::vector<float>::iterator>> vb_iterators = { {vbs[0].begin(),vbs[0].end()},{vbs[1].begin(),vbs[1].end()} };
	std::pair< std::vector<uint32_t>::iterator, std::vector<uint32_t>::iterator> ib_iterators = { indices.begin(),indices.end() };

	m_mesh_data_storage.addMesh(vertex_descriptor, vb_iterators, ib_iterators,GL_UNSIGNED_INT,GL_STATIC_DRAW,GL_TRIANGLES);

	return true;
}
