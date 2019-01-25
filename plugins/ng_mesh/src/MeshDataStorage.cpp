/*
* DebugBatchedMeshesDataSource.cpp
*
* Copyright(C) 2019 by Universitaet Stuttgart(VISUS).
* All rights reserved.
*/

#include "stdafx.h"

#include "..\include\ng_mesh\MeshDataStorage.h"

using namespace megamol::ngmesh;

BatchedMeshesDataAccessor megamol::ngmesh::MeshDataStorage::generateDataAccessor()
{
	BatchedMeshesDataAccessor retval;

	auto it = m_mesh_batches.begin();

	for (size_t i = 0; i < m_mesh_batches.size(); i++)
	{
		retval.allocateNewBatch( it->mesh_data.vertex_data.size() );

		for (size_t j = 0; j < it->mesh_data.vertex_data.size(); j++)
		{
			BufferAccessor vb_data_accessor;
			vb_data_accessor.raw_data = it->mesh_data.vertex_data[j].data();
			vb_data_accessor.byte_size = it->mesh_data.vertex_data[j].size();
			retval.setVertexDataAccessor(i, j, vb_data_accessor);
		}

		BufferAccessor ib_data_accessor;
		ib_data_accessor.raw_data = it->mesh_data.index_data.data();
		ib_data_accessor.byte_size = it->mesh_data.index_data.size();

		retval.setIndexDataAccess(
			i,
			ib_data_accessor.raw_data,
			ib_data_accessor.byte_size,
			it->mesh_data.index_type);

		retval.setMeshMetaDataAccess(
			i,
			it->mesh_data.vertex_descriptor.stride,
			it->mesh_data.vertex_descriptor.attributes.size(),
			it->mesh_data.vertex_descriptor.attributes.data(),
			it->mesh_data.usage,
			it->mesh_data.primitive_type);

		retval.setDrawCommandsDataAcess(
			i,
			it->draw_commands.data(),
			it->draw_commands.size());

		++it;
	}

	retval.mesh_index_lut = reinterpret_cast<MeshIndexLookup*>(m_mesh_lut.data());
	retval.mesh_cnt = m_mesh_lut.size();

	return retval;
}
