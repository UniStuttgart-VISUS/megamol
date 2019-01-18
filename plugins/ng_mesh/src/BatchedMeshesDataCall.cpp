/*
* BatchedMeshesDataCall.cpp
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "stdafx.h"
#include "..\include\ng_mesh\BatchedMeshesDataCall.h"

megamol::ngmesh::BatchedMeshesDataAccessor::BatchedMeshesDataAccessor()
	: buffer_accessors(nullptr),
	buffer_accessor_cnt(0),
	draw_command_batches(nullptr),
	mesh_data_batches(nullptr),
	batch_cnt(0)
{
	
}

megamol::ngmesh::BatchedMeshesDataAccessor::~BatchedMeshesDataAccessor()
{
	if (buffer_accessors != nullptr) delete buffer_accessors;
	if (mesh_data_batches != nullptr) delete mesh_data_batches;
	if (draw_command_batches != nullptr) delete draw_command_batches;
}

size_t megamol::ngmesh::BatchedMeshesDataAccessor::allocateNewBatch(size_t mesh_vertex_buffer_cnt)
{
	// allocate new (larger) memory for buffer accessors
	BufferAccessor* new_buffer_accessors = new BufferAccessor[buffer_accessor_cnt + mesh_vertex_buffer_cnt + 1];
	// copy existing memory
	std::copy(buffer_accessors, buffer_accessors + buffer_accessor_cnt, new_buffer_accessors);
	// update buffer accessor count
	buffer_accessor_cnt = buffer_accessor_cnt + mesh_vertex_buffer_cnt + 1;
	// delete buffer_accessors and reassign pointers
	if (buffer_accessors != nullptr) delete buffer_accessors;
	buffer_accessors = new_buffer_accessors;

	// allocate new (larger) memory for batches
	MeshDataAccessor* new_mesh_data_batches = new MeshDataAccessor[batch_cnt+1];
	DrawCommandsDataAccessor* new_draw_command_batches = new DrawCommandsDataAccessor[batch_cnt+1];
	// copy existing data
	std::copy(mesh_data_batches, mesh_data_batches + batch_cnt, new_mesh_data_batches);
	std::copy(draw_command_batches, draw_command_batches + batch_cnt, new_draw_command_batches);
	// increment batch cnt
	++batch_cnt;
	// delete old memory and reassign pointers
	if (mesh_data_batches != nullptr) delete mesh_data_batches;
	if (draw_command_batches != nullptr) delete draw_command_batches;
	mesh_data_batches = new_mesh_data_batches;
	draw_command_batches = new_draw_command_batches;

	// set buffer indices of buffer accessors in new batch
	mesh_data_batches[batch_cnt - 1].vertex_buffers_accessors_base_index 
		= buffer_accessor_cnt - mesh_vertex_buffer_cnt - 1;
	mesh_data_batches[batch_cnt - 1].index_buffer_accessor_index = buffer_accessor_cnt - 1;

	// return index of new batch
	return (batch_cnt - 1);
}

void megamol::ngmesh::BatchedMeshesDataAccessor::setVertexDataAccess(size_t batch_idx, size_t vertex_buffers_cnt, BufferAccessor ...)
{
	va_list args;
	va_start(args, vertex_buffers_cnt);
	for (size_t i = 0; i < vertex_buffers_cnt; ++i)
	{
		size_t buffer_idx = mesh_data_batches[batch_idx].vertex_buffers_accessors_base_index + i;

		BufferAccessor accessor = va_arg(args, BufferAccessor);

		buffer_accessors[buffer_idx].raw_data = accessor.raw_data;
		buffer_accessors[buffer_idx].byte_size = accessor.byte_size;
	}
}

void megamol::ngmesh::BatchedMeshesDataAccessor::setIndexDataAccess(size_t batch_idx, std::byte * raw_data, size_t byte_size, GLenum index_type)
{
	size_t buffer_idx = mesh_data_batches[batch_idx].index_buffer_accessor_index;
	buffer_accessors[buffer_idx].raw_data = raw_data;
	buffer_accessors[buffer_idx].byte_size = byte_size;
	mesh_data_batches[batch_idx].index_type = index_type;
}

void megamol::ngmesh::BatchedMeshesDataAccessor::setMeshMetaDataAccess(size_t batch_idx, GLsizei stride, GLuint attribute_cnt, VertexLayout::Attribute * attributes, GLenum usage, GLenum primitive_type)
{
	mesh_data_batches[batch_idx].vertex_stride = stride;
	mesh_data_batches[batch_idx].vertex_attribute_cnt = attribute_cnt;
	mesh_data_batches[batch_idx].vertex_attributes = attributes;
	mesh_data_batches[batch_idx].usage = usage;
	mesh_data_batches[batch_idx].primitive_type = primitive_type;
}

void megamol::ngmesh::BatchedMeshesDataAccessor::setDrawCommandsDataAcess(size_t batch_idx, DrawElementsCommand * draw_commands, GLsizei draw_cnt)
{
	draw_command_batches[batch_idx].draw_commands = draw_commands;
	draw_command_batches[batch_idx].draw_cnt = draw_cnt;
	
}

megamol::ngmesh::BatchedMeshesDataCall::BatchedMeshesDataCall()
	: megamol::core::AbstractGetData3DCall(), m_data_accessor(nullptr)
{
}

megamol::ngmesh::BatchedMeshesDataCall::~BatchedMeshesDataCall()
{
}
