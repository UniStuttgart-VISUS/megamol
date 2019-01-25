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
	batch_cnt(0),
	mesh_index_lut(nullptr),
	mesh_cnt(0)
{
	
}

megamol::ngmesh::BatchedMeshesDataAccessor::~BatchedMeshesDataAccessor()
{
	if (buffer_accessors != nullptr) delete buffer_accessors;
	if (mesh_data_batches != nullptr) delete mesh_data_batches;
	if (draw_command_batches != nullptr) delete draw_command_batches;
}

megamol::ngmesh::BatchedMeshesDataAccessor::BatchedMeshesDataAccessor(const BatchedMeshesDataAccessor & cpy)
	: buffer_accessors(nullptr),
	buffer_accessor_cnt(0),
	draw_command_batches(nullptr),
	mesh_data_batches(nullptr),
	batch_cnt(0),
	mesh_index_lut(nullptr),
	mesh_cnt(0)
{
	for (size_t i = 0; i < cpy.batch_cnt; i++)
	{
		MeshDataAccessor& cpy_mesh_data = cpy.mesh_data_batches[i];

		allocateNewBatch(cpy_mesh_data.vertex_buffer_cnt);

		for (size_t j = 0; j < cpy_mesh_data.vertex_buffer_cnt; j++)
		{
			BufferAccessor tmp = cpy.buffer_accessors[cpy_mesh_data.vertex_buffers_accessors_base_index + j];
			setVertexDataAccessor(i, j, tmp);
		}

		BufferAccessor index_data_accessor = cpy.buffer_accessors[cpy_mesh_data.index_buffer_accessor_index];

		setIndexDataAccess(i, index_data_accessor.raw_data, index_data_accessor.byte_size, cpy_mesh_data.index_type);

		setMeshMetaDataAccess(
			i,
			cpy_mesh_data.vertex_stride,
			cpy_mesh_data.vertex_attribute_cnt,
			cpy_mesh_data.vertex_attributes,
			cpy_mesh_data.usage,
			cpy_mesh_data.primitive_type);

		setDrawCommandsDataAcess(
			i,
			cpy.draw_command_batches[i].draw_commands,
			cpy.draw_command_batches[i].draw_cnt);
	}

	this->mesh_index_lut = cpy.mesh_index_lut;
	this->mesh_cnt = cpy.mesh_cnt;
}

megamol::ngmesh::BatchedMeshesDataAccessor::BatchedMeshesDataAccessor(BatchedMeshesDataAccessor && other)
	: buffer_accessors(nullptr),
	buffer_accessor_cnt(0),
	draw_command_batches(nullptr),
	mesh_data_batches(nullptr),
	batch_cnt(0),
	mesh_index_lut(nullptr),
	mesh_cnt(0)
{
	BufferAccessor* tmp_buffer_accessors = this->buffer_accessors;
	size_t tmp_buffer_accessor_cnt = this->buffer_accessor_cnt;
	MeshDataAccessor* tmp_mesh_data_batches = this->mesh_data_batches;
	DrawCommandsDataAccessor* tmp_draw_command_batches = this->draw_command_batches;
	size_t tmp_batch_cnt = this->batch_cnt;
	MeshIndexLookup* tmp_mesh_index_lut = this->mesh_index_lut;
	size_t tmp_mesh_cnt = this->mesh_cnt;

	this->buffer_accessors = other.buffer_accessors;
	this->buffer_accessor_cnt = other.buffer_accessor_cnt;
	this->mesh_data_batches = other.mesh_data_batches;
	this->draw_command_batches = other.draw_command_batches;
	this->batch_cnt = other.batch_cnt;
	this->mesh_index_lut = other.mesh_index_lut;
	this->mesh_cnt = other.mesh_cnt;

	other.buffer_accessors = tmp_buffer_accessors;
	other.buffer_accessor_cnt = tmp_buffer_accessor_cnt;
	other.mesh_data_batches = tmp_mesh_data_batches;
	other.draw_command_batches = tmp_draw_command_batches;
	other.batch_cnt = tmp_batch_cnt;
	other.mesh_index_lut = tmp_mesh_index_lut;
	other.mesh_cnt = tmp_mesh_cnt;
}

megamol::ngmesh::BatchedMeshesDataAccessor & megamol::ngmesh::BatchedMeshesDataAccessor::operator=(BatchedMeshesDataAccessor && rhs)
{
	BufferAccessor* tmp_buffer_accessors = this->buffer_accessors;
	size_t tmp_buffer_accessor_cnt = this->buffer_accessor_cnt;
	MeshDataAccessor* tmp_mesh_data_batches = this->mesh_data_batches;
	DrawCommandsDataAccessor* tmp_draw_command_batches = this->draw_command_batches;
	size_t tmp_batch_cnt = this->batch_cnt;
	MeshIndexLookup* tmp_mesh_index_lut = this->mesh_index_lut;
	size_t tmp_mesh_cnt = this->mesh_cnt;

	this->buffer_accessors = rhs.buffer_accessors;
	this->buffer_accessor_cnt = rhs.buffer_accessor_cnt;
	this->mesh_data_batches = rhs.mesh_data_batches;
	this->draw_command_batches = rhs.draw_command_batches;
	this->batch_cnt = rhs.batch_cnt;
	this->mesh_index_lut = rhs.mesh_index_lut;
	this->mesh_cnt = rhs.mesh_cnt;

	rhs.buffer_accessors = tmp_buffer_accessors;
	rhs.buffer_accessor_cnt = tmp_buffer_accessor_cnt;
	rhs.mesh_data_batches = tmp_mesh_data_batches;
	rhs.draw_command_batches = tmp_draw_command_batches;
	rhs.batch_cnt = tmp_batch_cnt;
	rhs.mesh_index_lut = tmp_mesh_index_lut;
	rhs.mesh_cnt = tmp_mesh_cnt;

	return *this;
}

megamol::ngmesh::BatchedMeshesDataAccessor & megamol::ngmesh::BatchedMeshesDataAccessor::operator=(const BatchedMeshesDataAccessor & rhs)
{
	// TODO: insert return statement here
	*this = BatchedMeshesDataAccessor(rhs);
	return *this;
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
	mesh_data_batches[batch_cnt - 1].vertex_buffer_cnt = mesh_vertex_buffer_cnt;
	mesh_data_batches[batch_cnt - 1].index_buffer_accessor_index = buffer_accessor_cnt - 1;

	// return index of new batch
	return (batch_cnt - 1);
}

//void megamol::ngmesh::BatchedMeshesDataAccessor::setVertexDataAccess(size_t batch_idx, size_t vertex_buffers_cnt, BufferAccessor ...)
//{
//	va_list args;
//	va_start(args, vertex_buffers_cnt);
//	for (size_t i = 0; i < vertex_buffers_cnt; ++i)
//	{
//		size_t buffer_idx = mesh_data_batches[batch_idx].vertex_buffers_accessors_base_index + i;
//
//		BufferAccessor accessor = va_arg(args, BufferAccessor);
//
//		buffer_accessors[buffer_idx].raw_data = accessor.raw_data;
//		buffer_accessors[buffer_idx].byte_size = accessor.byte_size;
//	}
//}

void megamol::ngmesh::BatchedMeshesDataAccessor::setVertexDataAccessor(size_t batch_idx, size_t vertex_buffer_idx, BufferAccessor data_accessor)
{
	size_t buffer_idx = mesh_data_batches[batch_idx].vertex_buffers_accessors_base_index + vertex_buffer_idx;

	buffer_accessors[buffer_idx].raw_data = data_accessor.raw_data;
	buffer_accessors[buffer_idx].byte_size = data_accessor.byte_size;
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
	: megamol::core::AbstractGetData3DCall(), m_data_accessor(nullptr), m_update_flags(UPDATE_ALL_BIT)
{
}

megamol::ngmesh::BatchedMeshesDataCall::~BatchedMeshesDataCall()
{
}
