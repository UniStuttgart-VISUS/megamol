/*
* NGMeshRenderBatchBakery.h
*
* Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef NG_MESH_RENDER_BATCH_BAKERY_H_INCLUDED
#define NG_MESH_RENDER_BATCH_BAKERY_H_INCLUDED

#include <tuple>

#include "CallNGMeshRenderBatches.h"

namespace megamol {
namespace ngmesh {

	namespace {

	}

	typedef MeshDataAccessor::VertexData::Buffer VertexBufferAccessor;

	/**
	* Build a vector of vertex buffer accessors, i.e. a vector of MeshDataAccessor::VertexData::Buffer,
	* where each element holds a pointer to the memory of the vertex buffer and the respective byte size.
	*
	* @param vertex_buffers Nested std-containers containing vertex buffer data. The outer container is expected
	* to hold either one std-container (i.e. a single vertex buffer) for interleaved vertex data
	* or several std-containers for non-interleaved vertex data. The inner containers contain the actual
	* vertex data. In case of non-interleaved data, each inner container is expected to contain all data for
	* one vertex attribute. Technically, a single inner container containing all non-interleaved data should
	* also be possible if the offsets of all vertex attributes are set accordingly, but this has not been tested yet.
	* Example: std::vector<std::vector<float>>
	*
	* @return Vector of vertex buffer accessors for the given vertex buffer data
	*/
	template<typename VertexBufferContainer>
	std::vector<VertexBufferAccessor> buildVertexBufferAccessors(VertexBufferContainer& vertex_buffers)
	{
		std::vector<VertexBufferAccessor> rtn(vertex_buffers.size());

		for (int i = 0; i < rtn.size(); ++i)
		{
			rtn[i].byte_size = vertex_buffers[i].size() * sizeof(VertexBufferContainer::value_type::value_type);
			rtn[i].raw_data = reinterpret_cast<uint8_t*>( vertex_buffers[i].data() );
		}

		return rtn;
	}

	/** 
	* Build a vector of vertex buffer accessors, i.e. a vector of MeshDataAccessor::VertexData::Buffer,
	* where each element holds a pointer to the memory of the vertex buffer and the respective byte size.
	*
	* @param vb_ptrs Std-container containing pointers to vertex buffer data
	* @param vb_byteSizes Std-container containing byte sizes of vertex buffer data
	*
	* @return Vector of vertex buffer accessors for the given vertex buffer data
	*/
	template<typename VertexBufferPtrContainer, typename VertexBufferByteSizeContainer>
	std::vector<VertexBufferAccessor> buildVertexBufferAccessors(
		VertexBufferPtrContainer const& vb_ptrs,
		VertexBufferByteSizeContainer const & vb_byteSizes)
	{
		std::vector<VertexBufferAccessor> rtn(vb_ptrs.size());

		for (int i = 0; i < rtn.size(); ++i)
		{
			rtn[i].byte_size = vb_byteSizes[i];
			rtn[i].raw_data = reinterpret_cast<uint8_t*>(vb_ptrs[i]);
		}

		return rtn;
	}


	/**
	* Build and return a mesh data accessor struct for the given mesh data.
	*
	* @param vb_accessors Vector of valid VertexBufferAccessor. See buildVertexBufferAccessors(...)
	* @param attributes Description of vertex attributes layout stored in a std-container with a value_type of MeshDataAccessor::VertexLayoutData::Attribute
	* @param attributes_stride The stride values used by all vertex attributes (individual stride per attribute not supported for now)
	* @param index_data Index buffer data stored in a std-container
	* @param index_type Datatype of indices stored in the index buffer
	* @param mesh_usage Usage type of the mesh data, e.g. GL_STATIC_DRAW
	* @param mesh_primtive_type Primtive type of the mesh data, e.g. GL_TRIANGLE
    *
    * @return A valid MeshDataAccessor with/for the given input data
	*/
	template<typename AttribCont, typename IndexCont>
	MeshDataAccessor buildMeshDataAccessor(
		std::vector<VertexBufferAccessor>& vb_accessors,
		AttribCont& attributes,
		GLsizei attributes_stride,
		IndexCont& index_data,
		GLenum index_type,
		GLenum mesh_usage,
		GLenum mesh_primtive_type)
	{
		MeshDataAccessor accessor;

		accessor.usage = mesh_usage;
		accessor.primitive_type = mesh_primtive_type;

		accessor.vertex_data.buffer_cnt = vb_accessors.size();
		accessor.vertex_data.buffers = vb_accessors.data();

		accessor.index_data.index_type = index_type;
		accessor.index_data.byte_size = index_data.size() * sizeof(IndexCont::value_type);
		accessor.index_data.raw_data = reinterpret_cast<uint8_t*>(index_data.data());

		accessor.vertex_descriptor.stride = attributes_stride;
		accessor.vertex_descriptor.attribute_cnt = attributes.size();
		accessor.vertex_descriptor.attributes = attributes.data();

		return accessor;
	}

}
}

#endif