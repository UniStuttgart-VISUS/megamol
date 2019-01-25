/*
* BatchedMeshesDataCall.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef BATCHED_MESHES_DATA_CALL_H_INCLUDED
#define BATCHED_MESHES_DATA_CALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "ng_mesh.h"
#include "mmcore/AbstractGetData3DCall.h"

#include "NGMeshStructs.h"

namespace megamol {
	namespace ngmesh {

		/////////////////////////////////
		// Data Accessors for C API
		// (fearless warriors set out the cross the dll boundary)
		/////////////////////////////////

		struct MeshDataAccessor
		{
			size_t	                 vertex_buffers_accessors_base_index;
			size_t		             vertex_buffer_cnt;

			size_t                   index_buffer_accessor_index;
			GLenum		             index_type;

			GLsizei                  vertex_stride;
			GLuint                   vertex_attribute_cnt;
			VertexLayout::Attribute* vertex_attributes;

			GLenum                   usage;
			GLenum                   primitive_type;
		};

		struct DrawCommandsDataAccessor
		{
			DrawElementsCommand*	draw_commands;
			GLsizei					draw_cnt;
		};

		struct MeshIndexLookup
		{
			size_t batch_index;
			size_t draw_command_index;
		};

		class NG_MESH_API BatchedMeshesDataAccessor
		{
		public:
			BatchedMeshesDataAccessor();
			~BatchedMeshesDataAccessor();
			BatchedMeshesDataAccessor(const BatchedMeshesDataAccessor& cpy);
			BatchedMeshesDataAccessor(BatchedMeshesDataAccessor&& other);
			BatchedMeshesDataAccessor& operator=(BatchedMeshesDataAccessor&& rhs);
			BatchedMeshesDataAccessor& operator=(const BatchedMeshesDataAccessor& rhs);

			size_t allocateNewBatch(size_t mesh_vertex_buffer_cnt);

			//void setVertexDataAccess(
			//	size_t batch_idx,
			//	size_t vertex_buffers_cnt,
			//	BufferAccessor...);

			void setVertexDataAccessor(
				size_t batch_idx,
				size_t vertex_buffer_idx,
				BufferAccessor buffer_accessor);

			void setIndexDataAccess(
				size_t batch_idx,
				std::byte* raw_data,
				size_t byte_size,
				GLenum index_type);

			void setMeshMetaDataAccess(
				size_t batch_idx,
				GLsizei stride,
				GLuint attribute_cnt,
				VertexLayout::Attribute* attributes,
				GLenum usage,
				GLenum primitive_type);

			void setDrawCommandsDataAcess(
				size_t batch_idx,
				DrawElementsCommand* draw_commands,
				GLsizei draw_cnt);

			BufferAccessor*           buffer_accessors;
			size_t                    buffer_accessor_cnt;

			MeshDataAccessor*         mesh_data_batches;
			DrawCommandsDataAccessor* draw_command_batches;
			size_t                    batch_cnt;

			MeshIndexLookup*          mesh_index_lut;
			size_t                    mesh_cnt;
		};


		class NG_MESH_API BatchedMeshesDataCall : public megamol::core::AbstractGetData3DCall
		{
		public:


			BatchedMeshesDataCall();
			~BatchedMeshesDataCall();

			/**
			* Answer the name of the objects of this description.
			*
			* @return The name of the objects of this description.
			*/
			static const char *ClassName(void) {
				return "BatchedMeshesDataCall";
			}

			/**
			* Gets a human readable description of the module.
			*
			* @return A human readable description of the module.
			*/
			static const char *Description(void) {
				return "Call that gives access to mesh data stored in batches for rendering.";
			}

			/**
			* Answer the number of functions used for this call.
			*
			* @return The number of functions used for this call.
			*/
			static unsigned int FunctionCount(void) {
				return AbstractGetData3DCall::FunctionCount();
			}

			/**
			* Answer the name of the function used for this call.
			*
			* @param idx The index of the function to return it's name.
			*
			* @return The name of the requested function.
			*/
			static const char * FunctionName(unsigned int idx) {
				return AbstractGetData3DCall::FunctionName(idx);
			}

			void setBatchedMeshesDataAccessor(BatchedMeshesDataAccessor* data_accessor){
				m_data_accessor = data_accessor;
			}

			BatchedMeshesDataAccessor* getBatchedMeshesDataAccessor() {
				return m_data_accessor;
			}

			uint32_t getUpdateFlags() {
				return m_update_flags;
			}

			void resetUpdateFlags() {
				m_update_flags = 0;
			}

		private:
			BatchedMeshesDataAccessor* m_data_accessor;
			uint32_t                   m_update_flags;
		};

		/** Description class typedef */
		typedef megamol::core::factories::CallAutoDescription<BatchedMeshesDataCall> BatchedMeshesDataCallDescription;
	}
}


#endif // !BATCHED_MESHES_DATA_CALL_H_INCLUDED
