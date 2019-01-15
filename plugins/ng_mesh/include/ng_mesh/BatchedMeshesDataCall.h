/*
* BatchedMeshesDataCall.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
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

		class NG_MESH_API BatchedMeshesDataCall : public megamol::core::AbstractGetData3DCall
		{
		public:

			struct MeshDataAccessor
			{
				struct VertexData
				{
					struct Buffer {
						std::byte*	raw_data_pointer;
						size_t		byte_size;
					};

					size_t		buffer_cnt;
					Buffer*		buffers;
				};

				struct IndexData
				{
					std::byte*	raw_data_pointer;
					size_t		byte_size;
					GLenum		index_type;
				};

				struct VertexLayoutData
				{
					GLsizei						stride;
					GLuint						attribute_cnt;
					VertexLayout::Attribute*	attributes_pointer;
				};

				VertexData			vertex_data;
				IndexData			index_data;
				VertexLayoutData	vertex_descriptor;
				GLenum				usage;
				GLenum				primitive_type;
			};

			struct DrawCommandDataAccessor
			{
				DrawElementsCommand*	draw_commands_pointer;
				GLsizei					draw_cnt;
			};

			class NG_MESH_API BatchedMeshesData
			{
			public:
				typedef MeshDataAccessor::VertexData::Buffer VertexBufferAccessor;

				size_t allocateNewBatch(size_t mesh_vertex_buffer_cnt);

				void setVertexData(size_t batch_idx,
					size_t mesh_vertex_buffer_cnt,
					VertexBufferAccessor...);

				void setIndexData(
					std::byte* raw_data_pointer,
					size_t byte_size,
					GLenum index_type);

				void setVertexLayout(
					GLsizei stride,
					GLuint attribute_cnt,
					VertexLayout::Attribute* attributes_pointer);

			private:
				DrawCommandDataAccessor* draw_command_batches;
				MeshDataAccessor*        mesh_data_batches;
				size_t                   batch_cnt;
			};


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

		private:

		};
	}
}


#endif // !BATCHED_MESHES_DATA_CALL_H_INCLUDED
