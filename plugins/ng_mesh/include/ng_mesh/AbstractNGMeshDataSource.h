/*
* AbstractNGMeshDataSource.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef ABSTRACT_NG_MESH_DATASOURCE_H_INCLUDED
#define ABSTRACT_NG_MESH_DATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"

#include "ng_mesh/CallNGMeshRenderBatches.h"

namespace megamol {
namespace ngmesh {

	class NG_MESH_API AbstractNGMeshDataSource : public core::Module
	{
	public:
		AbstractNGMeshDataSource();

		virtual ~AbstractNGMeshDataSource();

	protected:

		/**
		* Class for storing and managing rendering data in a NGMesh-friendly fashion.
		* Either hand over your generated/loaded data (this will use a copy operation)
		* or preallocate the space you need and generate/load your data in-place.
		*/
		class RenderBatchesData
		{
		private:
			struct MeshData
			{
				std::vector<std::vector<std::byte>>	vertex_data;
				std::vector<std::byte>				index_data;

				size_t used_vertex_cnt;
				size_t used_index_cnt;
				size_t allocated_vertex_cnt;
				size_t allocated_index_cnt;

				VertexLayout	vertex_descriptor;
				GLenum			index_type;
				GLenum			usage;
				GLenum			primitive_type;
			};

			struct RenderBatch
			{
				std::string							program_name;
				MeshData							mesh_data;
				std::vector<DrawElementsCommand>	draw_commands;
				std::vector<std::byte>				perObject_shader_params;
				uint32_t							update_flags;
			};

			struct RenderTask // TODO Move to dedicated DataStructure to split Mesh/Material/RenderTask Data to different Datasources
			{
				size_t batch_idx;
				size_t draw_commands_offset;
				size_t perObject_params_byte_offset;
				size_t vertex_data_offset; // TODO remove, as already contained in DrawElementsCommand?
				size_t index_data_offset; // TODO remove, as already contained in DrawElementsCommand?
			};

			std::list<RenderBatch> m_render_batches;

			std::vector<RenderTask> m_render_tasks;

		public:
			template<
				typename VertexBuffersInterator,
				typename IndexBufferIterator,
				typename PerObjectShaderParams>
			void addRenderTask(
				std::string const& program_name,
				VertexLayout vertex_descriptor,
				VertexBuffersInterator vertex_buffers_begin, VertexBuffersInterator vertex_buffers_end,
				IndexBufferIterator index_buffer_begin, IndexBufferIterator index_buffer_end,
				PerObjectShaderParams per_object_params)
			{
				//TODO check if batch with same program and vertex layout already exits
				auto it = m_render_batches.begin()
				for (; it != m_render_batches.end() ++it)
				{
					size_t index_byte_size = it->mesh_data.index_type == 5123 ? 2 : (it->mesh_data.index_type == 5125 ? 4 : 0);
					
					bool prgm_check = it->program_name.compare(program_name) == 0;
					bool vert_layout_check = it->mesh_data.vertex_descriptor == vertex_descriptor;
					bool vert_buffer_cnt_check = it->mesh_data.vertex_data.size() == std::distance(vertex_buffers_begin, vertex_buffers_end);
					bool idx_type_check = index_byte_size == sizeof(std::iterator_traits<IndexBufferIterator>::value_type);

					if( prgm_check && vert_layout_check && vert_buffer_cnt_check && idx_type_check)
					{
						

						// check whether there is enough space left in batch
						//size_t req_vert_byte_size = sizeof(std::iterator_traits<VertexBufferInterator>::value_type) * ;
						size_t ava_vert_byte_size = (it->mesh_data.allocated_vertex_cnt - it->mesh_data.used_vertex_cnt) * it->mesh_data.vertex_descriptor.stride; // assume stride to equal byte size (i.e. tightly packed)

						size_t req_index_byte_size = sizeof(std::iterator_traits<IndexBufferIterator>::value_type) * std::distance(index_buffer_begin, index_buffer_end);
						size_t ava_index_byte_size = (it->mesh_data.allocated_index_cnt - it->mesh_data.used_index_cnt) * index_byte_size;

						if ((req_byte_size < ava_byte_size)&&(req_index_byte_size<ava_index_byte_size))
						{
							// TODO add render task to batch

							int batch_idx = std::distance(m_render_batches.begin(), it);
						}
					}
				}

				//TODO if no batch was found, create new one and add task
				if (it == m_render_batches.end())
				{

				}

			}

			template<typename PerObjectShaderParams>
			size_t reserveRenderTask(
				std::string const& program_name,
				VertexLayout vertex_descriptor,
				GLenum index_type,
				size_t vertex_cnt,
				size_t index_cnt,
				PerObjectShaderParams per_object_params)
			{

			}

			std::byte* accessVertexBufferData(size_t task_idx, size_t attribute_idx);

			std::byte* accessIndexBufferData(size_t task_idx);

			std::byte* accessPerObjectParamsData(size_t task_idx);

			void shrinkRenderTask(size_t vertex_cnt, size_t index_cnt);

			RenderBatchesDataAccessor generateAccessor()
			{

			}


		};

		/**
		* Implementation of 'Create'.
		*
		* @return 'true' on success, 'false' otherwise.
		*/
		virtual bool create(void);

		/**
		* Gets the data from the source.
		*
		* @param caller The calling call.
		*
		* @return 'true' on success, 'false' on failure.
		*/
		virtual bool getDataCallback(core::Call& caller) = 0;

		/**
		* Gets the data from the source.
		*
		* @param caller The calling call.
		*
		* @return 'true' on success, 'false' on failure.
		*/
		virtual bool getExtentCallback(core::Call& caller);

		/**
		* Implementation of 'Release'.
		*/
		virtual void release();

		/** The data storage for the render batches */
		CallNGMeshRenderBatches::RenderBatchesData m_render_batches;

		RenderBatchesData m_new_batches;

		/** The bounding box */
		vislib::math::Cuboid<float> m_bbox;

	private:

		/** The slot for requesting data */
		megamol::core::CalleeSlot m_getData_slot;
	};

}
}

#endif

