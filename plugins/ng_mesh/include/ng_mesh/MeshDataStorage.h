/*
* MeshDataStorage.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef MESH_DATA_STORAGE_H_INCLUDED
#define MESH_DATA_STORAGE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <list>
#include <utility>
#include <vector>

#include "NGMeshStructs.h"
#include "BatchedMeshesDataCall.h"

namespace megamol {
	namespace ngmesh {

		class MeshDataStorage
		{
		public:

			template<typename T>
			using IteratorPair = std::pair< T, T>;

			template<
				typename VertexBufferIterator,
				typename IndexBufferIterator>
			void addMesh(
				VertexLayout                                           vertex_descriptor,
				std::vector<IteratorPair<VertexBufferIterator>> const& vertex_buffers,
				IteratorPair<IndexBufferIterator>                      index_buffer,
				GLenum                                                 index_type,
				GLenum                                                 usage,
				GLenum                                                 primitive_type,
				bool                                                   seperate = false);


			/**
			 * Resevere space for a mesh of the given size.
			 */
			size_t reserveMesh(
				VertexLayout vertex_descriptor,
				GLenum index_type,
				size_t vertex_cnt,
				size_t index_cnt);

			std::byte* accessVertexBufferData(size_t task_idx, size_t attribute_idx);

			std::byte* accessIndexBufferData(size_t task_idx);

			BatchedMeshesDataAccessor generateDataAccessor();

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

			struct MeshBatch
			{
				MeshData							mesh_data;
				std::vector<DrawElementsCommand>	draw_commands;
			};

			std::list<MeshBatch> m_mesh_batches;

		};

		template<
			typename VertexBufferIterator,
			typename IndexBufferIterator>
		inline void MeshDataStorage::addMesh(
			VertexLayout                                           vertex_descriptor,
			std::vector<IteratorPair<VertexBufferIterator>> const& vertex_buffers,
			IteratorPair<IndexBufferIterator>                      index_buffer,
			GLenum                                                 index_type,
			GLenum                                                 usage,
			GLenum                                                 primitive_type,
			bool                                                   seperate)
		{
			// compute the number of requested vertices and indices
			size_t req_vertex_byte_size = 0; // compute byte size of per vertex data in first vertex buffer
			if (vertex_buffers.size() == 1) {
				for (auto& attr : vertex_descriptor.attributes) {
					req_vertex_byte_size += computeAttributeByteSize(attr);
				}
			}
			else {
				req_vertex_byte_size = computeAttributeByteSize(vertex_descriptor.attributes.front());
			}

			size_t req_vb_byte_size = sizeof(std::iterator_traits<VertexBufferIterator>::value_type) *
				std::distance(std::get<0>(vertex_buffers.front()), std::get<1>(vertex_buffers.front()));
			size_t req_vert_cnt = req_vb_byte_size / req_vertex_byte_size;
			size_t req_index_byte_size = sizeof(std::iterator_traits<IndexBufferIterator>::value_type) * std::distance(std::get<0>(index_buffer), std::get<1>(index_buffer));
			size_t req_index_cnt = req_index_byte_size/computeByteSize(index_type);

			// check if a mesh batch with matching vertex layout and index type already exists
			auto it = m_mesh_batches.begin();
			for (; it != m_mesh_batches.end(); ++it)
			{
				size_t index_byte_size = it->mesh_data.index_type == 5123 ? 2 : (it->mesh_data.index_type == 5125 ? 4 : 0);

				bool layout_check = it->mesh_data.vertex_descriptor == vertex_descriptor;
				bool buffer_cnt_check = it->mesh_data.vertex_data.size() == vertex_buffers.size();
				bool idx_type_check = index_byte_size == sizeof(std::iterator_traits<IndexBufferIterator>::value_type);

				if (layout_check && buffer_cnt_check && idx_type_check)
				{
					// check whether there is enough space left in batch
					size_t ava_vert_cnt = (it->mesh_data.allocated_vertex_cnt - it->mesh_data.used_vertex_cnt);
					size_t ava_index_byte_size = (it->mesh_data.allocated_index_cnt - it->mesh_data.used_index_cnt) * index_byte_size;

					if ((req_vert_cnt < ava_vert_cnt) && (req_index_byte_size < ava_index_byte_size))
					{
						break;
					}
				}
			}

			// if no batch was found, create new one
			if (it == m_mesh_batches.end())
			{
				m_mesh_batches.push_back(MeshBatch());

				size_t new_allocation_vertex_cnt = std::max<size_t>(req_vert_cnt, 800000);
				size_t new_allocation_index_cnt = std::max<size_t>(req_index_cnt, 3200000);
				
				for (int i=0; i<vertex_buffers.size(); ++i)
				{
					size_t new_allocation_vb_byte_size = computeAttributeByteSize(vertex_descriptor.attributes[i]) * new_allocation_vertex_cnt;
					m_mesh_batches.back().mesh_data.vertex_data.push_back(std::vector<std::byte>(new_allocation_vb_byte_size));
				}
				
				size_t new_allocation_index_byte_size = new_allocation_index_cnt * computeByteSize(index_type);
				m_mesh_batches.back().mesh_data.index_data = std::vector<std::byte>(new_allocation_index_byte_size);

				m_mesh_batches.back().mesh_data.allocated_vertex_cnt = new_allocation_vertex_cnt;
				m_mesh_batches.back().mesh_data.used_vertex_cnt = 0;
				m_mesh_batches.back().mesh_data.allocated_index_cnt = new_allocation_index_cnt;
				m_mesh_batches.back().mesh_data.used_index_cnt = 0;

				m_mesh_batches.back().mesh_data.vertex_descriptor = vertex_descriptor;
				m_mesh_batches.back().mesh_data.index_type = index_type;
				m_mesh_batches.back().mesh_data.usage = usage;
				m_mesh_batches.back().mesh_data.primitive_type = primitive_type;

				it = m_mesh_batches.end();
				--it;
			}

			// copy mesh vertex data
			for (int i = 0; i < vertex_buffers.size(); ++i)
			{
				size_t vb_offset = computeAttributeByteSize(vertex_descriptor.attributes[i]) * it->mesh_data.used_vertex_cnt;
				auto dest = it->mesh_data.vertex_data[i].data();
				auto src_first = reinterpret_cast<std::byte*>(&*std::get<0>(vertex_buffers[i]));
				auto src_last = reinterpret_cast<std::byte*>(&*std::get<1>(vertex_buffers[i]));

				std::copy(src_first, src_last, dest);
			}

			// copy mesh index data
			size_t ib_offset = computeByteSize(index_type) * it->mesh_data.used_index_cnt;
			auto dest = it->mesh_data.index_data.data();
			auto src_first = reinterpret_cast<std::byte*>(&*std::get<0>(index_buffer));
			auto src_last = reinterpret_cast<std::byte*>(&*std::get<1>(index_buffer));
			std::copy(src_first, src_last, dest);

			// add draw command
			DrawElementsCommand new_draw_command;
			new_draw_command.cnt = req_index_cnt;
			new_draw_command.instance_cnt = 1;
			new_draw_command.first_idx = it->mesh_data.used_index_cnt;
			new_draw_command.base_vertex = it->mesh_data.used_vertex_cnt;
			new_draw_command.base_instance = 0;

			it->draw_commands.push_back(new_draw_command);

			// adjust used vertex/index count
			it->mesh_data.used_vertex_cnt += req_vert_cnt;
			it->mesh_data.used_index_cnt += req_index_cnt;
		}
	}
}

#endif // !MESH_DATA_STORAGE_H_INCLUDED

