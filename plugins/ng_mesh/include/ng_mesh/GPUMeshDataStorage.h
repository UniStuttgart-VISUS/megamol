/*
* GPUMeshDataStorage.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef GPU_MESH_DATA_STORAGE_H_INCLUDED
#define GPU_MESH_DATA_STORAGE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <memory>
#include <vector>

#include "glowl/Mesh.h"

namespace megamol {
	namespace ngmesh {

		class GPUMeshDataStorage
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
					bool                                                   store_seperate = false);

		private:
			struct BatchedMeshes
			{
				BatchedMeshes(
					unsigned int vertices_allocated,
					unsigned int indices_allocated)
					: mesh(nullptr),
					vertices_allocated(vertices_allocated),
					vertices_used(0),
					indices_allocated(indices_allocated),
					indices_used(0) {}

				std::shared_ptr<Mesh> mesh;
				unsigned int          vertices_allocated;
				unsigned int          vertices_used;
				unsigned int          indices_allocated;
				unsigned int          indices_used;
			};

			struct SubMeshData
			{
				size_t              batch_index;
				DrawElementsCommand sub_mesh_draw_command;
			};

			std::vector<BatchedMeshes> m_batched_meshes;
			std::vector<SubMeshData>   m_sub_mesh_data;
		};

		template<typename VertexBufferIterator, typename IndexBufferIterator>
		inline void GPUMeshDataStorage::addMesh(
			VertexLayout vertex_descriptor,
			std::vector<IteratorPair<VertexBufferIterator>> const & vertex_buffers,
			IteratorPair<IndexBufferIterator> index_buffer,
			GLenum index_type,
			GLenum usage,
			GLenum primitive_type,
			bool store_seperate)
		{
			typedef typename std::iterator_traits<IndexBufferIterator>::value_type IndexBufferType;
			typedef typename std::iterator_traits<VertexBufferIterator>::value_type VertexBufferType;

			// compute byte size of per vertex data in first vertex buffer
			std::vector<size_t> vb_attrib_byte_sizes;
			// single vertex buffer signals possible interleaved vertex layout, sum up all attribute byte sizes
			if (vertex_buffers.size() == 1) {
				vb_attrib_byte_sizes.push_back(0);
				for (auto& attr : vertex_descriptor.attributes) {
					vb_attrib_byte_sizes.back() += computeAttributeByteSize(attr);
				}
			}
			else {
				for (auto& attr : vertex_descriptor.attributes) {
					vb_attrib_byte_sizes.push_back(computeAttributeByteSize(attr));
				}
			}

			// get vertex buffer data pointers and byte sizes
			std::vector<GLvoid*> vb_data;
			std::vector<size_t> vb_byte_sizes;
			for (auto& vb : vertex_buffers){
				vb_data.push_back( reinterpret_cast<GLvoid*>( &(*std::get<0>(vb)) ) );
				vb_byte_sizes.push_back(sizeof(VertexBufferType) * std::distance(std::get<0>(vb), std::get<1>(vb)));
			}				
			// compute overall byte size of index buffer
			size_t ib_byte_size = sizeof(VertexBufferType) * std::distance(std::get<0>(index_buffer), std::get<1>(index_buffer));

			// computer number of requested vertices and indices
			size_t req_vertex_cnt = vb_byte_sizes.front() / vb_attrib_byte_sizes.front();
			size_t req_index_cnt = ib_byte_size / computeByteSize(index_type);

			auto it = m_batched_meshes.begin();
			if (!store_seperate)
			{
				// check for existing mesh batch with matching vertex layout and index type and enough available space
				for (; it != m_batched_meshes.end(); ++it)
				{
					bool layout_check = (vertex_descriptor == it->mesh->getVertexLayout());
					//TODO check interleaved vs non-interleaved
					bool idx_type_check = (index_type == it->mesh_data.index_type);

					if (layout_check && idx_type_check)
					{
						// check whether there is enough space left in batch
						size_t ava_vertex_cnt = (it->vertices_allocated - it->vertices_used);
						size_t ava_index_cnt = (it->indices_allocated - it->indices_used);

						if ((req_vertex_cnt < ava_vertex_cnt) && (req_index_cnt < ava_index_cnt))
						{
							break;
						}
					}
				}
			}
			else
			{
				it = m_batched_meshes.end();
			}

			// if either no batch was found or mesh is requested to be stored seperated, create a new GPU mesh batch
			if (it == m_batched_meshes.end())
			{
				size_t new_allocation_vertex_cnt = store_seperate ? req_vertex_cnt : std::max<size_t>(req_vert_cnt, 800000);
				size_t new_allocation_index_cnt = store_seperate ? req_index_cnt : std::max<size_t>(req_index_cnt, 3200000);

				m_batched_meshes.push_back(BatchedMeshes());

				std::vector<GLvoid*> alloc_data(vertex_buffers.size(),NULL);
				std::vector<size_t> alloc_vb_byte_sizes;
				for (size_t attrib_byte_size : vb_attrib_byte_sizes) {
					vb_byte_sizes.push_back(attrib_byte_size * req_vertex_cnt);
				}

				m_batched_meshes.back().mesh = std::make_shared<Mesh>(
					alloc_data,
					alloc_vb_byte_sizes,
					NULL,
					req_index_cnt * computeByteSize(index_type),
					vertex_descriptor,
					index_type,
					usage,
					primitive_type);

				m_batched_meshes.back().vertices_allocated = new_allocation_vertex_cnt;
				m_batched_meshes.back().vertices_used = 0;
				m_batched_meshes.back().indices_allocated = new_allocation_index_cnt;
				m_batched_meshes.back().indices_used = 0;
			}

			// upload data to GPU
			for (size_t i = 0; i < vb_data.size(); ++i)
			{
				// at this point, it should be guaranteed that it points at a mesh with matching vertex layout,
				// hence it's legal to multiply requested attrib byte sizes with vertex used count
				it->mesh->loadVertexSubData(i, vb_data[i], vb_byte_sizes[i], vb_attrib_byte_sizes[i] * it->vertices_used);
			}

			it->mesh->loadIndexSubData(reinterpret_cast<GLvoid*>(&*std::get<0>(index_buffer)), ib_byte_size, computeByteSize(index_type) * it->indices_used);
		}

	}
}

#endif // !GPU_MESH_DATA_STORAGE_H_INCLUDED
