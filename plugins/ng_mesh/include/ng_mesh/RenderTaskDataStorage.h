/*
* RenderTaskDataStorage.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef RENDER_TASK_DATA_STORAGE_H_INCLUDED
#define RENDER_TASK_DATA_STORAGE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <any>
#include <vector>

namespace megamol {
	namespace ngmesh
	{
		class RenderTaskDataStorage
		{
		public:
			template<typename T>
			using IteratorPair = std::pair< T, T>;

			template<typename PerObjectDataIterator>
			void addRenderTask(
				size_t mesh_batch_idx,
				size_t draw_commands_base_offset,
				size_t draw_cnt,
				size_t material_idx,
				IteratorPair<PerObjectDataIterator> per_object_data);

			inline void reserveBatch(
				size_t mesh_batch_idx,
				size_t material_idx,
				size_t task_cnt,
				size_t per_object_data_byte_size)
			{
				m_batched_render_task.push_back(BatchedRenderTasks());
				BatchedRenderTasks& new_batch = m_batched_render_task.back();

				new_batch.render_tasks.reserve(task_cnt);
				new_batch.total_draw_cnt = 0;

				new_batch.mesh_batch_idx = mesh_batch_idx;
				new_batch.material_idx = material_idx;

				new_batch.per_object_data = std::vector<std::byte>(per_object_data_byte_size);
				new_batch.per_object_data_allocated_bytes = per_object_data_byte_size;
				new_batch.per_object_data_used_bytes = 0;
			}

			struct RenderTask
			{
				size_t draw_commands_base_offset;
				size_t draw_commands_cnt;
			};

			struct BatchedRenderTasks
			{
				/** individual render task */
				std::vector<RenderTask> render_tasks;
				size_t total_draw_cnt;

				size_t mesh_batch_idx; //< reference to subset of mesh data used for this render task
				size_t material_idx; //< reference to material (i.e. shader) used for this render tas
				
				std::vector<std::byte> per_object_data; //< (all) per object data used (in the shader) for this batch of render tasks
				size_t per_object_data_allocated_bytes;
				size_t per_object_data_used_bytes;
			};

			std::vector<BatchedRenderTasks> m_batched_render_task;
		};

		template<typename PerObjectDataIterator>
		inline void RenderTaskDataStorage::addRenderTask(
			size_t mesh_batch_idx,
			size_t draw_commands_base_offset,
			size_t draw_cnt,
			size_t material_idx,
			IteratorPair<PerObjectDataIterator> per_object_data)
		{
			// check if render task batch with matching mesh batch index and material index already exists
			auto it = m_batched_render_task.begin();
			for (; it != m_batched_render_task.end(); ++it)
			{
				if( ( it->mesh_batch_idx == mesh_batch_idx)
					&& (it->material_idx == material_idx) )
				{
					break;
				}
			}

			if (it == m_batched_render_task.end())
			{
				m_batched_render_task.push_back(BatchedRenderTasks());
				it = m_batched_render_task.end();
				--it;

				it->total_draw_cnt = 0;
				it->per_object_data_allocated_bytes = 0;
				it->per_object_data_used_bytes = 0;
			}

			it->total_draw_cnt += draw_cnt;
			it->mesh_batch_idx = mesh_batch_idx;
			it->material_idx = material_idx;

			it->render_tasks.push_back(RenderTask());

			RenderTask& new_task = it->render_tasks.back();
			
			new_task.draw_commands_base_offset = draw_commands_base_offset;
			new_task.draw_commands_cnt = draw_cnt;
			
			size_t per_obj_data_byte_size = sizeof(std::iterator_traits<PerObjectDataIterator>::value_type) *
				std::distance(std::get<0>(per_object_data), std::get<1>(per_object_data));

			if ((per_obj_data_byte_size + it->per_object_data_used_bytes) > it->per_object_data_allocated_bytes)
			{
				size_t new_byte_size = per_obj_data_byte_size * 100 + it->per_object_data_allocated_bytes;
				std::vector<std::byte> new_per_obj_data(new_byte_size);

				// copy old data
				std::copy(it->per_object_data.begin(), it->per_object_data.end(), new_per_obj_data.begin());

				it->per_object_data_allocated_bytes += per_obj_data_byte_size * 100;

				it->per_object_data = std::move(new_per_obj_data);
			}

			// copy new data
			typedef typename std::iterator_traits<PerObjectDataIterator>::value_type PerObjectDataType;
			auto dest = reinterpret_cast<PerObjectDataType*>(it->per_object_data.data() + it->per_object_data_used_bytes);
			auto src_first = std::get<0>(per_object_data);
			auto src_last = std::get<1>(per_object_data);
			std::copy(src_first, src_last, dest);

			it->per_object_data_used_bytes += per_obj_data_byte_size;
		}
	}
}

#endif // !RENDER_TASK_DATA_STORAGE_H_INCLUDED
