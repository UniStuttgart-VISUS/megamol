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

		private:
			struct RenderTask
			{
				// reference to subset of mesh data used for this render task
				size_t mesh_batch_idx;
				size_t draw_commands_base_offset;
				size_t draw_cnt;

				// reference to material (i.e. shader) used for this render task
				size_t material_idx;
			};

			struct BatchedRenderTasks
			{
				std::vector<RenderTask> m_render_tasks;

				// (all) per object data used (in the shader) for this batch of render tasks
				std::vector<std::byte> per_object_data;
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
				if( ( it->m_render_tasks.front().mesh_batch_idx == mesh_batch_idx)
					&& (it->m_render_tasks.front().material_idx == material_idx) )
				{
					break;
				}
			}

			if (it == m_batched_render_task.end())
			{
				m_render_tasks.push_back(BatchedRenderTasks());
				it = m_render_tasks.back();
				--it;
			}

			it->m_render_tasks.push_back(RenderTask);

			RenderTask& new_task = it->m_render_tasks.back();
			
			new_task.mesh_batch_idx = mesh_batch_idx;
			new_task.draw_commands_base_offset = draw_commands_base_offset;
			new_task.draw_cnt = draw_cnt;
			new_task.material_idx = material_idx;

			size_t per_obj_data_byte_size = sizeof(std::iterator_traits<PerObjectDataIterator>::value_type) *
				std::distance(std::get<0>(per_object_data), std::get<1>(per_object_data));

			size_t allocated_byte_size = it->per_object_data.size();
			size_t new_byte_size = per_obj_data_byte_size + allocated_byte_size;

			std::vector<byte> new_per_obj_data(new_byte_size);
			auto dest = new_per_obj_data.back().per_object_data.data();
			auto src_first = reinterpret_cast<std::byte*>(&*std::get<0>(per_object_data));
			auto src_last = reinterpret_cast<std::byte*>(&*std::get<1>(per_object_data));
			std::copy(src_first, src_last, dest);

			it->per_object_data = std::move(new_per_obj_data);
		}
	}
}

#endif // !RENDER_TASK_DATA_STORAGE_H_INCLUDED
