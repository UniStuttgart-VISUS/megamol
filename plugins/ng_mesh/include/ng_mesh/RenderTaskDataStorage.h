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
			RenderTaskDataStorage();
			~RenderTaskDataStorage();

			template<typename T>
			using IteratorPair = std::pair< T, T>;

			template<typename PerObjectDataIterator>
			void addRenderTask(
				size_t mesh_batch_idx,
				size_t draw_commands_base_offset,
				size_t draw_cnt,
				size_t material_idx,
				PerObjectDataIterator per_object_data);

		private:
			struct RenderTask
			{
				// reference to subset of mesh data used for this render task
				size_t mesh_batch_idx;
				size_t draw_commands_base_offset;
				size_t draw_cnt;

				// reference to material (i.e. shader) used for this render task
				size_t material_idx;

				// (all) per object data used (in the shader) for this render task
				std::vector<std::byte> per_object_shader_data;
			};

			std::vector<RenderTask> m_render_tasks; //use list?
		};
	}
}

#endif // !RENDER_TASK_DATA_STORAGE_H_INCLUDED
