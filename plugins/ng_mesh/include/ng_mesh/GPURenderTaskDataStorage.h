/*
* GPURenderTaskDataStorage.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef GPU_RENDER_TASK_DATA_STORAGE_H_INCLUDED
#define GPU_RENDER_TASK_DATA_STORAGE_H_INCLUDED

#include <memory>
#include <set>
#include <vector>

#include "glowl/Mesh.h"
#include "glowl/BufferObject.h"

#include "GPUMaterialDataStorage.h"

namespace megamol {
	namespace ngmesh {

		class GPURenderTaskDataStorage
		{
		public:
			template<typename T>
			using IteratorPair = std::pair< T, T>;

			struct RenderTasks
			{
				/**
				 * Compare RenderTasks by shader program and mesh pointer addresses.
				 * Note: The goal is to have RenderTasks using the same shader/mesh stored next to each other
				 * to eventually reduce the amount of needed OpenGL state changes during rendering.
				 */
				inline friend bool operator< (const RenderTasks& lhs, const RenderTasks& rhs) {
					return (lhs.shader_program == rhs.shader_program ? lhs.mesh < rhs.mesh : lhs.shader_program < rhs.shader_program);
				}

				std::shared_ptr<GLSLShader>   shader_program;
				std::shared_ptr<Mesh>         mesh;
				std::shared_ptr<BufferObject> draw_commands;
				std::shared_ptr<BufferObject> per_draw_data;

				size_t                        draw_cnt;
				size_t                        allocated_draw_cnt;

				size_t                        used_per_draw_data;
				size_t                        allocated_per_draw_data;
			};

			void reserveRenderTask(
				std::shared_ptr<GLSLShader> const& shader_prgm,
				std::shared_ptr<Mesh> const&       mesh,
				size_t                             draw_cnt,
				size_t                             per_draw_data_byte_size
			);

			template<typename PerTaskDataIterator>
			void addSingleRenderTask(
				std::shared_ptr<GLSLShader> const& shader_prgm,
				std::shared_ptr<Mesh> const&       mesh,
				DrawElementsCommand const&         draw_command,
				IteratorPair<PerTaskDataIterator>  per_draw_data);

			template<typename DrawCommandIterator, typename PerTaskDataIterator>
			void addRenderTasks(
				std::shared_ptr<GLSLShader> const& shader_prgm,
				std::shared_ptr<Mesh> const&       mesh,
				IteratorPair<DrawCommandIterator>  draw_commands,
				IteratorPair<PerTaskDataIterator>  per_draw_data);

			std::set<RenderTasks> const& getRenderTasks() { return m_render_tasks; }

		private:
			/**
			 * Render tasks storage. Store tasks sorted by shader program and mesh.
			 */
			std::set<RenderTasks> m_render_tasks;

		};

		template<typename PerTaskDataIterator>
		inline void GPURenderTaskDataStorage::addSingleRenderTask(
			std::shared_ptr<GLSLShader> const & shader_prgm,
			std::shared_ptr<Mesh> const & mesh,
			DrawElementsCommand const & draw_command,
			IteratorPair<PerTaskDataIterator> per_draw_data)
		{
		}

		template<typename DrawCommandIterator, typename PerTaskDataIterator>
		inline void GPURenderTaskDataStorage::addRenderTasks(
			std::shared_ptr<GLSLShader> const & shader_prgm,
			std::shared_ptr<Mesh> const & mesh,
			IteratorPair<DrawCommandIterator> draw_commands,
			IteratorPair<PerTaskDataIterator> per_draw_data)
		{
		}

	}
}

#endif // !GPU_RENDER_TASK_DATA_STORAGE_H_INCLUDED
