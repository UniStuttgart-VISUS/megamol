/*
* GPURenderTaskCollection.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef GPU_RENDER_TASK_DATA_STORAGE_H_INCLUDED
#define GPU_RENDER_TASK_DATA_STORAGE_H_INCLUDED

#include <memory>
#include <set>
#include <vector>

#include "ng_mesh.h"

#include "glowl/BufferObject.h"
#include "glowl/Mesh.h"

#include "GPUMaterialCollection.h"

namespace megamol {
	namespace ngmesh {

		class NG_MESH_API GPURenderTaskCollection
		{
		public:
			//template<typename T>
			//using IteratorPair = std::pair< T, T>;

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

				std::shared_ptr<Shader>       shader_program;
				std::shared_ptr<Mesh>         mesh;
				std::shared_ptr<BufferObject> draw_commands;
				std::shared_ptr<BufferObject> per_draw_data;

				size_t                        draw_cnt;
			};

			//void reserveRenderTask(
			//	std::shared_ptr<GLSLShader> const& shader_prgm,
			//	std::shared_ptr<Mesh> const&       mesh,
			//	size_t                             draw_cnt,
			//	size_t                             per_draw_data_byte_size
			//);

			template<typename PerDrawDataContainer>
			void addSingleRenderTask(
				std::shared_ptr<Shader> const& shader_prgm,
				std::shared_ptr<Mesh> const&   mesh,
				DrawElementsCommand const&     draw_command,
				PerDrawDataContainer const&    per_draw_data);

			template<typename DrawCommandContainer, typename PerDrawDataContainer>
			void addRenderTasks(
				std::shared_ptr<Shader> const& shader_prgm,
				std::shared_ptr<Mesh> const&   mesh,
				DrawCommandContainer const&    draw_commands,
				PerDrawDataContainer const&    per_draw_data);

			template<typename PerFrameDataContainer>
            void addPerFrameDataBuffer(PerFrameDataContainer const& per_frame_data, uint32_t buffer_binding_point);

			template <typename PerFrameDataContainer>
            void updatePerFrameDataBuffer(PerFrameDataContainer const& per_frame_data, uint32_t buffer_binding_point);

			//void updateGPUBuffers();

			void clear() {
				m_render_tasks.clear();
                m_per_frame_data_buffers.clear();
			}

			size_t getTotalDrawCount() { size_t retval = 0; for (auto& rt : m_render_tasks) { retval += rt.draw_cnt; } return retval; };

			std::vector<RenderTasks> const& getRenderTasks() { return m_render_tasks; }

			std::vector<std::pair<std::shared_ptr<BufferObject>, uint32_t>> const& getPerFrameBuffers() { return m_per_frame_data_buffers; }

		private:
			/**
			 * Render tasks storage. Store tasks sorted by shader program and mesh.
			 */
			std::vector<RenderTasks> m_render_tasks;

			/**
			 * Flexible number of OpenGL Buffers (SSBOs) for data shared by all render tasks,
			 * e.g. scene meta data, lights or (dynamic) simulation data
			 */
			std::vector<std::pair<std::shared_ptr<BufferObject>,uint32_t>> m_per_frame_data_buffers;
		};

		template<typename PerDrawDataContainer>
		inline void GPURenderTaskCollection::addSingleRenderTask(
			std::shared_ptr<Shader> const & shader_prgm,
			std::shared_ptr<Mesh> const & mesh,
			DrawElementsCommand const & draw_command,
			PerDrawDataContainer const& per_draw_data)
		{
			typedef typename PerDrawDataContainer::value_type PerDrawDataType;

			bool task_added = false;

			// find matching RenderTasks set
			for (auto& rts : m_render_tasks)
			{
				if (rts.shader_program == shader_prgm && rts.mesh == mesh)
				{
					size_t old_dcs_byte_size = rts.draw_commands->getByteSize();
					size_t old_pdd_byte_size = rts.per_draw_data->getByteSize();
					size_t new_dcs_byte_size = old_dcs_byte_size + sizeof(DrawElementsCommand);
					size_t new_pdd_byte_size = old_pdd_byte_size + sizeof(PerDrawDataType) * per_draw_data.size();

					auto new_dcs_buffer = std::make_shared<BufferObject>(GL_DRAW_INDIRECT_BUFFER, nullptr, new_dcs_byte_size, GL_DYNAMIC_DRAW);
					auto new_pdd_buffer = std::make_shared<BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, new_pdd_byte_size, GL_DYNAMIC_DRAW);

					BufferObject::copy(rts.draw_commands.get(), new_dcs_buffer.get());
					BufferObject::copy(rts.per_draw_data.get(), new_pdd_buffer.get());

					new_dcs_buffer->loadSubData(&draw_command, sizeof(DrawElementsCommand), old_dcs_byte_size);
					new_pdd_buffer->loadSubData(per_draw_data.data(), sizeof(PerDrawDataType) * per_draw_data.size(), old_pdd_byte_size);

					rts.draw_commands = new_dcs_buffer;
					rts.per_draw_data = new_pdd_buffer;
					rts.draw_cnt += 1;

					task_added = true;
				}
			}

			//TODO add new RenderTasks if necessary and sort vector
			if (!task_added)
			{
				m_render_tasks.push_back(RenderTasks());

				RenderTasks& new_task = m_render_tasks.back();

				size_t new_dcs_byte_size = sizeof(DrawElementsCommand);
				size_t new_pdd_byte_size = sizeof(PerDrawDataType) * per_draw_data.size();

				new_task.shader_program = shader_prgm;
				new_task.mesh = mesh;
				new_task.draw_commands = std::make_shared<BufferObject>(GL_DRAW_INDIRECT_BUFFER, &draw_command, new_dcs_byte_size, GL_DYNAMIC_DRAW);
				new_task.per_draw_data = std::make_shared<BufferObject>(GL_SHADER_STORAGE_BUFFER, per_draw_data.data(), new_pdd_byte_size, GL_DYNAMIC_DRAW);
				new_task.draw_cnt = 1;
			}
		}

		template<typename DrawCommandContainer, typename PerDrawDataContainer>
		inline void GPURenderTaskCollection::addRenderTasks(
			std::shared_ptr<Shader> const & shader_prgm,
			std::shared_ptr<Mesh> const & mesh,
			DrawCommandContainer const& draw_commands,
			PerDrawDataContainer const& per_draw_data)
		{
			typedef typename PerDrawDataContainer::value_type PerDrawDataType;
			typedef typename DrawCommandContainer::value_type DrawCommandType;

			bool task_added = false;

			// find matching RenderTasks set
			for (auto& rts : m_render_tasks)
			{
				if (rts.shader_program == shader_prgm && rts.mesh == mesh)
				{
					size_t old_dcs_byte_size = rts.draw_commands->getByteSize();
					size_t old_pdd_byte_size = rts.per_draw_data->getByteSize();
					size_t new_dcs_byte_size = old_dcs_byte_size + sizeof(DrawCommandType) * draw_commands.size();
					size_t new_pdd_byte_size = old_pdd_byte_size + sizeof(PerDrawDataType) * per_draw_data.size();

					auto new_dcs_buffer = std::make_shared<BufferObject>(GL_DRAW_INDIRECT_BUFFER, nullptr, new_dcs_byte_size, GL_DYNAMIC_DRAW);
					auto new_pdd_buffer = std::make_shared<BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, new_pdd_byte_size, GL_DYNAMIC_DRAW);

					BufferObject::copy(rts.draw_commands.get(), new_dcs_buffer.get());
					BufferObject::copy(rts.per_draw_data.get(), new_pdd_buffer.get());

					new_dcs_buffer->loadSubData(draw_commands.data(), sizeof(DrawCommandType) * draw_commands.size(), old_dcs_byte_size);
					new_pdd_buffer->loadSubData(per_draw_data.data(), sizeof(PerDrawDataType) * per_draw_data.size(), old_pdd_byte_size);

					rts.draw_commands = new_dcs_buffer;
					rts.per_draw_data = new_pdd_buffer;
					rts.draw_cnt += draw_commands.size();

					task_added = true;
				}
			}

			//TODO add new RenderTasks if necessary and sort vector
			if (!task_added)
			{
				m_render_tasks.push_back(RenderTasks());

				RenderTasks& new_task = m_render_tasks.back();

				size_t new_dcs_byte_size = sizeof(DrawCommandType) * draw_commands.size();
				size_t new_pdd_byte_size = sizeof(PerDrawDataType) * per_draw_data.size();

				new_task.shader_program = shader_prgm;
				new_task.mesh = mesh;
				new_task.draw_commands = std::make_shared<BufferObject>(GL_DRAW_INDIRECT_BUFFER, draw_commands.data(), new_dcs_byte_size, GL_DYNAMIC_DRAW);
				new_task.per_draw_data = std::make_shared<BufferObject>(GL_SHADER_STORAGE_BUFFER, per_draw_data.data(), new_pdd_byte_size, GL_DYNAMIC_DRAW);
				new_task.draw_cnt = draw_commands.size();
			}
        }

        template <typename PerFrameDataContainer>
        inline void GPURenderTaskCollection::addPerFrameDataBuffer(
            PerFrameDataContainer const& per_frame_data, uint32_t buffer_binding_point)
		{
			if (buffer_binding_point == 0)
			{
				//TODO Error, 0 already in use for per draw data
			}
			else
			{
                typedef typename PerFrameDataContainer::value_type PerFrameDataType;
                size_t pfd_byte_size = sizeof(PerFrameDataType) * per_frame_data.size();

                auto new_buffer = std::make_shared<BufferObject>(
                    GL_SHADER_STORAGE_BUFFER, per_frame_data.data(), pfd_byte_size, GL_DYNAMIC_DRAW);

                m_per_frame_data_buffers.push_back(std::make_pair(new_buffer, buffer_binding_point));
			}
		}

        template <typename PerFrameDataContainer>
        inline void GPURenderTaskCollection::updatePerFrameDataBuffer(
            PerFrameDataContainer const& per_frame_data, uint32_t buffer_binding_point)
		{
            typedef typename PerFrameDataContainer::value_type PerFrameDataType;

			for (auto& buffer : m_per_frame_data_buffers)
			{
				if (buffer_binding_point == std::get<1>(buffer))
				{   
                    size_t pfd_byte_size = sizeof(PerFrameDataType) * per_frame_data.size();
                    std::get<0>(buffer)->loadSubData(per_frame_data);
				}
			}
		}

	}
}

#endif // !GPU_RENDER_TASK_DATA_STORAGE_H_INCLUDED
