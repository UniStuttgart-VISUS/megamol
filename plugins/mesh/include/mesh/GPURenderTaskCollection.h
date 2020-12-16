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

#include "mesh.h"

#define GLOWL_OPENGL_INCLUDE_GLAD
#include "glowl/BufferObject.hpp"
#include "glowl/Mesh.hpp"

#include "GPUMaterialCollection.h"

namespace megamol {
namespace mesh {

    class MESH_API GPURenderTaskCollection {
    public:
        // template<typename T>
        // using IteratorPair = std::pair< T, T>;

        struct GLState {
            std::pair<std::vector<GLuint>, bool> capability;

            GLState(const std::pair<std::vector<GLuint>, bool>& c) : capability(c) {}
        };

        struct GPURenderTask {
            /**
             * Compare RenderTasks by shader program and mesh pointer addresses.
             * Note: The goal is to have RenderTasks using the same shader/mesh stored next to each other
             * to eventually reduce the amount of needed OpenGL state changes during rendering.
             */
            inline friend bool operator<(const GPURenderTask& lhs, const GPURenderTask& rhs) {
                return (lhs.shader_program == rhs.shader_program ? lhs.mesh < rhs.mesh
                                                                 : lhs.shader_program < rhs.shader_program);
            }

            std::shared_ptr<Shader> shader_program;
            std::shared_ptr<glowl::Mesh> mesh;
            std::shared_ptr<glowl::BufferObject> draw_commands;
            std::shared_ptr<glowl::BufferObject> per_draw_data;
            std::vector<GLState> states;

            size_t draw_cnt;
        };

        /**
         * Meta data describing an individual render task.
         */
        inline friend bool operator<(const RenderTasks& lhs, const RenderTasks& rhs) {
            return (lhs.shader_program == rhs.shader_program ? lhs.mesh < rhs.mesh
                                                             : lhs.shader_program < rhs.shader_program);
        }

        std::shared_ptr<Shader> shader_program;
        std::shared_ptr<glowl::Mesh> mesh;
        std::shared_ptr<glowl::BufferObject> draw_commands;
        std::shared_ptr<glowl::BufferObject> per_draw_data;

		std::function<void()> setStates;
		std::function<void()> resetStates;

        size_t draw_cnt;
    };

    /**
     * Meta data describing an individual render task.
     */
    struct RenderTaskMetaData {
        size_t rts_idx; // index of the render task bundle that contains this render task

        size_t draw_command_byteOffset;
        size_t per_draw_data_byteOffset;
        size_t per_draw_data_byteSize;
    };

    // void reserveRenderTask(
    //	std::shared_ptr<GLSLShader> const& shader_prgm,
    //	std::shared_ptr<Mesh> const&       mesh,
    //	size_t                             draw_cnt,
    //	size_t                             per_draw_data_byte_size
    //);

    template <typename PerDrawDataType>
    size_t addSingleRenderTask(std::shared_ptr<Shader> const& shader_prgm, std::shared_ptr<glowl::Mesh> const& mesh,
        glowl::DrawElementsCommand const& draw_command,
        PerDrawDataType const& per_draw_data, // single struct of per draw data assumed?
		std::function<void()> set = [] { // function to set opengl states
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
	}, 
		std::function<void()> reset = [] { // reset them to default
		glDisable(GL_DEPTH_TEST);
	});

    template <typename DrawCommandContainer, typename PerDrawDataContainer>
    size_t addRenderTasks(std::shared_ptr<Shader> const& shader_prgm, std::shared_ptr<glowl::Mesh> const& mesh,
        DrawCommandContainer const& draw_commands,
        PerDrawDataContainer const& per_draw_data, // list of per draw data assumed?
		std::function<void()> set = [] { // function to set opengl states
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
	},
		std::function<void()> reset = [] { // reset them to default
		glDisable(GL_DEPTH_TEST);
	});

        void copyGPURenderTask(std::string const& identifier, RenderTaskMetaData render_task_meta_data);

        void deleteRenderTask(std::string const& identifier);

        template<typename PerDrawDataContainer>
        void updatePerDrawData(std::string const& identifier, PerDrawDataContainer const& per_draw_data);

        template<typename PerFrameDataContainer>
        void addPerFrameDataBuffer(
            std::string const& identifier, PerFrameDataContainer const& per_frame_data, uint32_t buffer_binding_point);

        template<typename PerFrameDataContainer>
        void updatePerFrameDataBuffer(
            std::string const& identifier, PerFrameDataContainer const& per_frame_data, uint32_t buffer_binding_point);

        void deletePerFrameDataBuffer(uint32_t buffer_binding_point);

        void clear();

        size_t getTotalDrawCount();

        std::vector<std::shared_ptr<GPURenderTaskCollection::GPURenderTask>> const& getRenderTasks();

        GPURenderTaskCollection::RenderTaskMetaData const& getRenderTaskMetaData(std::string const& identifier);
template <typename PerDrawDataType>
inline size_t GPURenderTaskCollection::addSingleRenderTask(std::shared_ptr<Shader> const& shader_prgm,
    std::shared_ptr<glowl::Mesh> const& mesh, glowl::DrawElementsCommand const& draw_command,
    PerDrawDataType const& per_draw_data, std::function<void()> set, std::function<void()> reset) {
    bool task_added = false;

        std::vector<std::pair<std::shared_ptr<glowl::BufferObject>, uint32_t>> const& getPerFrameBuffers();

    private:
        void copyGPURenderTask(std::vector<GPURenderTask>& tgt_gpu_render_tasks,
            std::unordered_multimap<std::string, RenderTaskMetaData>& tgt_render_tas_meta_data,
            std::string const& identifier, RenderTaskMetaData const& src_render_task);

        /**
         * Render tasks storage. Store tasks sorted by shader program and mesh.
         */
        std::vector<std::shared_ptr<GPURenderTask>> m_gpu_render_tasks;
    // find matching RenderTasks set
    for (auto& rts : m_render_tasks) {
        if (rts.shader_program == shader_prgm && rts.mesh == mesh) {
            size_t old_dcs_byte_size = rts.draw_commands->getByteSize();
            size_t old_pdd_byte_size = rts.per_draw_data->getByteSize();
            size_t new_dcs_byte_size = old_dcs_byte_size + sizeof(glowl::DrawElementsCommand);
            size_t new_pdd_byte_size = old_pdd_byte_size + sizeof(PerDrawDataType);

            auto new_dcs_buffer = std::make_shared<glowl::BufferObject>(
                GL_DRAW_INDIRECT_BUFFER, nullptr, new_dcs_byte_size, GL_DYNAMIC_DRAW);
            auto new_pdd_buffer = std::make_shared<glowl::BufferObject>(
                GL_SHADER_STORAGE_BUFFER, nullptr, new_pdd_byte_size, GL_DYNAMIC_DRAW);

            glowl::BufferObject::copy(rts.draw_commands.get(), new_dcs_buffer.get());
            glowl::BufferObject::copy(rts.per_draw_data.get(), new_pdd_buffer.get());

            new_dcs_buffer->bufferSubData(&draw_command, sizeof(glowl::DrawElementsCommand), old_dcs_byte_size);
            new_pdd_buffer->bufferSubData(&per_draw_data, sizeof(PerDrawDataType), old_pdd_byte_size);

        /**
         * Store per render task meta data to identify GPU memory of individual tasks for updating.
         */
        // std::vector<RenderTaskMetaData> m_render_task_meta_data;

        std::unordered_map<std::string, RenderTaskMetaData> m_render_task_meta_data;

        /**
         * Flexible number of OpenGL Buffers (SSBOs) for data shared by all render tasks,
         * e.g. scene meta data, lights or (dynamic) simulation data
         */
        std::vector<std::pair<std::shared_ptr<glowl::BufferObject>, uint32_t>> m_per_frame_data_buffers;
    };

    template<typename PerDrawDataType>
    inline void GPURenderTaskCollection::addRenderTask(std::string const& identifier,
        std::shared_ptr<Shader> const& shader_prgm, std::shared_ptr<glowl::Mesh> const& mesh,
        glowl::DrawElementsCommand const& draw_command, PerDrawDataType const& per_draw_data,
        std::vector<GLState> const& states) {

        std::vector<std::string> identifiers = {identifier};
        std::vector<glowl::DrawElementsCommand> draw_command_vector = {draw_command};
        std::vector<PerDrawDataType> per_draw_data_vector = {per_draw_data};

        addRenderTasks(identifiers, shader_prgm, mesh, draw_command_vector, per_draw_data_vector, states);
    }

    template<typename IdentifierContainer, typename DrawCommandContainer, typename PerDrawDataContainer>
    inline void GPURenderTaskCollection::addRenderTasks(IdentifierContainer const& identifiers,
        std::shared_ptr<Shader> const& shader_prgm, std::shared_ptr<glowl::Mesh> const& mesh,
        DrawCommandContainer const& draw_commands, PerDrawDataContainer const& per_draw_data,
        std::vector<GLState> const& states) {
    // TODO add new RenderTasks if necessary and sort vector
    if (!task_added) {
        rts_idx = m_render_tasks.size();
        m_render_tasks.push_back(RenderTasks());

        RenderTasks& new_task = m_render_tasks.back();

        size_t new_dcs_byte_size = sizeof(glowl::DrawElementsCommand);
        size_t new_pdd_byte_size = sizeof(PerDrawDataType);

        new_task.shader_program = shader_prgm;
        new_task.mesh = mesh;
        new_task.draw_commands = std::make_shared<glowl::BufferObject>(
            GL_DRAW_INDIRECT_BUFFER, &draw_command, new_dcs_byte_size, GL_DYNAMIC_DRAW);
        new_task.per_draw_data = std::make_shared<glowl::BufferObject>(
            GL_SHADER_STORAGE_BUFFER, &per_draw_data, new_pdd_byte_size, GL_DYNAMIC_DRAW);
        new_task.draw_cnt = 1;
		new_task.setStates = set;
		new_task.resetStates = reset;

        // Add render task meta data entry
        RenderTaskMetaData rt_meta;
        rt_meta.rts_idx = rts_idx;
        rt_meta.draw_command_byteOffset = 0;
        rt_meta.per_draw_data_byteOffset = 0;
        rt_meta.per_draw_data_byteSize = sizeof(PerDrawDataType);
        m_render_task_meta_data.push_back(rt_meta);

        retval = m_render_task_meta_data.size() - 1;
    }

    return retval;
}

template <typename DrawCommandContainer, typename PerDrawDataContainer>
inline size_t GPURenderTaskCollection::addRenderTasks(std::shared_ptr<Shader> const& shader_prgm,
    std::shared_ptr<glowl::Mesh> const& mesh, DrawCommandContainer const& draw_commands,
    PerDrawDataContainer const& per_draw_data, std::function<void()> set, std::function<void()> reset) {
    typedef typename PerDrawDataContainer::value_type PerDrawDataType;
    typedef typename DrawCommandContainer::value_type DrawCommandType;

    bool task_added = false;

        typedef typename PerDrawDataContainer::value_type PerDrawDataType;
        typedef typename DrawCommandContainer::value_type DrawCommandType;

        auto query = std::find_if(m_gpu_render_tasks.begin(), m_gpu_render_tasks.end(),
            [&shader_prgm, &mesh](std::shared_ptr<GPURenderTask> const& rt) {
                return (rt->shader_program == shader_prgm && rt->mesh == mesh);
            });

        if (query != m_gpu_render_tasks.end()) {
            size_t old_dcs_byte_size = (*query)->draw_commands->getByteSize();
            size_t old_pdd_byte_size = (*query)->per_draw_data->getByteSize();
            size_t new_dcs_byte_size = old_dcs_byte_size + sizeof(DrawCommandType) * draw_commands.size();
            size_t new_pdd_byte_size = old_pdd_byte_size + sizeof(PerDrawDataType) * per_draw_data.size();

            auto new_dcs_buffer = std::make_shared<glowl::BufferObject>(
                GL_DRAW_INDIRECT_BUFFER, nullptr, new_dcs_byte_size, GL_DYNAMIC_DRAW);
            auto new_pdd_buffer = std::make_shared<glowl::BufferObject>(
                GL_SHADER_STORAGE_BUFFER, nullptr, new_pdd_byte_size, GL_DYNAMIC_DRAW);

            glowl::BufferObject::copy((*query)->draw_commands.get(), new_dcs_buffer.get());
            glowl::BufferObject::copy((*query)->per_draw_data.get(), new_pdd_buffer.get());

            new_dcs_buffer->bufferSubData(
                draw_commands.data(), sizeof(DrawCommandType) * draw_commands.size(), old_dcs_byte_size);
            new_pdd_buffer->bufferSubData(
                per_draw_data.data(), sizeof(PerDrawDataType) * per_draw_data.size(), old_pdd_byte_size);

            (*query)->draw_commands = new_dcs_buffer;
            (*query)->per_draw_data = new_pdd_buffer;
            (*query)->draw_cnt += draw_commands.size();
            
            assert(identifiers.size() == draw_commands.size());
            
            for (int dc_idx = 0; dc_idx < draw_commands.size(); ++dc_idx) {
                // Add render task meta data entry
                RenderTaskMetaData rt_meta;
                rt_meta.render_tasks = (*query);
                rt_meta.draw_command_byteOffset = old_dcs_byte_size + dc_idx * sizeof(DrawCommandType);
                rt_meta.per_draw_data_byteOffset = old_pdd_byte_size + dc_idx * sizeof(PerDrawDataType);
                rt_meta.per_draw_data_byteSize = sizeof(PerDrawDataType);
                
                auto rtn = m_render_task_meta_data.insert({identifiers[dc_idx], rt_meta});
                if (rtn.second == false) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "AddRenderTasks error: identifier already in use.");
                }
            }
        } else {
            m_gpu_render_tasks.push_back(std::make_shared<GPURenderTask>());

            auto new_task = m_gpu_render_tasks.back();

            size_t new_dcs_byte_size = sizeof(DrawCommandType) * draw_commands.size();
            size_t new_pdd_byte_size = sizeof(PerDrawDataType) * per_draw_data.size();

        new_task.shader_program = shader_prgm;
        new_task.mesh = mesh;
        new_task.draw_commands = std::make_shared<glowl::BufferObject>(
            GL_DRAW_INDIRECT_BUFFER, draw_commands.data(), new_dcs_byte_size, GL_DYNAMIC_DRAW);
        new_task.per_draw_data = std::make_shared<glowl::BufferObject>(
            GL_SHADER_STORAGE_BUFFER, per_draw_data.data(), new_pdd_byte_size, GL_DYNAMIC_DRAW);
        new_task.draw_cnt = draw_commands.size();
        new_task.setStates = set;
		new_task.resetStates = reset;

            assert(identifiers.size() == draw_commands.size());

            for (int dc_idx = 0; dc_idx < draw_commands.size(); ++dc_idx) {
                // Add render task meta data entry
                RenderTaskMetaData rt_meta;
                rt_meta.render_tasks = new_task;
                rt_meta.draw_command_byteOffset = dc_idx * sizeof(DrawCommandType);
                rt_meta.per_draw_data_byteOffset = dc_idx * sizeof(PerDrawDataType);
                rt_meta.per_draw_data_byteSize = sizeof(PerDrawDataType);
                
                auto rtn = m_render_task_meta_data.insert({identifiers[dc_idx], rt_meta});
                if (rtn.second == false) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "AddRenderTasks error: identifier already in use.");
                }
            }
        }
    }

    template<typename PerDrawDataContainer>
    inline void GPURenderTaskCollection::updatePerDrawData(
        std::string const& identifier, PerDrawDataContainer const& per_draw_data) {

        auto query = m_render_task_meta_data.find(identifier);

        if (query != m_render_task_meta_data.end()) {
            RenderTaskMetaData& rt_meta = query->second;
            rt_meta.render_tasks->per_draw_data->bufferSubData(per_draw_data, rt_meta.per_draw_data_byteOffset);
        }
    }

    template<typename PerFrameDataContainer>
    inline void GPURenderTaskCollection::addPerFrameDataBuffer(
        std::string const& identifier, PerFrameDataContainer const& per_frame_data, uint32_t buffer_binding_point) {
        if (buffer_binding_point == 0) {
            // TODO Error, 0 already in use for per draw data
        } else {
            typedef typename PerFrameDataContainer::value_type PerFrameDataType;
            size_t pfd_byte_size = sizeof(PerFrameDataType) * per_frame_data.size();

            auto new_buffer = std::make_shared<glowl::BufferObject>(
                GL_SHADER_STORAGE_BUFFER, per_frame_data.data(), pfd_byte_size, GL_DYNAMIC_DRAW);

            m_per_frame_data_buffers.push_back(std::make_pair(new_buffer, buffer_binding_point));
        }
    }

    template<typename PerFrameDataContainer>
    inline void GPURenderTaskCollection::updatePerFrameDataBuffer(
        std::string const& identifier, PerFrameDataContainer const& per_frame_data, uint32_t buffer_binding_point) {
        typedef typename PerFrameDataContainer::value_type PerFrameDataType;
        for (auto& buffer : m_per_frame_data_buffers) {
            if (buffer_binding_point == std::get<1>(buffer)) {
                size_t pfd_byte_size = sizeof(PerFrameDataType) * per_frame_data.size();
                std::get<0>(buffer)->bufferSubData(per_frame_data);
            }
        }
    }

} // namespace mesh
} // namespace megamol

#endif // !GPU_RENDER_TASK_DATA_STORAGE_H_INCLUDED
