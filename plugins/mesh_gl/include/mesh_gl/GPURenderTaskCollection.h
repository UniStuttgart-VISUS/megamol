/*
 * GPURenderTaskCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include <functional>
#include <memory>
#include <set>
#include <vector>

#include <glowl/BufferObject.hpp>
#include <glowl/Mesh.hpp>

#include "GPUMaterialCollection.h"

namespace megamol::mesh_gl {

class GPURenderTaskCollection;

namespace rendering {
/**
 * Utility function for processing gpu render task, i.e.
 * for rendering geometry with modern (OpenGL 4.3+) features.
 * Objects for rendering are supplied in batches. Each  render batch can contain
 * many objects that use the same shader program and also share the same geometry
 * or at least the same vertex format.
 * Per render batch, a single call of glMultiDrawElementsIndirect is made. The data
 * for the indirect draw call is stored and accessed via SSBOs.
 */
void processGPURenderTasks(
    std::shared_ptr<GPURenderTaskCollection> const& rt_collection, glm::mat4 const& view_mx, glm::mat4 const& proj_mx);
} // namespace rendering

class GPURenderTaskCollection {
public:
    // template<typename T>
    // using IteratorPair = std::pair< T, T>;

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

        std::function<void()> set_states;
        std::function<void()> reset_states;

        size_t draw_cnt;
    };

    /**
     * Meta data describing an individual render task.
     */
    struct RenderTaskMetaData {
        std::shared_ptr<GPURenderTask>
            render_tasks; // IDEA: store only this shared pointer to the actual GPU render task data s.t. it is
                          // automatically deleted if now longer used by any render task

        size_t draw_command_byteOffset;
        size_t per_draw_data_byteOffset;
        size_t per_draw_data_byteSize;
    };

    //void reserveRenderTask(
    //    std::shared_ptr<GLSLShader> const& shader_prgm,
    //    std::shared_ptr<Mesh> const&       mesh,
    //    size_t                             draw_cnt,
    //    size_t                             per_draw_data_byte_size
    //);

    template<typename PerDrawDataType>
    inline void addRenderTask(
        std::string const& identifier, std::shared_ptr<Shader> const& shader_prgm,
        std::shared_ptr<glowl::Mesh> const& mesh, glowl::DrawElementsCommand const& draw_command,
        PerDrawDataType const& per_draw_data,
        std::function<void()> set =
            [] { // function to set opengl states
                glEnable(GL_DEPTH_TEST);
                glDisable(GL_CULL_FACE);
            },
        std::function<void()> reset = [] { // reset them to default
            glDisable(GL_DEPTH_TEST);
        });

    template<typename IdentifierContainer, typename DrawCommandContainer, typename PerDrawDataContainer>
    inline void addRenderTasks(
        IdentifierContainer const& identifiers, std::shared_ptr<Shader> const& shader_prgm,
        std::shared_ptr<glowl::Mesh> const& mesh, DrawCommandContainer const& draw_commands,
        PerDrawDataContainer const& per_draw_data,
        std::function<void()> set =
            [] { // function to set opengl states
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

    // TODO identifier not actually used...
    template<typename PerFrameDataContainer>
    void addPerFrameDataBuffer(
        std::string const& identifier, PerFrameDataContainer const& per_frame_data, uint32_t buffer_binding_point);

    //TODO identifier not actually used...
    template<typename PerFrameDataContainer>
    void updatePerFrameDataBuffer(
        std::string const& identifier, PerFrameDataContainer const& per_frame_data, uint32_t buffer_binding_point);

    void deletePerFrameDataBuffer(uint32_t buffer_binding_point);

    void clear();

    size_t getTotalDrawCount();

    std::vector<std::shared_ptr<GPURenderTaskCollection::GPURenderTask>> const& getRenderTasks();

    GPURenderTaskCollection::RenderTaskMetaData const& getRenderTaskMetaData(std::string const& identifier);

    std::vector<std::pair<std::shared_ptr<glowl::BufferObject>, uint32_t>> const& getPerFrameBuffers();


private:
    void copyGPURenderTask(std::vector<GPURenderTask>& tgt_gpu_render_tasks,
        std::unordered_multimap<std::string, RenderTaskMetaData>& tgt_render_tas_meta_data,
        std::string const& identifier, RenderTaskMetaData const& src_render_task);

    /**
     * Render tasks storage. Store tasks sorted by shader program and mesh.
     */
    std::vector<std::shared_ptr<GPURenderTask>> m_gpu_render_tasks;

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
    glowl::DrawElementsCommand const& draw_command, PerDrawDataType const& per_draw_data, std::function<void()> set,
    std::function<void()> reset) {

    std::vector<std::string> identifiers = {identifier};
    std::vector<glowl::DrawElementsCommand> draw_command_vector = {draw_command};
    std::vector<PerDrawDataType> per_draw_data_vector = {per_draw_data};

    addRenderTasks(identifiers, shader_prgm, mesh, draw_command_vector, per_draw_data_vector, set, reset);
}

template<typename IdentifierContainer, typename DrawCommandContainer, typename PerDrawDataContainer>
inline void GPURenderTaskCollection::addRenderTasks(IdentifierContainer const& identifiers,
    std::shared_ptr<Shader> const& shader_prgm, std::shared_ptr<glowl::Mesh> const& mesh,
    DrawCommandContainer const& draw_commands, PerDrawDataContainer const& per_draw_data, std::function<void()> set,
    std::function<void()> reset) {

    typedef typename PerDrawDataContainer::value_type PerDrawDataType;
    typedef typename DrawCommandContainer::value_type DrawCommandType;

    auto query = std::find_if(m_gpu_render_tasks.begin(), m_gpu_render_tasks.end(),
        [&shader_prgm, &mesh](std::shared_ptr<GPURenderTask> const& rt) {
            return (rt->shader_program == shader_prgm && rt->mesh == mesh);
        });

    if (query != m_gpu_render_tasks.end()) {
        // maybe if found
        // m_rendertask_collection.first->deleteRenderTask(identifier);
        size_t old_dcs_byte_size = (*query)->draw_commands->getByteSize();
        size_t old_pdd_byte_size = (*query)->per_draw_data->getByteSize();
        size_t new_dcs_byte_size = old_dcs_byte_size + sizeof(DrawCommandType) * draw_commands.size();
        size_t new_pdd_byte_size = old_pdd_byte_size + sizeof(PerDrawDataType) * per_draw_data.size();

        auto new_dcs_buffer =
            std::make_shared<glowl::BufferObject>(GL_DRAW_INDIRECT_BUFFER, nullptr, new_dcs_byte_size, GL_DYNAMIC_DRAW);
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
            //std::cout << identifiers[dc_idx] << "\n";
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

        new_task->shader_program = shader_prgm;
        new_task->mesh = mesh;
        new_task->draw_commands = std::make_shared<glowl::BufferObject>(
            GL_DRAW_INDIRECT_BUFFER, draw_commands.data(), new_dcs_byte_size, GL_DYNAMIC_DRAW);
        new_task->per_draw_data = std::make_shared<glowl::BufferObject>(
            GL_SHADER_STORAGE_BUFFER, per_draw_data.data(), new_pdd_byte_size, GL_DYNAMIC_DRAW);
        new_task->draw_cnt = draw_commands.size();
        new_task->set_states = set;
        new_task->reset_states = reset;

        assert(identifiers.size() == draw_commands.size());

        for (int dc_idx = 0; dc_idx < draw_commands.size(); ++dc_idx) {
            // Add render task meta data entry
            //std::cout << identifiers[dc_idx] << "\n";
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

} // namespace megamol::mesh_gl
