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

#include "glowl/BufferObject.h"
#include "glowl/Mesh.h"

#include "GPUMaterialCollection.h"

namespace megamol {
namespace mesh {

class MESH_API GPURenderTaskCollection {
public:
    // template<typename T>
    // using IteratorPair = std::pair< T, T>;

    struct RenderTasks {
        /**
         * Compare RenderTasks by shader program and mesh pointer addresses.
         * Note: The goal is to have RenderTasks using the same shader/mesh stored next to each other
         * to eventually reduce the amount of needed OpenGL state changes during rendering.
         */
        inline friend bool operator<(const RenderTasks& lhs, const RenderTasks& rhs) {
            return (lhs.shader_program == rhs.shader_program ? lhs.mesh < rhs.mesh
                                                             : lhs.shader_program < rhs.shader_program);
        }

        std::shared_ptr<Shader> shader_program;
        std::shared_ptr<Mesh> mesh;
        std::shared_ptr<BufferObject> draw_commands;
        std::shared_ptr<BufferObject> per_draw_data;

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
    size_t addSingleRenderTask(std::shared_ptr<Shader> const& shader_prgm, std::shared_ptr<Mesh> const& mesh,
        DrawElementsCommand const& draw_command,
        PerDrawDataType const& per_draw_data); // single struct of per draw data assumed?

    template <typename DrawCommandContainer, typename PerDrawDataContainer>
    size_t addRenderTasks(std::shared_ptr<Shader> const& shader_prgm, std::shared_ptr<Mesh> const& mesh,
        DrawCommandContainer const& draw_commands,
        PerDrawDataContainer const& per_draw_data); // list of per draw data assumed?

    void deleteSingleRenderTask(size_t rt_idx);

    template <typename PerDrawDataContainer>
    void updatePerDrawData(size_t rt_base_idx, PerDrawDataContainer const& per_draw_data);

    template <typename PerFrameDataContainer>
    void addPerFrameDataBuffer(PerFrameDataContainer const& per_frame_data, uint32_t buffer_binding_point);

    template <typename PerFrameDataContainer>
    void updatePerFrameDataBuffer(PerFrameDataContainer const& per_frame_data, uint32_t buffer_binding_point);

    // void updateGPUBuffers();

    void clear() {
        m_render_tasks.clear();
        m_per_frame_data_buffers.clear();
    }

    size_t getTotalDrawCount();

    std::vector<RenderTasks> const& getRenderTasks();

    std::vector<std::pair<std::shared_ptr<BufferObject>, uint32_t>> const& getPerFrameBuffers();

private:
    /**
     * Render tasks storage. Store tasks sorted by shader program and mesh.
     */
    std::vector<RenderTasks> m_render_tasks;

    /**
     * Store per render task meta data to identify GPU memory of individual tasks for updating.
     */
    std::vector<RenderTaskMetaData> m_render_task_meta_data;

    /**
     * Flexible number of OpenGL Buffers (SSBOs) for data shared by all render tasks,
     * e.g. scene meta data, lights or (dynamic) simulation data
     */
    std::vector<std::pair<std::shared_ptr<BufferObject>, uint32_t>> m_per_frame_data_buffers;
};

template <typename PerDrawDataType>
inline size_t GPURenderTaskCollection::addSingleRenderTask(std::shared_ptr<Shader> const& shader_prgm,
    std::shared_ptr<Mesh> const& mesh, DrawElementsCommand const& draw_command, PerDrawDataType const& per_draw_data) {
    bool task_added = false;

    size_t rts_idx = 0;

    size_t retval;

    // find matching RenderTasks set
    for (auto& rts : m_render_tasks) {
        if (rts.shader_program == shader_prgm && rts.mesh == mesh) {
            size_t old_dcs_byte_size = rts.draw_commands->getByteSize();
            size_t old_pdd_byte_size = rts.per_draw_data->getByteSize();
            size_t new_dcs_byte_size = old_dcs_byte_size + sizeof(DrawElementsCommand);
            size_t new_pdd_byte_size = old_pdd_byte_size + sizeof(PerDrawDataType);

            auto new_dcs_buffer =
                std::make_shared<BufferObject>(GL_DRAW_INDIRECT_BUFFER, nullptr, new_dcs_byte_size, GL_DYNAMIC_DRAW);
            auto new_pdd_buffer =
                std::make_shared<BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, new_pdd_byte_size, GL_DYNAMIC_DRAW);

            BufferObject::copy(rts.draw_commands.get(), new_dcs_buffer.get());
            BufferObject::copy(rts.per_draw_data.get(), new_pdd_buffer.get());

            new_dcs_buffer->loadSubData(&draw_command, sizeof(DrawElementsCommand), old_dcs_byte_size);
            new_pdd_buffer->loadSubData(&per_draw_data, sizeof(PerDrawDataType), old_pdd_byte_size);

            rts.draw_commands = new_dcs_buffer;
            rts.per_draw_data = new_pdd_buffer;
            rts.draw_cnt += 1;

            task_added = true;

            // Add render task meta data entry
            RenderTaskMetaData rt_meta;
            rt_meta.rts_idx = rts_idx;
            rt_meta.draw_command_byteOffset = old_dcs_byte_size;
            rt_meta.per_draw_data_byteOffset = old_pdd_byte_size;
            rt_meta.per_draw_data_byteSize = sizeof(PerDrawDataType);
            m_render_task_meta_data.push_back(rt_meta);

            retval = m_render_task_meta_data.size() - 1;
        }

        ++rts_idx;
    }

    // TODO add new RenderTasks if necessary and sort vector
    if (!task_added) {
        rts_idx = m_render_tasks.size();
        m_render_tasks.push_back(RenderTasks());

        RenderTasks& new_task = m_render_tasks.back();

        size_t new_dcs_byte_size = sizeof(DrawElementsCommand);
        size_t new_pdd_byte_size = sizeof(PerDrawDataType);

        new_task.shader_program = shader_prgm;
        new_task.mesh = mesh;
        new_task.draw_commands =
            std::make_shared<BufferObject>(GL_DRAW_INDIRECT_BUFFER, &draw_command, new_dcs_byte_size, GL_DYNAMIC_DRAW);
        new_task.per_draw_data = std::make_shared<BufferObject>(
            GL_SHADER_STORAGE_BUFFER, &per_draw_data, new_pdd_byte_size, GL_DYNAMIC_DRAW);
        new_task.draw_cnt = 1;

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
    std::shared_ptr<Mesh> const& mesh, DrawCommandContainer const& draw_commands,
    PerDrawDataContainer const& per_draw_data) {
    typedef typename PerDrawDataContainer::value_type PerDrawDataType;
    typedef typename DrawCommandContainer::value_type DrawCommandType;

    bool task_added = false;

    size_t rts_idx = 0;

    size_t retval;

    // find matching RenderTasks set
    for (auto& rts : m_render_tasks) {
        if (rts.shader_program == shader_prgm && rts.mesh == mesh) {
            size_t old_dcs_byte_size = rts.draw_commands->getByteSize();
            size_t old_pdd_byte_size = rts.per_draw_data->getByteSize();
            size_t new_dcs_byte_size = old_dcs_byte_size + sizeof(DrawCommandType) * draw_commands.size();
            size_t new_pdd_byte_size = old_pdd_byte_size + sizeof(PerDrawDataType) * per_draw_data.size();

            auto new_dcs_buffer =
                std::make_shared<BufferObject>(GL_DRAW_INDIRECT_BUFFER, nullptr, new_dcs_byte_size, GL_DYNAMIC_DRAW);
            auto new_pdd_buffer =
                std::make_shared<BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, new_pdd_byte_size, GL_DYNAMIC_DRAW);

            BufferObject::copy(rts.draw_commands.get(), new_dcs_buffer.get());
            BufferObject::copy(rts.per_draw_data.get(), new_pdd_buffer.get());

            new_dcs_buffer->loadSubData(
                draw_commands.data(), sizeof(DrawCommandType) * draw_commands.size(), old_dcs_byte_size);
            new_pdd_buffer->loadSubData(
                per_draw_data.data(), sizeof(PerDrawDataType) * per_draw_data.size(), old_pdd_byte_size);

            rts.draw_commands = new_dcs_buffer;
            rts.per_draw_data = new_pdd_buffer;
            rts.draw_cnt += draw_commands.size();

            task_added = true;

            retval = m_render_task_meta_data.size();

            for (int dc_idx = 0; dc_idx < draw_commands.size(); ++dc_idx) {
                // Add render task meta data entry
                RenderTaskMetaData rt_meta;
                rt_meta.rts_idx = rts_idx;
                rt_meta.draw_command_byteOffset = old_dcs_byte_size + dc_idx * sizeof(DrawCommandType);
                rt_meta.per_draw_data_byteOffset = old_pdd_byte_size + dc_idx * sizeof(PerDrawDataType);
                rt_meta.per_draw_data_byteSize = sizeof(PerDrawDataType);
                m_render_task_meta_data.push_back(rt_meta);
            }
        }

        ++rts_idx;
    }

    // TODO add new RenderTasks if necessary and sort vector
    if (!task_added) {
        rts_idx = m_render_tasks.size();

        m_render_tasks.push_back(RenderTasks());

        RenderTasks& new_task = m_render_tasks.back();

        size_t new_dcs_byte_size = sizeof(DrawCommandType) * draw_commands.size();
        size_t new_pdd_byte_size = sizeof(PerDrawDataType) * per_draw_data.size();

        new_task.shader_program = shader_prgm;
        new_task.mesh = mesh;
        new_task.draw_commands = std::make_shared<BufferObject>(
            GL_DRAW_INDIRECT_BUFFER, draw_commands.data(), new_dcs_byte_size, GL_DYNAMIC_DRAW);
        new_task.per_draw_data = std::make_shared<BufferObject>(
            GL_SHADER_STORAGE_BUFFER, per_draw_data.data(), new_pdd_byte_size, GL_DYNAMIC_DRAW);
        new_task.draw_cnt = draw_commands.size();

        retval = m_render_task_meta_data.size();

        for (int dc_idx = 0; dc_idx < draw_commands.size(); ++dc_idx) {
            // Add render task meta data entry
            RenderTaskMetaData rt_meta;
            rt_meta.rts_idx = rts_idx;
            rt_meta.draw_command_byteOffset = dc_idx * sizeof(DrawCommandType);
            rt_meta.per_draw_data_byteOffset = dc_idx * sizeof(PerDrawDataType);
            rt_meta.per_draw_data_byteSize = sizeof(PerDrawDataType);
            m_render_task_meta_data.push_back(rt_meta);
        }
    }

    return retval;
}

template <typename PerDrawDataContainer>
inline void GPURenderTaskCollection::updatePerDrawData(size_t rt_base_idx, PerDrawDataContainer const& per_draw_data) {
    if (rt_base_idx > m_render_task_meta_data.size()) {
        vislib::sys::Log::DefaultLog.WriteError("RenderTask update error: Index out of bounds.");
        return;
    }

    RenderTaskMetaData rt_meta = m_render_task_meta_data[rt_base_idx];

    auto& rts = m_render_tasks[rt_meta.rts_idx];

    rts.per_draw_data->loadSubData(per_draw_data, rt_meta.per_draw_data_byteOffset);
}

template <typename PerFrameDataContainer>
inline void GPURenderTaskCollection::addPerFrameDataBuffer(
    PerFrameDataContainer const& per_frame_data, uint32_t buffer_binding_point) {
    if (buffer_binding_point == 0) {
        // TODO Error, 0 already in use for per draw data
    } else {
        typedef typename PerFrameDataContainer::value_type PerFrameDataType;
        size_t pfd_byte_size = sizeof(PerFrameDataType) * per_frame_data.size();

        auto new_buffer = std::make_shared<BufferObject>(
            GL_SHADER_STORAGE_BUFFER, per_frame_data.data(), pfd_byte_size, GL_DYNAMIC_DRAW);

        m_per_frame_data_buffers.push_back(std::make_pair(new_buffer, buffer_binding_point));
    }
}

template <typename PerFrameDataContainer>
inline void GPURenderTaskCollection::updatePerFrameDataBuffer(
    PerFrameDataContainer const& per_frame_data, uint32_t buffer_binding_point) {
    typedef typename PerFrameDataContainer::value_type PerFrameDataType;

    for (auto& buffer : m_per_frame_data_buffers) {
        if (buffer_binding_point == std::get<1>(buffer)) {
            size_t pfd_byte_size = sizeof(PerFrameDataType) * per_frame_data.size();
            std::get<0>(buffer)->loadSubData(per_frame_data);
        }
    }
}

} // namespace mesh
} // namespace megamol

#endif // !GPU_RENDER_TASK_DATA_STORAGE_H_INCLUDED
