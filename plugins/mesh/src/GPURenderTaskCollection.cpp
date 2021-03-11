#include "mesh/GPURenderTaskCollection.h"

namespace megamol {
namespace mesh {

    void GPURenderTaskCollection::copyGPURenderTask(
        std::string const& identifier, RenderTaskMetaData render_task_meta_data) {
        // TODO!!!
    }

    void GPURenderTaskCollection::deleteRenderTask(std::string const& identifier) {

        glGetError(); // TODO fuck this

        auto query = m_render_task_meta_data.find(identifier);

        if (query != m_render_task_meta_data.end()) {

            auto rt_meta = query->second;

            auto rts = rt_meta.render_tasks;

            rts->draw_cnt -= 1;

            if (rts->draw_cnt == 0) {
                for (size_t i = 0; i < m_gpu_render_tasks.size(); ++i) {
                    if (m_gpu_render_tasks[i] == rts) {
                        m_gpu_render_tasks.erase(m_gpu_render_tasks.begin() + i);
                        break;
                    }
                }
            } else {

                // create new draw command and per draw data buffers with (size - 1)
                size_t old_dcs_byte_size = rts->draw_commands->getByteSize();
                size_t old_pdd_byte_size = rts->per_draw_data->getByteSize();
                size_t new_dcs_byte_size = old_dcs_byte_size - sizeof(glowl::DrawElementsCommand);
                size_t new_pdd_byte_size = old_pdd_byte_size - rt_meta.per_draw_data_byteSize;

                try {
                    auto new_dcs_buffer = std::make_shared<glowl::BufferObject>(
                        GL_DRAW_INDIRECT_BUFFER, nullptr, new_dcs_byte_size, GL_DYNAMIC_DRAW);
                    auto new_pdd_buffer = std::make_shared<glowl::BufferObject>(
                        GL_SHADER_STORAGE_BUFFER, nullptr, new_pdd_byte_size, GL_DYNAMIC_DRAW);

                    // copy data from beg to delete
                    glowl::BufferObject::copy(
                        rts->draw_commands.get(), new_dcs_buffer.get(), 0, 0, rt_meta.draw_command_byteOffset);
                    glowl::BufferObject::copy(
                        rts->per_draw_data.get(), new_pdd_buffer.get(), 0, 0, rt_meta.per_draw_data_byteOffset);

                    // copy data from delete to end
                    glowl::BufferObject::copy(rts->draw_commands.get(), new_dcs_buffer.get(),
                        rt_meta.draw_command_byteOffset + sizeof(glowl::DrawElementsCommand),
                        rt_meta.draw_command_byteOffset, new_dcs_byte_size - rt_meta.draw_command_byteOffset);
                    glowl::BufferObject::copy(rts->per_draw_data.get(), new_pdd_buffer.get(),
                        rt_meta.per_draw_data_byteOffset + rt_meta.per_draw_data_byteSize,
                        rt_meta.per_draw_data_byteOffset, new_pdd_byte_size - rt_meta.per_draw_data_byteOffset);

                    rts->draw_commands = new_dcs_buffer;
                    rts->per_draw_data = new_pdd_buffer;

                    // adjust byte offsets of remaining render tasks
                    for (auto& rtm : m_render_task_meta_data) {
                        if (rtm.second.draw_command_byteOffset > rt_meta.draw_command_byteOffset) {
                            rtm.second.draw_command_byteOffset -= sizeof(glowl::DrawElementsCommand);
                        }
                        if (rtm.second.per_draw_data_byteOffset > rt_meta.per_draw_data_byteOffset) {
                            rtm.second.per_draw_data_byteOffset -= rt_meta.per_draw_data_byteSize;
                        }
                    }

                } catch (glowl::BufferObjectException const& e) {
                    megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
                        "Error during GPU render task deletion: %s\n", e.what());
                }
            }

            m_render_task_meta_data.erase(query);
        }
    }

    void GPURenderTaskCollection::deletePerFrameDataBuffer(uint32_t buffer_binding_point) {
        for (int i = 0; i < m_per_frame_data_buffers.size(); ++i) {
            if (m_per_frame_data_buffers[i].second == buffer_binding_point) {
                m_per_frame_data_buffers.erase(m_per_frame_data_buffers.begin() + i);
            }
        }
    }

    void GPURenderTaskCollection::clear() {
        m_gpu_render_tasks.clear();
        m_render_task_meta_data.clear();
        m_per_frame_data_buffers.clear();
    }

    size_t GPURenderTaskCollection::getTotalDrawCount() {
        size_t retval = 0;
        for (auto& rt : m_gpu_render_tasks) {
            retval += rt->draw_cnt;
        }
        return retval;
    };

    std::vector<std::shared_ptr<GPURenderTaskCollection::GPURenderTask>> const&
    GPURenderTaskCollection::getRenderTasks() {
        return m_gpu_render_tasks;
    }

    GPURenderTaskCollection::RenderTaskMetaData const& GPURenderTaskCollection::getRenderTaskMetaData(
        std::string const& identifier) {
        RenderTaskMetaData retval;

        auto query = m_render_task_meta_data.find(identifier);

        if (query != m_render_task_meta_data.end()) {
            retval = query->second;
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "getRenderTaskMetaData error: identifier not found.");
        }

        return retval;
    }

    std::vector<std::pair<std::shared_ptr<glowl::BufferObject>, uint32_t>> const&
    GPURenderTaskCollection::getPerFrameBuffers() {
        return m_per_frame_data_buffers;
    }

    void GPURenderTaskCollection::copyGPURenderTask(std::vector<GPURenderTask>& tgt_gpu_render_tasks,
        std::unordered_multimap<std::string, RenderTaskMetaData>& tgt_render_tas_meta_data,
        std::string const& identifier, RenderTaskMetaData const& src_render_task) {}

} // namespace mesh
} // namespace megamol
