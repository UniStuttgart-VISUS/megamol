#include "mesh/GPURenderTaskCollection.h"

namespace megamol {
namespace mesh {

void GPURenderTaskCollection::deleteSingleRenderTask(size_t rt_idx) {

    if (rt_idx > m_render_task_meta_data.size()) {
        vislib::sys::Log::DefaultLog.WriteError("RenderTask update error: Index out of bounds.");
        return;
    }

    RenderTaskMetaData rt_meta = m_render_task_meta_data[rt_idx];

    auto& rts = m_render_tasks[rt_meta.rts_idx];

    // create new draw command and per draw data buffers with (size - 1)
    size_t old_dcs_byte_size = rts.draw_commands->getByteSize();
    size_t old_pdd_byte_size = rts.per_draw_data->getByteSize();
    size_t new_dcs_byte_size = old_dcs_byte_size - sizeof(DrawElementsCommand);
    size_t new_pdd_byte_size = old_pdd_byte_size - rt_meta.per_draw_data_byteSize;

    auto new_dcs_buffer =
        std::make_shared<BufferObject>(GL_DRAW_INDIRECT_BUFFER, nullptr, new_dcs_byte_size, GL_DYNAMIC_DRAW);
    auto new_pdd_buffer =
        std::make_shared<BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, new_pdd_byte_size, GL_DYNAMIC_DRAW);

    // copy data from beg to delete
    BufferObject::copy(rts.draw_commands.get(), new_dcs_buffer.get(),0,0,rt_meta.draw_command_byteOffset);
    BufferObject::copy(rts.per_draw_data.get(), new_pdd_buffer.get(),0,0,rt_meta.per_draw_data_byteOffset);

    // copy data from delete to end
    BufferObject::copy(rts.draw_commands.get(), new_dcs_buffer.get(),
        rt_meta.draw_command_byteOffset + sizeof(DrawElementsCommand), rt_meta.draw_command_byteOffset,
        new_dcs_byte_size - rt_meta.draw_command_byteOffset);
    BufferObject::copy(rts.per_draw_data.get(), new_pdd_buffer.get(),
        rt_meta.per_draw_data_byteOffset + rt_meta.per_draw_data_byteSize, rt_meta.per_draw_data_byteOffset,
        new_pdd_byte_size - rt_meta.per_draw_data_byteOffset);

    rts.draw_commands = new_dcs_buffer;
    rts.per_draw_data = new_pdd_buffer;
    rts.draw_cnt -= 1;

    // Set rt meta entry to zero, keep it in list so as to not invalidate any saved indices into that vector
    m_render_task_meta_data[rt_idx].rts_idx = 0;
    m_render_task_meta_data[rt_idx].draw_command_byteOffset = 0;
    m_render_task_meta_data[rt_idx].per_draw_data_byteOffset = 0;
    m_render_task_meta_data[rt_idx].per_draw_data_byteSize = 0;
}

size_t GPURenderTaskCollection::getTotalDrawCount() {
    size_t retval = 0;
    for (auto& rt : m_render_tasks) {
        retval += rt.draw_cnt;
    }
    return retval;
};

std::vector<GPURenderTaskCollection::RenderTasks> const& GPURenderTaskCollection::getRenderTasks() {
    return m_render_tasks;
}

std::vector<std::pair<std::shared_ptr<BufferObject>, uint32_t>> const& GPURenderTaskCollection::getPerFrameBuffers() {
    return m_per_frame_data_buffers;
}

} // namespace mesh
} // namespace megamol