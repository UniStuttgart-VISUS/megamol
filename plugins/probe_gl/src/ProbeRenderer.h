/*
 * ProbeRenderTasks.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef PROBE_RENDER_TASK_H_INCLUDED
#define PROBE_RENDER_TASK_H_INCLUDED

#include "mesh_gl/BaseMeshRenderer.h"

namespace megamol {
namespace probe_gl {

class ProbeRenderTasks : public mesh_gl::BaseMeshRenderer {
public:
    ProbeRenderTasks();
    ~ProbeRenderTasks();

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ProbeRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "...";
    }

protected:
    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    void createMaterialCollection() override;
    void updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update) override;

private:
    struct PerProbeDrawData {
        glm::mat4x4 object_transform;
        int highlighted;
        float pad0;
        float pad1;
        float pad2;
    };

    std::vector<std::string> m_identifiers;

    std::vector<glowl::DrawElementsCommand> m_draw_commands;

    std::vector<PerProbeDrawData> m_probe_draw_data;

    bool m_show_probes;

    core::CallerSlot m_probes_slot;

    core::CallerSlot m_event_slot;
};

} // namespace probe_gl
} // namespace megamol


#endif // !PROBE_RENDER_TASK_H_INCLUDED
