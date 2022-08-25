/*
 * ProbeDetailViewRenderTasks.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef PROBE_DETAIL_VIEW_RENDER_TASK_H_INCLUDED
#define PROBE_DETAIL_VIEW_RENDER_TASK_H_INCLUDED

#include "mesh_gl/BaseRenderTaskRenderer.h"

#include "probe/ProbeCollection.h"

namespace megamol {
namespace probe_gl {

class ProbeDetailViewRenderer : public mesh_gl::BaseRenderTaskRenderer {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ProbeDetailViewRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "...";
    }

    ProbeDetailViewRenderer();
    ~ProbeDetailViewRenderer();

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
    uint32_t m_version;

    core::CallerSlot m_transfer_function_Slot;

    core::CallerSlot m_probes_slot;

    core::CallerSlot m_event_slot;

    std::shared_ptr<glowl::Mesh> m_ui_mesh; // for depth scale (parallel to probe, offset by cam right vector)

    std::shared_ptr<glowl::Mesh> m_probes_mesh;

    std::shared_ptr<glowl::Texture2D> m_transfer_function;

    float m_tf_min;

    float m_tf_max;

    /**
     *
     */

    struct PerFrameData {
        int padding0;
        int padding1;
        int padding2;
        int padding3;
    };

    struct VectorProbeData {
        glm::vec4 position;
        glm::vec4 probe_direction;
        float scale;

        int probe_id;
        int state;

        float sample_cnt;
        std::array<float, 4> samples[32];

        GLuint64 tf_texture_handle;
        float tf_min;
        float tf_max;
    };

    std::vector<VectorProbeData> m_vector_probe_data;

    std::vector<glowl::DrawElementsCommand> m_vector_probe_draw_commands;

    std::vector<std::string> m_vector_probe_identifiers;

    std::vector<bool> m_vector_probe_selected;
};


} // namespace probe_gl
} // namespace megamol


#endif // !PROBE_BILLBOARD_GLYPH_RENDER_TASK_H_INCLUDED
