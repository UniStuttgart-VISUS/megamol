/*
 * ProbeDetailViewRenderTasks.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include "mesh_gl/AbstractGPURenderTaskDataSource.h"

#include "probe/ProbeCollection.h"

namespace megamol {
namespace probe_gl {

class ProbeDetailViewRenderTasks : public mesh_gl::AbstractGPURenderTaskDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ProbeDetailViewRenderTasks";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "...";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    bool create();

    void release();

    ProbeDetailViewRenderTasks();
    ~ProbeDetailViewRenderTasks();

protected:
    bool getDataCallback(core::Call& caller);

    bool getMetaDataCallback(core::Call& caller);

private:
    uint32_t m_version;

    core::CallerSlot m_transfer_function_Slot;

    core::CallerSlot m_probes_slot;

    core::CallerSlot m_event_slot;

    /** In-place material collection (initialized with probe detail view btf) */
    std::shared_ptr<mesh_gl::GPUMaterialCollection> m_material_collection;

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
