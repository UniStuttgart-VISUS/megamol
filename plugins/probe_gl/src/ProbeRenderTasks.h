/*
 * ProbeRenderTasks.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef PROBE_RENDER_TASK_H_INCLUDED
#define PROBE_RENDER_TASK_H_INCLUDED

#include "mesh/AbstractGPURenderTaskDataSource.h"

namespace megamol {
namespace probe_gl {

class ProbeRenderTasks : public mesh::AbstractGPURenderTaskDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ProbeRenderTasks"; }

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
    static bool IsAvailable(void) { return true; }

    ProbeRenderTasks();
    ~ProbeRenderTasks();

protected:
    virtual bool getDataCallback(core::Call& caller);

    virtual bool getMetaDataCallback(core::Call& caller);

private:
    struct PerProbeDrawData {
        glm::mat4x4 object_transform;
        int highlighted;
        float pad0;
        float pad1;
        float pad2;
    };

    uint32_t m_version;

    std::vector<PerProbeDrawData> m_probe_draw_data;

    core::CallerSlot m_probes_slot;

    core::CallerSlot m_probe_manipulation_slot;
};

} // namespace mesh
} // namespace megamol


#endif // !PROBE_RENDER_TASK_H_INCLUDED
