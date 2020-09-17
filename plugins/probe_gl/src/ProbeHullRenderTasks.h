/*
 * ProbeHullRenderTasks.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef PROBE_HULL_RENDER_TASK_H_INCLUDED
#define PROBE_HULL_RENDER_TASK_H_INCLUDED

#include "mesh/AbstractGPURenderTaskDataSource.h"

namespace megamol {
namespace probe_gl {

class ProbeHullRenderTasks : public megamol::mesh::AbstractGPURenderTaskDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ProbeHullRenderTasks"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Simple mesh viewer for the enclosing hull for probe placement.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    ProbeHullRenderTasks();
    ~ProbeHullRenderTasks();

protected:
    virtual bool getDataCallback(core::Call& caller);

    virtual bool getMetaDataCallback(core::Call& caller);

private:
    uint32_t m_version;

    bool m_show_hull;

    std::vector<std::vector<glowl::DrawElementsCommand>> m_draw_commands;
    std::vector<std::vector<std::array<float, 16>>> m_object_transforms;
    std::vector<std::shared_ptr<glowl::Mesh>> m_batch_meshes;

    core::CallerSlot m_probe_manipulation_slot;
};

} // namespace mesh
} // namespace megamol


#endif // !SIMPLE_PROBE_HULL_RENDER_TASK_H_INCLUDED
