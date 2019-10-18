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
    core::CallerSlot m_probes_slot;
    size_t m_probes_cached_hash;
};

} // namespace mesh
} // namespace megamol


#endif // !PROBE_RENDER_TASK_H_INCLUDED
