/*
 * mmvtkmMeshRenderTasks.h
 *
 * Copyright (C) 2020-2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMVTKM_MESH_RENDER_TASKS_H_INCLUDED
#define MMVTKM_MESH_RENDER_TASKS_H_INCLUDED

#include "mesh_gl/AbstractGPURenderTaskDataSource.h"
#include "mmcore/CallGeneric.h"

namespace megamol {
namespace mmvtkm_gl {

class mmvtkmMeshRenderTasks : public mesh_gl::AbstractGPURenderTaskDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "vtkmMeshRenderTasks";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Simple mesh viewer: Creates a single render task for each available GPU mesh.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    mmvtkmMeshRenderTasks();
    ~mmvtkmMeshRenderTasks();

protected:
    virtual bool getDataCallback(core::Call& caller);

private:
    uint32_t m_version;

    /** The slot for querying material data, i.e. a rhs connection */
    megamol::core::CallerSlot m_material_slot;
};

} // namespace mmvtkm_gl
} // namespace megamol


#endif // !MMVTKM_MESH_RENDER_TASKS_H_INCLUDED
