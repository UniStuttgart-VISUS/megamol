/*
 * MSMRenderTaskDataSource.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MSM_RENDER_TASK_DATA_SOURCE
#define MSM_RENDER_TASK_DATA_SOURCE
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mesh_gl/AbstractGPURenderTaskDataSource.h"

namespace megamol {
namespace archvis {

class MSMRenderTaskDataSource : public mesh_gl::AbstractGPURenderTaskDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "MSMRenderTaskDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Data source for creating RenderTasks for given MSM input data.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    MSMRenderTaskDataSource();
    ~MSMRenderTaskDataSource();

protected:
    virtual bool getDataCallback(core::Call& caller);

    virtual bool getMetaDataCallback(core::Call& caller) override;

private:
    megamol::core::CallerSlot m_MSM_callerSlot;

    /** In-place material collection (initialized with msm btf) */
    std::shared_ptr<mesh_gl::GPUMaterialCollection> m_material_collection;

    uint32_t m_version;
};

} // namespace archvis
} // namespace megamol

#endif // !MSM_RENDER_TASK_DATA_SOURCE
