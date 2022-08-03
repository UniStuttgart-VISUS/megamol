/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mesh_gl/BaseMeshRenderer.h"

namespace megamol::archvis_gl {

class MSMRenderer : public mesh_gl::BaseMeshRenderer {
public:
    MSMRenderer();
    ~MSMRenderer();

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "MSMRenderer";
    }
    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Renderer for MSM input data";
    }

protected:
    // Override extents query
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    void createMaterialCollection() override;
    void updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update) override;

private:
    megamol::core::CallerSlot m_MSM_callerSlot;
};

} // namespace megamol::archvis_gl
