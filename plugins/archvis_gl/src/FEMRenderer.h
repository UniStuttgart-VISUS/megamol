/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mesh_gl/BaseRenderTaskRenderer.h"

namespace megamol::archvis_gl {

class FEMRenderer : public mesh_gl::BaseRenderTaskRenderer {
public:
    FEMRenderer();
    ~FEMRenderer();

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "FEMRenderer";
    }
    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Renderer for FEM (node, edge, box) data.";
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

    bool updateMeshCollection() override;
    void updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update) override;


private:
    megamol::core::CallerSlot m_fem_callerSlot;
};

} // namespace megamol::archvis_gl
