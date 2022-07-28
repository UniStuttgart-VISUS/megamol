/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mesh_gl/BaseRenderTaskRenderer.h"

namespace megamol::archvis_gl {

inline constexpr char femrenderer_name[] = "FEMRenderer";
inline constexpr char femrenderer_desc[] =
    "Renderer for FEM (node, edge, box) data.";

class FEMRenderer : public mesh_gl::BaseRenderTaskRenderer<femrenderer_name, femrenderer_desc> {
public:
    FEMRenderer();
    ~FEMRenderer();

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
