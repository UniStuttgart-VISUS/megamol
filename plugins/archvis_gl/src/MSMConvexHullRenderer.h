/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mesh_gl/BaseRenderTaskRenderer.h"
#include "mmcore/CallerSlot.h"

namespace megamol::archvis_gl {

inline constexpr char msmconvexhullrenderer_name[] = "MSMConvexHullRenderer";
inline constexpr char msmconvexhullrenderer_desc[] =
    "Renderer for generating and displaying convex hulls from MSM displacement values.";

class MSMConvexHullRenderer
        : public mesh_gl::BaseRenderTaskRenderer<msmconvexhullrenderer_name, msmconvexhullrenderer_desc> {
public:
    MSMConvexHullRenderer();
    ~MSMConvexHullRenderer();

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
    megamol::core::CallerSlot m_MSM_callerSlot;
};

} // namespace megamol::archvis_gl
