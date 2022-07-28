/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mesh_gl/BaseMeshRenderer.h"

namespace megamol::archvis_gl {

inline constexpr char msmrenderer_name[] = "MSMRenderer";
inline constexpr char msmrenderer_desc[] = "Renderer for MSM input data";

class MSMRenderer : public mesh_gl::BaseMeshRenderer<msmrenderer_name, msmrenderer_desc> {
public:
    MSMRenderer();
    ~MSMRenderer();

protected:
    // Override extents query
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    void createMaterialCollection() override;
    void updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update) override;

private:
    megamol::core::CallerSlot m_MSM_callerSlot;
};

} // namespace megamol::archvis_gl
