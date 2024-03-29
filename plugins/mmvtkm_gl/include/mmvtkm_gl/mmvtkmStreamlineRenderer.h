/*
 * mmvtkmStreamlineRenderer.h
 *
 * Copyright (C) 2020-2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include "mmstd/generic/CallGeneric.h"

#include "mesh_gl/BaseMeshRenderer.h"

namespace megamol::mmvtkm_gl {

class mmvtkmStreamlineRenderer : public mesh_gl::BaseMeshRenderer {
public:
    mmvtkmStreamlineRenderer() = default;
    ~mmvtkmStreamlineRenderer() override = default;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "mmvtkmStreamlineRenderer";
    }
    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Renderer for streamline meshes generated by vtkm.";
    }

protected:
    void createMaterialCollection() override;
    void updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update) override;
};

} // namespace megamol::mmvtkm_gl
