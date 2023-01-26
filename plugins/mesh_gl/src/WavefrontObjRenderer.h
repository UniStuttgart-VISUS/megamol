/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef WAVEFRONT_OBJ_RENDERER_H_INCLUDED
#define WAVEFRONT_OBJ_RENDERER_H_INCLUDED

#include "mesh_gl/BaseMeshRenderer.h"

namespace megamol {
namespace mesh_gl {

class WavefrontObjRenderer : public BaseMeshRenderer {
public:
    WavefrontObjRenderer();
    ~WavefrontObjRenderer() override;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "WavefrontObjRenderer";
    }
    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Renderer for wavefront obj files using shaders for forward rendering.";
    }

protected:
    void createMaterialCollection() override;
    void updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update) override;

private:
    /** Slot to retrieve the gltf model */
    megamol::core::CallerSlot lights_slot_;
};

} // namespace mesh_gl
} // namespace megamol

#endif // !WAVEFRONT_OBJ_RENDERER_H_INCLUDED
