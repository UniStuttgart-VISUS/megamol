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

    inline constexpr char wavefrontobjrenderer_name[] = "WavefrontObjRenderer";

    inline constexpr char wavefrontobjrenderer_desc[] =
        "Renderer for wavefront obj files using shaders for forward rendering.";

    class WavefrontObjRenderer : public BaseMeshRenderer<wavefrontobjrenderer_name, wavefrontobjrenderer_desc> {
    public:
        using BaseRenderTaskRenderer<wavefrontobjrenderer_name, wavefrontobjrenderer_desc>::material_collection_;
        using BaseRenderTaskRenderer<wavefrontobjrenderer_name, wavefrontobjrenderer_desc>::mesh_collection_;
        using BaseRenderTaskRenderer<wavefrontobjrenderer_name, wavefrontobjrenderer_desc>::render_task_collection_;

        WavefrontObjRenderer();
        ~WavefrontObjRenderer();

    protected:
        void createMaterialCollection() override;
        void updateRenderTaskCollection(bool force_update) override;

    private:
        /** Slot to retrieve the gltf model */
        megamol::core::CallerSlot lights_slot_;
    };

}
}

#endif // !WAVEFRONT_OBJ_RENDERER_H_INCLUDED
