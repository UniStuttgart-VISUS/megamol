/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef WAVEFRONT_OBJ_RENDERER_H_INCLUDED
#define WAVEFRONT_OBJ_RENDERER_H_INCLUDED

#include "mesh_gl/BaseRenderTaskRenderer.h"

namespace megamol {
namespace mesh_gl {

    inline constexpr char wavefrontobjrenderer_name[] = "WavefrontObjRenderer";

    inline constexpr char wavefrontobjrenderer_desc[] =
        "Renderer for wavefront obj files using shaders for forward rendering.";

    class WavefrontObjRenderer : public BaseRenderTaskRenderer<wavefrontobjrenderer_name, wavefrontobjrenderer_desc> {
    public:
        using BaseRenderTaskRenderer<wavefrontobjrenderer_name, wavefrontobjrenderer_desc>::material_collection_;
        using BaseRenderTaskRenderer<wavefrontobjrenderer_name, wavefrontobjrenderer_desc>::mesh_collection_;
        using BaseRenderTaskRenderer<wavefrontobjrenderer_name, wavefrontobjrenderer_desc>::render_task_collection_;

        WavefrontObjRenderer();
        ~WavefrontObjRenderer();

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
        bool GetExtents(mmstd_gl::CallRender3DGL& call);

        void createMaterialCollection() override;

        bool updateMeshCollection() override;

        void updateRenderTaskCollection(bool force_update) override;

    private:
        /** Slot to retrieve the gltf model */
        megamol::core::CallerSlot lights_slot_;

        /** Slot to retrieve the mesh data of the gltf model */
        megamol::core::CallerSlot mesh_slot_;
    };

}
}

#endif // !WAVEFRONT_OBJ_RENDERER_H_INCLUDED
