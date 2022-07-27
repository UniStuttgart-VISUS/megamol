/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef DEFERRED_GLTF_RENDERER_H_INCLUDED
#define DEFERRED_GLTF_RENDERER_H_INCLUDED

#include "BaseGltfRenderer.h"

namespace megamol {
namespace mesh_gl {

inline constexpr char deferredgltfrenderer_name[] = "DeferredGltfRenderer";

inline constexpr char deferredgltfrenderer_desc[] = "Renderer for gltf files using shaders for deferred rendering.";

class DeferredGltfRenderer : public BaseGltfRenderer<deferredgltfrenderer_name, deferredgltfrenderer_desc> {
public:
    DeferredGltfRenderer() : BaseGltfRenderer() {}
    ~DeferredGltfRenderer() override = default;

protected:
    void createMaterialCollection() override {
        material_collection_ = std::make_shared<GPUMaterialCollection>();
        material_collection_->addMaterial(this->instance(), "gltfMaterial",
            {"mesh_gl/gltf_example.vert.glsl",
                /*"mesh_gl/gltf_example_geom.glsl",*/ "mesh_gl/dfr_gltf_example.frag.glsl"});
    }
};

} // namespace mesh_gl
} // namespace megamol

#endif // !DEFERRED_GLTF_RENDERER_H_INCLUDED
