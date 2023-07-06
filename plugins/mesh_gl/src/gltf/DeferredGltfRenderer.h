/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "BaseGltfRenderer.h"

namespace megamol::mesh_gl {

class DeferredGltfRenderer : public BaseGltfRenderer {
public:
    DeferredGltfRenderer() : BaseGltfRenderer() {}
    ~DeferredGltfRenderer() override = default;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "DeferredGltfRenderer";
    }
    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Renderer for gltf files using shaders for deferred rendering.";
    }

protected:
    void createMaterialCollection() override {
        material_collection_ = std::make_shared<GPUMaterialCollection>();
        material_collection_->addMaterial(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>(),
            "gltfMaterial",
            {"mesh_gl/gltf_example.vert.glsl",
                /*"mesh_gl/gltf_example_geom.glsl",*/ "mesh_gl/dfr_gltf_example.frag.glsl"});
        material_collection_->addMaterial(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>(),
            "gltfTexturedMaterial",
            {"mesh_gl/gltf_textured_example.vert.glsl",
                /*"mesh_gl/gltf_example_geom.glsl",*/ "mesh_gl/dfr_gltf_textured_example.frag.glsl"});
    }
};

} // namespace megamol::mesh_gl
