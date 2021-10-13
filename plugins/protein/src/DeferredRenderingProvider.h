/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include "glowl/BufferObject.hpp"
#include "glowl/FramebufferObject.hpp"
#include "glowl/GLSLProgram.hpp"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/light/CallLight.h"

namespace megamol::protein {

class DeferredRenderingProvider {
public:
    DeferredRenderingProvider(void);
    virtual ~DeferredRenderingProvider(void);
    void setup(core::CoreInstance* coreInstance);
    void refreshLights(core::view::light::CallLight* lightCall, const glm::vec3 camDir);
    void draw(void);

private:
    struct LightParams {
        float x, y, z, intensity;
    };

    std::vector<LightParams> pointLights_;
    std::vector<LightParams> distantLights_;

    /** The framebuffer object */
    std::shared_ptr<glowl::FramebufferObject> fbo_;
    /** The shader for deferred lighting */
    std::shared_ptr<glowl::GLSLProgram> lightingShader_;

    /** SSBO containing point lights */
    std::unique_ptr<glowl::BufferObject> pointLightBuffer_;
    /** SSBO containing distant lights */
    std::unique_ptr<glowl::BufferObject> distantLightBuffer_;
};

} // namespace megamol::protein
