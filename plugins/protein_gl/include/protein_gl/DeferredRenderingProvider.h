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
#include "mmcore/param/ParamSlot.h"
#include "mmstd/light/CallLight.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"

namespace megamol::protein_gl {

class DeferredRenderingProvider {
public:
    DeferredRenderingProvider(void);
    virtual ~DeferredRenderingProvider(void);
    void setup(core::CoreInstance* coreInstance);
    void draw(mmstd_gl::CallRender3DGL& call, core::view::light::CallLight* lightCall, bool noShading = false);
    void setFramebufferExtents(uint32_t width, uint32_t height);
    void bindDeferredFramebufferToDraw(void);
    void resetToPreviousFramebuffer(void);
    std::vector<core::param::ParamSlot*> getUsedParamSlots(void);

    struct LightParams {
        float x, y, z, intensity;
    };

private:
    void refreshLights(core::view::light::CallLight* lightCall, const glm::vec3 camDir);

    std::vector<LightParams> pointLights_;
    std::vector<LightParams> distantLights_;

    GLint drawFBOid_;
    GLint readFBOid_;
    GLint FBOid_;

    /** The framebuffer object */
    std::shared_ptr<glowl::FramebufferObject> fbo_;
    /** The shader for deferred lighting */
    std::shared_ptr<glowl::GLSLProgram> lightingShader_;

    /** SSBO containing point lights */
    std::unique_ptr<glowl::BufferObject> pointLightBuffer_;
    /** SSBO containing distant lights */
    std::unique_ptr<glowl::BufferObject> distantLightBuffer_;

    core::param::ParamSlot ambientColorParam;
    core::param::ParamSlot diffuseColorParam;
    core::param::ParamSlot specularColorParam;
    core::param::ParamSlot ambientFactorParam;
    core::param::ParamSlot diffuseFactorParam;
    core::param::ParamSlot specularFactorParam;
    core::param::ParamSlot specularExponentParam;
    core::param::ParamSlot useLambertParam;
    core::param::ParamSlot enableShadingParam;
};

} // namespace megamol::protein_gl
