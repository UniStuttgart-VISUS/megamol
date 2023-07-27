/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include <glowl/BufferObject.hpp>
#include <glowl/FramebufferObject.hpp>
#include <glowl/GLSLProgram.hpp>

#include "mmcore/param/ParamSlot.h"
#include "mmstd/light/CallLight.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

namespace megamol::trisoup_gl {
class ModernTrisoupRenderer : public mmstd_gl::Renderer3DModuleGL {
public:
    static const char* ClassName() {
        return "ModernTrisoupRenderer";
    }

    static const char* Description() {
        return "Renderer for tri-mesh data";
    }

    static bool IsAvailable() {
        return true;
    }

    ModernTrisoupRenderer();
    ~ModernTrisoupRenderer() override;

protected:
    bool create() override;
    void release() override;
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;
    bool Render(mmstd_gl::CallRender3DGL& call) override;

private:
    void updateLights(core::view::light::CallLight* lightCall, glm::vec3 camDir);

    enum class WindingRule : int { CLOCK_WISE = 0, COUNTER_CLOCK_WISE = 1 };

    enum class RenderingMode : int { FILLED = 0, WIREFRAME = 1, POINTS = 2, NONE = 3 };

    struct LightParams {
        float x, y, z, intensity;
    };

    core::CallerSlot getDataSlot_;
    core::CallerSlot getLightsSlot_;
    core::CallerSlot getFramebufferSlot_;

    core::param::ParamSlot frontStyleParam_;
    core::param::ParamSlot backStyleParam_;
    core::param::ParamSlot windingRuleParam_;
    core::param::ParamSlot colorParam_;

    core::param::ParamSlot ambientColorParam_;
    core::param::ParamSlot diffuseColorParam_;
    core::param::ParamSlot specularColorParam_;
    core::param::ParamSlot ambientFactorParam_;
    core::param::ParamSlot diffuseFactorParam_;
    core::param::ParamSlot specularFactorParam_;
    core::param::ParamSlot specularExponentParam_;
    core::param::ParamSlot useLambertParam_;
    core::param::ParamSlot lightingParam_;

    std::shared_ptr<glowl::GLSLProgram> meshShader_;

    std::unique_ptr<glowl::BufferObject> positionBuffer_;
    std::unique_ptr<glowl::BufferObject> colorBuffer_;
    std::unique_ptr<glowl::BufferObject> normalBuffer_;
    std::unique_ptr<glowl::BufferObject> texcoordBuffer_;
    std::unique_ptr<glowl::BufferObject> indexBuffer_;

    std::unique_ptr<glowl::BufferObject> pointLightBuffer_;
    std::unique_ptr<glowl::BufferObject> directionalLightBuffer_;

    std::vector<LightParams> pointLights_;
    std::vector<LightParams> directionalLights_;

    GLuint vertexArray_;
};
} // namespace megamol::trisoup_gl
