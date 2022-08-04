/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include "glowl/BufferObject.hpp"
#include "glowl/FramebufferObject.hpp"
#include "glowl/GLSLProgram.hpp"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/light/CallLight.h"
#include "mmcore_gl/utility/SDFFont.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"

namespace megamol::molsurfmapcluster_gl {
class ProteinViewRenderer : public core_gl::view::Renderer3DModuleGL {
public:
    static const char* ClassName(void) {
        return "ProteinViewRenderer";
    }

    static const char* Description(void) {
        return "Renderer for tri-mesh data";
    }

    static bool IsAvailable(void) {
        return true;
    }

    ProteinViewRenderer(void);
    virtual ~ProteinViewRenderer(void);

protected:
    virtual bool create(void);
    virtual void release(void);
    virtual bool GetExtents(core_gl::view::CallRender3DGL& call);
    virtual bool Render(core_gl::view::CallRender3DGL& call);

private:
    void updateLights(core::view::light::CallLight* lightCall, glm::vec3 camDir);

    enum class WindingRule : int { CLOCK_WISE = 0, COUNTER_CLOCK_WISE = 1 };

    enum class RenderingMode : int { FILLED = 0, WIREFRAME = 1, POINTS = 2, NONE = 3 };

    struct LightParams {
        float x, y, z, intensity;
    };

    core::CallerSlot getDataSlot_;
    core::CallerSlot getLightsSlot_;
    core::CallerSlot get_texture_slot_;
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
    core::param::ParamSlot pdbid_param_;

    std::shared_ptr<glowl::GLSLProgram> meshShader_;
    std::shared_ptr<glowl::GLSLProgram> textureShader_;

    std::unique_ptr<glowl::BufferObject> positionBuffer_;
    std::unique_ptr<glowl::BufferObject> colorBuffer_;
    std::unique_ptr<glowl::BufferObject> normalBuffer_;
    std::unique_ptr<glowl::BufferObject> texcoordBuffer_;
    std::unique_ptr<glowl::BufferObject> indexBuffer_;
    std::unique_ptr<glowl::BufferObject> texture_buffer_;

    std::unique_ptr<glowl::BufferObject> pointLightBuffer_;
    std::unique_ptr<glowl::BufferObject> directionalLightBuffer_;

    std::shared_ptr<glowl::Texture2D> the_texture_;

    std::vector<LightParams> pointLights_;
    std::vector<LightParams> directionalLights_;

    core::utility::SDFFont font_;

    GLuint vertexArray_;
    GLuint tex_vertex_array_;

    size_t last_texture_hash_;
};
} // namespace megamol::molsurfmapcluster_gl
