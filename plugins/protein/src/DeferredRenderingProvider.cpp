/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "DeferredRenderingProvider.h"

#include "mmcore/utility/ShaderFactory.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/light/DistantLight.h"
#include "mmcore/view/light/PointLight.h"

using namespace megamol;
using namespace megamol::protein;

DeferredRenderingProvider::DeferredRenderingProvider(void)
        : fbo_(nullptr), lightingShader_(nullptr), pointLightBuffer_(nullptr), distantLightBuffer_(nullptr) {
    // TODO
}

DeferredRenderingProvider::~DeferredRenderingProvider(void) {
    // TODO
}

void DeferredRenderingProvider::setup(core::CoreInstance* coreIntstance) {
    try {
        auto const shdr_options = msf::ShaderFactoryOptionsOpenGL(coreIntstance->GetShaderPaths());

        lightingShader_ = core::utility::make_shared_glowl_shader("lighting", shdr_options,
            std::filesystem::path("deferred/lighting.vert.glsl"), std::filesystem::path("deferred/lighting.frag.glsl"));

    } catch (glowl::GLSLProgramException const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "[DeferredRenderingProvider] %s", ex.what());
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "[DeferredRenderingProvider] Unable to compile shader: Unknown exception: %s", ex.what());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "[DeferredRenderingProvider] Unable to compile shader: Unknown exception.");
    }

    fbo_ = std::make_shared<glowl::FramebufferObject>(1, 1);
    fbo_->createColorAttachment(GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT); // surface albedo
    fbo_->createColorAttachment(GL_RGB16F, GL_RGB, GL_HALF_FLOAT);   // normals
    fbo_->createColorAttachment(GL_R32F, GL_RED, GL_FLOAT);          // clip space depth

    pointLightBuffer_ = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    distantLightBuffer_ = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
}

void DeferredRenderingProvider::refreshLights(core::view::light::CallLight* lightCall, glm::vec3 camDir) {
    if (lightCall == nullptr || !(*lightCall)(core::view::light::CallLight::CallGetData)) {
        pointLights_.clear();
        distantLights_.clear();
        core::utility::log::Log::DefaultLog.WriteWarn(
            "[DeferredRenderingProvider]: There are no proper lights connected no shading is happening");
    } else {
        if (lightCall->hasUpdate()) {
            auto& lights = lightCall->getData();

            pointLights_.clear();
            distantLights_.clear();

            auto point_lights = lights.get<core::view::light::PointLightType>();
            auto distant_lights = lights.get<core::view::light::DistantLightType>();

            for (const auto& pl : point_lights) {
                pointLights_.push_back({pl.position[0], pl.position[1], pl.position[2], pl.intensity});
            }

            for (const auto& dl : distant_lights) {
                if (dl.eye_direction) {
                    auto cd = glm::normalize(camDir); // paranoia
                    distantLights_.push_back({cd.x, cd.y, cd.z, dl.intensity});
                } else {
                    distantLights_.push_back({dl.direction[0], dl.direction[1], dl.direction[2], dl.intensity});
                }
            }
        }
    }

    pointLightBuffer_->rebuffer(pointLights_);
    distantLightBuffer_->rebuffer(distantLights_);
}
