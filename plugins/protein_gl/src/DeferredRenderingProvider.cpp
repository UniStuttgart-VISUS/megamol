/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "DeferredRenderingProvider.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/light/DistantLight.h"
#include "mmcore/view/light/PointLight.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmcore_gl/utility/ShaderSourceFactory.h"

using namespace megamol;
using namespace megamol::protein_gl;

DeferredRenderingProvider::DeferredRenderingProvider(void)
        : ambientColorParam("lighting::ambientColor", "Ambient color of the used lights")
        , diffuseColorParam("lighting::diffuseColor", "Diffuse color of the used lights")
        , specularColorParam("lighting::specularColor", "Specular color of the used lights")
        , ambientFactorParam("lighting::ambientFactor", "Factor for the ambient lighting of Blinn-Phong")
        , diffuseFactorParam("lighting::diffuseFactor", "Factor for the diffuse lighting of Blinn-Phong")
        , specularFactorParam("lighting::specularFactor", "Factor for the specular lighting of Blinn-Phong")
        , specularExponentParam("lighting::specularExponent", "Exponent for the specular lighting of Blinn-Phong")
        , useLambertParam(
              "lighting::lambertShading", "If turned on, the local lighting uses lambert instead of Blinn-Phong.")
        , enableShadingParam(
              "lighting::enableShading", "If disabled, the shading performed by this module is turned off")
        , fbo_(nullptr)
        , lightingShader_(nullptr)
        , pointLightBuffer_(nullptr)
        , distantLightBuffer_(nullptr)
        , drawFBOid_(0)
        , readFBOid_(0)
        , FBOid_(0) {

    ambientColorParam.SetParameter(new core::param::ColorParam("#ffffff"));
    diffuseColorParam.SetParameter(new core::param::ColorParam("#ffffff"));
    specularColorParam.SetParameter(new core::param::ColorParam("#ffffff"));

    ambientFactorParam.SetParameter(new core::param::FloatParam(0.2f, 0.0f, 1.0f));
    diffuseFactorParam.SetParameter(new core::param::FloatParam(0.798f, 0.0f, 1.0f));
    specularFactorParam.SetParameter(new core::param::FloatParam(0.02f, 0.0f, 1.0f));
    specularExponentParam.SetParameter(new core::param::FloatParam(120.0f, 1.0f, 1000.0f));

    useLambertParam.SetParameter(new core::param::BoolParam(false));
    enableShadingParam.SetParameter(new core::param::BoolParam(true));
}

DeferredRenderingProvider::~DeferredRenderingProvider(void) {
    // TODO
}

void DeferredRenderingProvider::setup(core::CoreInstance* coreIntstance) {
    try {
        auto const shdr_options = msf::ShaderFactoryOptionsOpenGL(coreIntstance->GetShaderPaths());

        lightingShader_ = core::utility::make_shared_glowl_shader("lighting", shdr_options,
            std::filesystem::path("protein_gl/deferred/lighting.vert.glsl"),
            std::filesystem::path("protein_gl/deferred/lighting.frag.glsl"));

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

    if (pointLights_.empty() && distantLights_.empty()) {
        core::utility::log::Log::DefaultLog.WriteWarn("[DeferredRenderingProvider]: There are no directional or "
                                                      "positional lights connected. Lighting not available.");
    }
}

void DeferredRenderingProvider::draw(
    core_gl::view::CallRender3DGL& call, core::view::light::CallLight* lightCall, bool noShading) {

    bool no_lighting = !this->enableShadingParam.Param<core::param::BoolParam>()->Value();
    if (noShading) {
        no_lighting = true;
    }

    auto cam = call.GetCamera();
    this->refreshLights(lightCall, cam.getPose().direction);

    glm::mat4 MVinv = glm::inverse(cam.getViewMatrix());
    glm::mat4 Pinv = glm::inverse(cam.getProjectionMatrix());

    lightingShader_->use();

    pointLightBuffer_->bind(1);
    lightingShader_->setUniform("point_light_cnt", static_cast<GLint>(pointLights_.size()));

    distantLightBuffer_->bind(2);
    lightingShader_->setUniform("distant_light_cnt", static_cast<GLint>(distantLights_.size()));

    glActiveTexture(GL_TEXTURE0);
    fbo_->bindColorbuffer(0);
    lightingShader_->setUniform("albedo_tx2D", 0);

    glActiveTexture(GL_TEXTURE1);
    fbo_->bindColorbuffer(1);
    lightingShader_->setUniform("normal_tx2D", 1);

    glActiveTexture(GL_TEXTURE2);
    fbo_->bindColorbuffer(2);
    lightingShader_->setUniform("depth_tx2D", 2);

    lightingShader_->setUniform("camPos", cam.getPose().position);
    lightingShader_->setUniform("inv_view_mx", MVinv);
    lightingShader_->setUniform("inv_proj_mx", Pinv);
    lightingShader_->setUniform("use_lambert", this->useLambertParam.Param<core::param::BoolParam>()->Value());
    lightingShader_->setUniform("no_lighting", no_lighting);

    lightingShader_->setUniform(
        "ambientColor", glm::make_vec4(this->ambientColorParam.Param<core::param::ColorParam>()->Value().data()));
    lightingShader_->setUniform(
        "diffuseColor", glm::make_vec4(this->diffuseColorParam.Param<core::param::ColorParam>()->Value().data()));
    lightingShader_->setUniform(
        "specularColor", glm::make_vec4(this->specularColorParam.Param<core::param::ColorParam>()->Value().data()));
    lightingShader_->setUniform("k_amb", this->ambientFactorParam.Param<core::param::FloatParam>()->Value());
    lightingShader_->setUniform("k_diff", this->diffuseFactorParam.Param<core::param::FloatParam>()->Value());
    lightingShader_->setUniform("k_spec", this->specularFactorParam.Param<core::param::FloatParam>()->Value());
    lightingShader_->setUniform("k_exp", this->specularExponentParam.Param<core::param::FloatParam>()->Value());

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glUseProgram(0);
}

void DeferredRenderingProvider::bindDeferredFramebufferToDraw(void) {
    // request old fbo state and set new fbo
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFBOid_);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFBOid_);
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &FBOid_);
    fbo_->bind();

    // request present clear color
    glm::vec4 cc;
    glGetFloatv(GL_COLOR_CLEAR_VALUE, glm::value_ptr(cc));

    // clear everything
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // reset the clear color
    glClearColor(cc.r, cc.g, cc.b, cc.a);
}

void DeferredRenderingProvider::resetToPreviousFramebuffer(void) {
    glBindFramebuffer(GL_FRAMEBUFFER, FBOid_);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFBOid_);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, readFBOid_);
}

std::vector<core::param::ParamSlot*> DeferredRenderingProvider::getUsedParamSlots(void) {
    std::vector<core::param::ParamSlot*> result = {&ambientColorParam, &diffuseColorParam, &specularColorParam,
        &ambientFactorParam, &diffuseFactorParam, &specularFactorParam, &specularExponentParam, &useLambertParam,
        &enableShadingParam};
    return result;
}

void DeferredRenderingProvider::setFramebufferExtents(uint32_t width, uint32_t height) {
    if (width != fbo_->getWidth() || height != fbo_->getHeight()) {
        fbo_->resize(width, height);
    }
}
