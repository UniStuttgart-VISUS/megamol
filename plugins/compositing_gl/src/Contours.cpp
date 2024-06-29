/**
 * MegaMol
 * Copyright (c) 2023, MegaMol Dev Team
 * All rights reserved.
 */

#include "Contours.h"

#include <glm/glm.hpp>

#include "compositing_gl/CompositingCalls.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"

megamol::compositing_gl::Contours::Contours()
        : mmstd_gl::ModuleGL()
        , outputTexSlot_("OutputTexture", "Gives access to the resulting output texture")
        , inputColorSlot_("ColorTexture", "Connects the color texture")
        , inputNormalSlot_("NormalTexture", "Connects the normal render target texture")
        , inputDepthSlot_("DepthTexture", "Connects the depth render target texture")
        , cameraSlot_("Camera", "Connects a (copy of) camera state")
        , sobelOperatorThreshold_("Threshold", "Threshold, that determines which gradient values should be used as edge")
        , version_(0)
        , contoursShader_(nullptr)
        , outputTex_(nullptr) {

    outputTexSlot_.SetCallback(CallTexture2D::ClassName(), CallTexture2D::FunctionName(CallTexture2D::CallGetData),
        &Contours::getDataCallback);
    outputTexSlot_.SetCallback(CallTexture2D::ClassName(), CallTexture2D::FunctionName(CallTexture2D::CallGetMetaData),
        &Contours::getMetaDataCallback);
    this->MakeSlotAvailable(&outputTexSlot_);

    inputNormalSlot_.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&inputNormalSlot_);

    inputColorSlot_.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&inputColorSlot_);

    inputDepthSlot_.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&inputDepthSlot_);

    cameraSlot_.SetCompatibleCall<CallCameraDescription>();
    this->MakeSlotAvailable(&cameraSlot_);

    sobelOperatorThreshold_.SetParameter(new core::param::FloatParam(0.1f, 0.0f, 5.3f, 0.005f));
    this->MakeSlotAvailable(&sobelOperatorThreshold_);
    sobelOperatorThreshold_.ForceSetDirty();
}

megamol::compositing_gl::Contours::~Contours() {
    this->Release();
}

bool megamol::compositing_gl::Contours::create() {

    auto const shdr_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());
    try {
        contoursShader_ = core::utility::make_glowl_shader(
            "contours_darken", shdr_options, std::filesystem::path("compositing_gl/contours.comp.glsl"));

    } catch (glowl::GLSLProgramException const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[Contours] %s", ex.what());
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[Contours] Unable to compile shader: Unknown exception: %s", ex.what());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[Contours] Unable to compile shader: Unknown exception.");
    }

    glowl::TextureLayout tx_layout{GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_HALF_FLOAT, 1};
    outputTex_ = std::make_shared<glowl::Texture2D>("contours_output", tx_layout, nullptr);

    return true;
}

void megamol::compositing_gl::Contours::release() {}

bool megamol::compositing_gl::Contours::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    auto call_normal = inputNormalSlot_.CallAs<CallTexture2D>();
    auto call_color  = inputColorSlot_.CallAs<CallTexture2D>();
    auto call_camera = cameraSlot_.CallAs<CallCamera>();
    auto call_depth = inputDepthSlot_.CallAs<CallTexture2D>();

    if (lhs_tc == nullptr) {
        return false;
    }
    if (call_normal == nullptr) {
        return false;
    }
    if(call_color == nullptr) {
        return false;
    }
    if (call_camera == nullptr) {
        return false;
    }

    if (call_depth == nullptr) {
        return false;
    }

    if (!(*call_normal)(0)) {
        return false;
    }
    if (!(*call_color)(0)) {
        return false;
    }

    if (!(*call_depth)(0)) {
        return false;
    }

    if (!(*call_color)(0)) {
        return false;
    }

    if (!(*call_camera)(0)) {
        return false;
    }

    bool incomingChange = (call_normal != nullptr ? call_normal->hasUpdate() : false) ||
                          (call_camera != nullptr ? call_camera->hasUpdate() : false) ||
                          (call_depth != nullptr ? call_depth->hasUpdate() : false) ||
                          (call_color != nullptr ? call_color->hasUpdate() : false) || 
                          sobelOperatorThreshold_.IsDirty();

    if (incomingChange) {
        ++version_;

        sobelOperatorThreshold_.ResetDirty();

        auto normal_tex_2D = call_normal->getData();
        auto color_tex_2D = call_color->getData();
        auto depth_tex_2D = call_depth->getData();

        core::view::Camera cam = call_camera->getData();
        auto cam_pose = cam.get<core::view::Camera::Pose>();
        auto view_mx = cam.getViewMatrix();
        auto proj_mx = cam.getProjectionMatrix();

        std::array<float, 2> texture_resolution = {
            static_cast<float>(normal_tex_2D->getWidth()), 
            static_cast<float>(normal_tex_2D->getHeight())
        };

        if (outputTex_->getWidth() != std::get<0>(texture_resolution) || 
            outputTex_->getHeight() != std::get<1>(texture_resolution)){
            glowl::TextureLayout tx_layout(
                GL_RGBA16F, std::get<0>(texture_resolution), std::get<1>(texture_resolution), 1, GL_RGBA, GL_HALF_FLOAT, 1);
            outputTex_->reload(tx_layout, nullptr);
        }

        if(contoursShader_ != nullptr) {
            contoursShader_->use();

            contoursShader_->setUniform("threshold", sobelOperatorThreshold_.Param<core::param::FloatParam>()->Value());

            glActiveTexture(GL_TEXTURE0);
            normal_tex_2D->bindTexture();
            glUniform1i(contoursShader_->getUniformLocation("normal_tex_2D"), 0);

            glActiveTexture(GL_TEXTURE1);
            color_tex_2D->bindTexture();
            glUniform1i(contoursShader_->getUniformLocation("color_tex_2D"), 1);

            glActiveTexture(GL_TEXTURE2);
            depth_tex_2D->bindTexture();
            glUniform1i(contoursShader_->getUniformLocation("depth_tex_2D"), 2);

            glUniform3fv(contoursShader_->getUniformLocation("cam_pos"), 1, glm::value_ptr(cam_pose.position));


            auto inv_view_mx = glm::inverse(view_mx);
            auto inv_proj_mx = glm::inverse(proj_mx);

            glUniformMatrix4fv(
                contoursShader_->getUniformLocation("inv_view_mx"), 
                1, 
                GL_FALSE, 
                glm::value_ptr(inv_view_mx));

            glUniformMatrix4fv(
                contoursShader_->getUniformLocation("inv_proj_mx"), 
                1, 
                GL_FALSE, 
                glm::value_ptr(inv_proj_mx));

            outputTex_->bindImage(0, GL_WRITE_ONLY);

            glDispatchCompute(static_cast<int>(std::ceil(outputTex_->getWidth() / 8.0f)),
                static_cast<int>(std::ceil(outputTex_->getHeight() / 8.0f)), 1);

            glUseProgram(0);
        }
    }

    lhs_tc->setData(outputTex_, version_);

    return true;
}

bool megamol::compositing_gl::Contours::getMetaDataCallback(core::Call& caller) {
    return true;
}