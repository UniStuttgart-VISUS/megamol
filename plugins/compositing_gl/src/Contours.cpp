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
#include "mmcore/param/EnumParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"

megamol::compositing_gl::Contours::Contours()
        : mmstd_gl::ModuleGL()
        , version_(0)
        , contoursShader_(nullptr)
        , suggestiveContoursShader_(nullptr)
        , intensityTex_(nullptr)
        , outputTex_(nullptr)

        , outputTexSlot_("OutputTexture", "Gives access to the resulting output texture.")
        , inputColorSlot_("ColorTexture", "Connects the color texture.")
        , inputNormalSlot_("NormalTexture", "Connects the normal render target texture.")
        , inputDepthSlot_("DepthTexture", "Connects the depth render target texture.")
        , cameraSlot_("Camera", "Connects a (copy of) camera state.")
        , sobelThreshold_("Threshold", "Threshold, that determines which gradient values should be used as edge.")
        , radius_("Radius", "Radius for Valey detection inside suggestive contour algorithm.")
        , suggestiveThreshold_("Suggestive_Threshold", "Threshold for p_max - p_i")
        , mode_("contourMode", "Sets Contour Mode to different algorithms.") 
{

    outputTexSlot_.SetCallback(
        CallTexture2D::ClassName(), 
        CallTexture2D::FunctionName(CallTexture2D::CallGetData), 
        &Contours::getDataCallback
    );
    outputTexSlot_.SetCallback(
        CallTexture2D::ClassName(), 
        CallTexture2D::FunctionName(CallTexture2D::CallGetMetaData), 
        &Contours::getMetaDataCallback
    );
    this->MakeSlotAvailable(&outputTexSlot_);

    inputNormalSlot_.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&inputNormalSlot_);

    inputColorSlot_.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&inputColorSlot_);

    inputDepthSlot_.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&inputDepthSlot_);

    cameraSlot_.SetCompatibleCall<CallCameraDescription>();
    this->MakeSlotAvailable(&cameraSlot_);

    sobelThreshold_.SetParameter(new core::param::FloatParam(0.1f, 0.0f, 0.8f, 0.005f));
    this->MakeSlotAvailable(&sobelThreshold_);
    sobelThreshold_.ForceSetDirty();

    radius_.SetParameter(new core::param::IntParam(2));
    this->MakeSlotAvailable(&radius_);
    radius_.ForceSetDirty();

    suggestiveThreshold_.SetParameter(new core::param::FloatParam(0.1f, 0.0f, 10.f, 0.5f));
    this->MakeSlotAvailable(&suggestiveThreshold_);
    suggestiveThreshold_.ForceSetDirty();

    this->mode_ << new megamol::core::param::EnumParam(0);
    this->mode_.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Sobel");
    this->mode_.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "Suggestive");
    this->MakeSlotAvailable(&this->mode_);
}

megamol::compositing_gl::Contours::~Contours() {
    this->Release();
}

bool megamol::compositing_gl::Contours::create() {

    auto const shdr_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

    try {
        contoursShader_ = core::utility::make_glowl_shader(
            "contours", shdr_options, std::filesystem::path("compositing_gl/Contours/contours.comp.glsl"));

        intensityShader_ = core::utility::make_glowl_shader(
            "intensity", shdr_options, std::filesystem::path("compositing_gl/Contours/intensity.comp.glsl"));

        suggestiveContoursShader_ = core::utility::make_glowl_shader(
            "suggestive contours", shdr_options, std::filesystem::path("compositing_gl/Contours/suggestive_contours.comp.glsl"));

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
    intensityTex_ = std::make_shared<glowl::Texture2D>("intensity", tx_layout, nullptr);

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

    if (call_normal != nullptr) {
        if(!(*call_normal)(0)) {
            return false;
        }
    }

    if(call_color != nullptr) {
        if(!(*call_color)(0)) {
            return false;
        }
    }

    if (call_camera != nullptr) {
        if(!(*call_camera)(0)) {
            return false;
        }
    }

    if (call_depth != nullptr) {
        if(!(*call_depth)(0)) {
            return false;
        }
    }
    

    bool incomingChange = call_normal != nullptr && call_normal->hasUpdate() ||
                          call_camera !=nullptr && call_camera->hasUpdate() ||
                          call_depth != nullptr && call_depth->hasUpdate() ||
                          call_color != nullptr && call_color->hasUpdate() || 
                          radius_.IsDirty() ||
                          sobelThreshold_.IsDirty() ||
                          suggestiveThreshold_.IsDirty() ||
                          mode_.IsDirty();

    if (incomingChange) {
        ++version_;

        sobelThreshold_.ResetDirty();
        radius_.ResetDirty();
        suggestiveThreshold_.ResetDirty();
        mode_.ResetDirty();

        auto sobleThresholdVal = sobelThreshold_.Param<core::param::FloatParam>()->Value();
        auto radiusVal = radius_.Param<core::param::IntParam>()->Value();
        auto suggestiveThresholdVal = suggestiveThreshold_.Param<core::param::FloatParam>()->Value();

        auto normal_tex_2D = call_normal->getData();
        auto color_tex_2D = call_color->getData();
        auto depth_tex_2D = call_depth->getData();

        core::view::Camera cam = call_camera->getData();
        auto cam_pose = cam.get<core::view::Camera::Pose>();
        auto view_mx = cam.getViewMatrix();
        auto proj_mx = cam.getProjectionMatrix();

        fitTextures(depth_tex_2D);

        if(this->mode_.Param<core::param::EnumParam>()->Value() == 0) {

            // sobelThreshold_.Param<megamol::core::param::FloatParam>()->SetGUIVisible(true);
            // radius_.Param<megamol::core::param::FloatParam>()->SetGUIVisible(false);
            // suggestiveThreshold_.Param<megamol::core::param::FloatParam>()->SetGUIVisible(false);

            if(contoursShader_ != nullptr) {

                contoursShader_->use();
                contoursShader_->setUniform("threshold", sobelThreshold_.Param<core::param::FloatParam>()->Value());
                bindTexture(contoursShader_, depth_tex_2D, "depth_tex_2D", 0);
                bindTexture(contoursShader_, color_tex_2D, "color_tex_2D", 1);
                outputTex_->bindImage(0, GL_WRITE_ONLY);

                glDispatchCompute(static_cast<int>(std::ceil(outputTex_->getWidth() / 8.0f)),
                    static_cast<int>(std::ceil(outputTex_->getHeight() / 8.0f)), 1);

                glUseProgram(0);
            }

        } else if(this->mode_.Param<core::param::EnumParam>()->Value() == 1) {
            
            // sobelThreshold_.Param<megamol::core::param::FloatParam>()->SetGUIVisible(false);
            // radius_.Param<megamol::core::param::FloatParam>()->SetGUIVisible(true);
            // suggestiveThreshold_.Param<megamol::core::param::FloatParam>()->SetGUIVisible(true);

            intensityShader_->use();

            bindTexture(intensityShader_, depth_tex_2D, "depth_tex_2D", 0);
            bindTexture(intensityShader_, normal_tex_2D, "normal_tex_2D", 1);

            glUniform3fv(intensityShader_->getUniformLocation("cam_pos"), 1, glm::value_ptr(cam_pose.position));

            auto inv_view_mx = glm::inverse(view_mx);
            auto inv_proj_mx = glm::inverse(proj_mx);

            glUniformMatrix4fv(
                intensityShader_->getUniformLocation("inv_view_mx"), 
                1, 
                GL_FALSE, 
                glm::value_ptr(inv_view_mx));

            glUniformMatrix4fv(
                intensityShader_->getUniformLocation("inv_proj_mx"), 
                1, 
                GL_FALSE, 
                glm::value_ptr(inv_proj_mx));

            intensityTex_->bindImage(0, GL_WRITE_ONLY);

            glDispatchCompute(static_cast<int>(std::ceil(outputTex_->getWidth() / 8.0f)),
                static_cast<int>(std::ceil(outputTex_->getHeight() / 8.0f)), 1);

            glUseProgram(0);

            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

            suggestiveContoursShader_->use();

            suggestiveContoursShader_->setUniform("radius", radius_.Param<core::param::IntParam>()->Value());
            suggestiveContoursShader_->setUniform("threshold", suggestiveThreshold_.Param<core::param::FloatParam>()->Value());

            bindTexture(suggestiveContoursShader_, color_tex_2D, "color_tex_2D", 0);
            bindTexture(suggestiveContoursShader_, intensityTex_, "intensity_tex", 1);
            
            outputTex_->bindImage(0, GL_WRITE_ONLY);

            glDispatchCompute(static_cast<int>(std::ceil(outputTex_->getWidth() / 8.0f)),
                static_cast<int>(std::ceil(outputTex_->getHeight() / 8.0f)), 1);

            glUseProgram(0);

            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
        }
    }

    lhs_tc->setData(outputTex_, version_);

    return true;
}

bool megamol::compositing_gl::Contours::getMetaDataCallback(core::Call& caller) {
    return true;
}

void megamol::compositing_gl::Contours::fitTextures(std::shared_ptr<glowl::Texture2D> source) {
    std::pair<int, int> resolution(source->getWidth(), source->getHeight());
    std::vector<std::shared_ptr<glowl::Texture2D>> texVec = {outputTex_, intensityTex_ };
    for (auto& tex : texVec) {
        if (tex->getWidth() != resolution.first || tex->getHeight() != resolution.second) {
            glowl::TextureLayout tx_layout{
                GL_RGBA16F, resolution.first, resolution.second, 1, GL_RGBA, GL_HALF_FLOAT, 1};
            tex->reload(tx_layout, nullptr);
        }
    }
}

void megamol::compositing_gl::Contours::bindTexture(
    std::unique_ptr<glowl::GLSLProgram>& shader,
    std::shared_ptr<glowl::Texture2D> texture, 
    const char* tex_name,
    int num 
) {
    std::vector<int> glTex = {GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3, GL_TEXTURE4 };
    glActiveTexture(glTex[num]);
    texture->bindTexture();
    glUniform1i(shader->getUniformLocation(tex_name), num);
}