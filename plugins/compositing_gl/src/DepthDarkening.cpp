/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "DepthDarkening.h"

#include <glm/glm.hpp>

#include "compositing_gl/CompositingCalls.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"

megamol::compositing_gl::DepthDarkening::DepthDarkening()
        : mmstd_gl::ModuleGL()
        , outputTexSlot_("OutputTexture", "Gives access to the resulting output texture")
        , inputColorSlot_("ColorTexture", "Connects the color render target texture")
        , inputDepthSlot_("DepthTexture", "Connects the depth render target texture")
        , kernelRadiusParam_(
              "kernelRadius", "The radius of the used gauss kernel in pixels (for Full HD 40 is recommended)")
        , lambdaValueParam_("lambda", "Lambda value determining the strength of the darkening effect. 0 shuts it off")
        , out_texture_format_slot_("OutTexFormat", "texture format of output texture")
        , version_(0)
        , blurShader_(nullptr)
        , darkenShader_(nullptr)
        , intermediateTex_(nullptr)
        , intermediateTex2_(nullptr)
        , outputTex_(nullptr)
        , outFormatHandler_("OUTFORMAT", {GL_RGBA8_SNORM, GL_RGBA16F, GL_RGBA32F},
              std::function<bool()>(std::bind(&DepthDarkening::textureFormatUpdate, this))) {

    outputTexSlot_.SetCallback(CallTexture2D::ClassName(), CallTexture2D::FunctionName(CallTexture2D::CallGetData),
        &DepthDarkening::getDataCallback);
    outputTexSlot_.SetCallback(CallTexture2D::ClassName(), CallTexture2D::FunctionName(CallTexture2D::CallGetMetaData),
        &DepthDarkening::getMetaDataCallback);
    this->MakeSlotAvailable(&outputTexSlot_);

    inputColorSlot_.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&inputColorSlot_);

    inputDepthSlot_.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&inputDepthSlot_);

    kernelRadiusParam_.SetParameter(new core::param::IntParam(40, 1, 100));
    this->MakeSlotAvailable(&kernelRadiusParam_);
    kernelRadiusParam_.ForceSetDirty();

    lambdaValueParam_.SetParameter(new core::param::FloatParam(4.0f, 0.0f, 100.0f));
    this->MakeSlotAvailable(&lambdaValueParam_);
    lambdaValueParam_.ForceSetDirty();

    this->MakeSlotAvailable(outFormatHandler_.getFormatSelectorSlot());
}

megamol::compositing_gl::DepthDarkening::~DepthDarkening() {
    this->Release();
}

bool megamol::compositing_gl::DepthDarkening::create() {
    textureFormatUpdate();
    auto const shdr_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

    auto shader_options_flags = outFormatHandler_.addDefinitions(shdr_options);

    try {
        blurShader_ = core::utility::make_glowl_shader(
            "dd_blur", *shader_options_flags, std::filesystem::path("compositing_gl/gauss_blur.comp.glsl"));

        darkenShader_ = core::utility::make_glowl_shader(
            "dd_darken", *shader_options_flags, std::filesystem::path("compositing_gl/depth_darkening.comp.glsl"));

    } catch (glowl::GLSLProgramException const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[DepthDarkening] %s", ex.what());
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[DepthDarkening] Unable to compile shader: Unknown exception: %s", ex.what());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[DepthDarkening] Unable to compile shader: Unknown exception.");
    }

    glowl::TextureLayout tx_layout{(GLint)outFormatHandler_.getInternalFormat(), 1, 1, 1, outFormatHandler_.getFormat(),
        outFormatHandler_.getType(), 1};
    outputTex_ = std::make_shared<glowl::Texture2D>("depth_darkening_output", tx_layout, nullptr);
    intermediateTex_ = std::make_shared<glowl::Texture2D>("depth_darkening_intermediate", tx_layout, nullptr);
    intermediateTex2_ = std::make_shared<glowl::Texture2D>("depth_darkening_intermediate2", tx_layout, nullptr);

    gaussValues_ = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    return true;
}

void megamol::compositing_gl::DepthDarkening::release() {}

bool megamol::compositing_gl::DepthDarkening::getDataCallback(core::Call& caller) {
    if (outFormatHandler_.recentlyChanged()) {
        textureFormatUpdate();
    }

    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    auto call_color = inputColorSlot_.CallAs<CallTexture2D>();
    auto call_depth = inputDepthSlot_.CallAs<CallTexture2D>();

    if (lhs_tc == nullptr) {
        return false;
    }
    if (call_color == nullptr) {
        return false;
    }
    if (call_depth == nullptr) {
        return false;
    }

    if (!(*call_color)(CallTexture2D::CallGetData)) {
        return false;
    }
    if (!(*call_depth)(CallTexture2D::CallGetData)) {
        return false;
    }

    bool incomingChange = (call_color != nullptr ? call_color->hasUpdate() : false) ||
                          (call_depth != nullptr ? call_depth->hasUpdate() : false) || kernelRadiusParam_.IsDirty() ||
                          lambdaValueParam_.IsDirty();

    if (incomingChange) {
        ++version_;

        if (kernelRadiusParam_.IsDirty()) {
            this->recalcKernel();
            kernelRadiusParam_.ResetDirty();
        }
        lambdaValueParam_.ResetDirty();

        auto& color_tex = call_color->getData();
        auto& depth_tex = call_depth->getData();

        fitTextures(depth_tex);

        // first step : blur horizontally
        blurShader_->use();
        gaussValues_->bind(1);

        blurShader_->setUniform("kernel_radius", kernelRadiusParam_.Param<core::param::IntParam>()->Value());
        blurShader_->setUniform("kernel_direction", 1, 0);

        glActiveTexture(GL_TEXTURE0);
        depth_tex->bindTexture();
        blurShader_->setUniform("source_tex", 0);

        intermediateTex_->bindImage(0, GL_WRITE_ONLY);

        glDispatchCompute(static_cast<int>(std::ceil(intermediateTex_->getWidth() / 8.0f)),
            static_cast<int>(std::ceil(intermediateTex_->getHeight() / 8.0f)), 1);

        glUseProgram(0);

        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

        // second step : blur vertically
        blurShader_->use();
        gaussValues_->bind(1);

        blurShader_->setUniform("kernel_radius", kernelRadiusParam_.Param<core::param::IntParam>()->Value());
        blurShader_->setUniform("kernel_direction", 0, 1);

        glActiveTexture(GL_TEXTURE0);
        intermediateTex_->bindTexture();
        blurShader_->setUniform("source_tex", 0);

        intermediateTex2_->bindImage(0, GL_WRITE_ONLY);

        glDispatchCompute(static_cast<int>(std::ceil(intermediateTex2_->getWidth() / 8.0f)),
            static_cast<int>(std::ceil(intermediateTex2_->getHeight() / 8.0f)), 1);

        glUseProgram(0);

        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

        // last step : depth darkening
        darkenShader_->use();

        darkenShader_->setUniform("lambda", this->lambdaValueParam_.Param<core::param::FloatParam>()->Value());

        glActiveTexture(GL_TEXTURE0);
        color_tex->bindTexture();
        darkenShader_->setUniform("color_tex", 0);

        glActiveTexture(GL_TEXTURE1);
        depth_tex->bindTexture();
        darkenShader_->setUniform("depth_tex", 1);

        glActiveTexture(GL_TEXTURE2);
        intermediateTex2_->bindTexture();
        darkenShader_->setUniform("blurred_depth_tex", 2);

        outputTex_->bindImage(0, GL_WRITE_ONLY);

        glDispatchCompute(static_cast<int>(std::ceil(outputTex_->getWidth() / 8.0f)),
            static_cast<int>(std::ceil(outputTex_->getHeight() / 8.0f)), 1);

        glUseProgram(0);

        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    }

    lhs_tc->setData(outputTex_, version_);

    return true;
}

bool megamol::compositing_gl::DepthDarkening::getMetaDataCallback(core::Call& caller) {
    return true;
}

void megamol::compositing_gl::DepthDarkening::fitTextures(std::shared_ptr<glowl::Texture2D> source) {
    std::pair<int, int> resolution(source->getWidth(), source->getHeight());
    std::vector<std::shared_ptr<glowl::Texture2D>> texVec = {outputTex_, intermediateTex_, intermediateTex2_};
    for (auto& tex : texVec) {
        if (tex->getWidth() != resolution.first || tex->getHeight() != resolution.second) {
            glowl::TextureLayout tx_layout{out_tex_internal_format_, resolution.first, resolution.second, 1,
                (GLenum)out_tex_format_, (GLenum)out_tex_type_, 1};
            tex->reload(tx_layout, nullptr);
        }
    }
}

void megamol::compositing_gl::DepthDarkening::recalcKernel() {
    auto radius = kernelRadiusParam_.Param<core::param::IntParam>()->Value();
    auto length = 2 * radius - 1;
    std::vector<float> kernelVec(length, 0.0f);
    // the cutoff can be after 2 sigma to each side, therefore we set it appropriately
    // so that it fits to the kernel width
    auto sigma = 0.25f * static_cast<float>(length);
    float sum = 0.0f;
    // while we do not have full C++ 20 we have to do this:
    float pi = std::atan(1) * 4;
    // calc the kernel values
    for (int i = 0; i < kernelVec.size(); ++i) {
        float dist = static_cast<float>(i - radius + 1);
        float val = 1.0f / std::sqrt(2.0f * pi * sigma * sigma);
        val *= std::exp(-(dist * dist) / (2.0f * sigma * sigma));
        kernelVec[i] = val;
        sum += val;
    }
    // normalize the kernel
    for (auto& v : kernelVec) {
        v /= sum;
    }
    // upload the kernel to GPU
    gaussValues_->rebuffer(kernelVec);
}

bool megamol::compositing_gl::DepthDarkening::textureFormatUpdate() {

    // reinit all textures
    auto const shdr_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

    auto shader_options_flags = outFormatHandler_.addDefinitions(shdr_options);

    try {
        blurShader_ = core::utility::make_glowl_shader(
            "dd_blur", *shader_options_flags, std::filesystem::path("compositing_gl/gauss_blur.comp.glsl"));

        darkenShader_ = core::utility::make_glowl_shader(
            "dd_darken", *shader_options_flags, std::filesystem::path("compositing_gl/depth_darkening.comp.glsl"));

    } catch (glowl::GLSLProgramException const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[DepthDarkening] %s", ex.what());
        return false;
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[DepthDarkening] Unable to compile shader: Unknown exception: %s", ex.what());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[DepthDarkening] Unable to compile shader: Unknown exception.");
        return false;
    }
    glowl::TextureLayout tx_layout;

    //checks if slot is connected
    if (inputDepthSlot_.GetStatus() == 2) {
        tx_layout = glowl::TextureLayout{(GLint)outFormatHandler_.getInternalFormat(),
            static_cast<int>(inputDepthSlot_.CallAs<CallTexture2D>()->getData()->getWidth()),
            static_cast<int>(inputDepthSlot_.CallAs<CallTexture2D>()->getData()->getHeight()), 1,
            outFormatHandler_.getFormat(), outFormatHandler_.getType(), 1};
    } else {
        tx_layout = glowl::TextureLayout{(GLint)outFormatHandler_.getInternalFormat(), 1, 1, 1,
            outFormatHandler_.getFormat(), outFormatHandler_.getType(), 1};
    }
    outputTex_ = std::make_shared<glowl::Texture2D>("depth_darkening_output", tx_layout, nullptr);
    intermediateTex_ = std::make_shared<glowl::Texture2D>("depth_darkening_intermediate", tx_layout, nullptr);
    intermediateTex2_ = std::make_shared<glowl::Texture2D>("depth_darkening_intermediate2", tx_layout, nullptr);

    return true;
}
