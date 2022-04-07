/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore_gl/ResolutionScaler.h"
#include "mmcore/CoreInstance.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"

#include "glowl/Texture2D.hpp"


using namespace megamol;
using namespace megamol::core_gl;

// TODO: values must be in [0, 1], see line 84 following in ffx_fsr1.h
// is this automatically done in opengl?
// TODO: refactor dispatching code into a single function --> no redundant code duplication
// TODO: MIP LOD Bios

ResolutionScaler::ResolutionScaler(void)
        : core::view::RendererModule<core_gl::view::CallRender3DGL, ModuleGL>()
        , scale_mode_("Scale Mode", "Sets the scale mode for the input fbo, e.g. no scale, bilinear, FSR.")
        , rcas_sharpness_attenuation_(
              "Sharpness", "Sets the sharpness attenuation parameter used in RCAS.")
        , fsr_resolution_presets_("Scale Factor", "Sets the scale factor for the resolution (i.e. 2x means the "
                                                      "image is rendered with half the resolution)")
        , scale_factor_(2.f) {
    
    auto scale_modes = new core::param::EnumParam(1);
    scale_modes->SetTypePair(0, "None");
    scale_modes->SetTypePair(1, "Naive");
    scale_modes->SetTypePair(2, "Bilinear");
    scale_modes->SetTypePair(3, "FSR (EASU)");
    scale_modes->SetTypePair(4, "FSR (EASU + RCAS)");
    this->scale_mode_.SetParameter(scale_modes);
    this->scale_mode_.SetUpdateCallback(&ResolutionScaler::scaleModeCallback);
    this->MakeSlotAvailable(&scale_mode_);

    auto presets = new core::param::EnumParam(0);
    presets->SetTypePair(0, "Ultra Quality (1.3x)");
    presets->SetTypePair(1, "Quality (1.5x)");
    presets->SetTypePair(2, "Balanced (1.7x)");
    presets->SetTypePair(3, "Performance (2x)");
    presets->SetGUIVisible(false);
    this->fsr_resolution_presets_.SetParameter(presets);
    this->fsr_resolution_presets_.SetUpdateCallback(&ResolutionScaler::presetCallback);
    this->MakeSlotAvailable(&fsr_resolution_presets_);

    auto sharpness = new core::param::FloatParam(0.25f, 0.f, 2.f);
    sharpness->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Slider);
    sharpness->SetGUIVisible(false);
    this->rcas_sharpness_attenuation_.SetParameter(sharpness);
    this->MakeSlotAvailable(&rcas_sharpness_attenuation_);

    this->MakeSlotAvailable(&this->chainRenderSlot);
    this->MakeSlotAvailable(&this->renderSlot);
}

bool ResolutionScaler::scaleModeCallback(core::param::ParamSlot& slot) {
    int mode = slot.Param<core::param::EnumParam>()->Value();

    if (mode > 1) {
        fsr_resolution_presets_.Parameter()->SetGUIVisible(true);

        if (mode == 4) {
            rcas_sharpness_attenuation_.Parameter()->SetGUIVisible(true);
        } else {
            rcas_sharpness_attenuation_.Parameter()->SetGUIVisible(false);
        }
    } else {
        rcas_sharpness_attenuation_.Parameter()->SetGUIVisible(false);
        fsr_resolution_presets_.Parameter()->SetGUIVisible(false);
        // for now the default factor for naive scale mode is fix 2.f
        scale_factor_ = 2.f;
    }

    return true;
}

bool ResolutionScaler::presetCallback(core::param::ParamSlot& slot) {
    int preset = slot.Param<core::param::EnumParam>()->Value();

    switch (preset) {
    case 0:
        scale_factor_ = 1.3f;   // Ultra Quality
        break;
    case 1:
        scale_factor_ = 1.5f;   // Quality
        break;
    case 2: 
        scale_factor_ = 1.7f;   // Balanced
        break;
    case 3:
        scale_factor_ = 2.f;    // Performance
        break;
    default:
        scale_factor_ = 2.f;
    }

    return true;
}


ResolutionScaler::~ResolutionScaler(void) {
    this->Release();
}


bool ResolutionScaler::create(void) {
    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());

    try {
        naive_upsample_prgm_ =
            core::utility::make_glowl_shader("naive_upscale", shader_options, "ResolutionScaler/naive_upscale.comp.glsl");

        fsr_bilinear_upsample_prgm_ = core::utility::make_glowl_shader(
            "fsr_upscale_bilinear", shader_options, "ResolutionScaler/fsr_upscale_bilinear.comp.glsl");

        fsr_easu_upsample_prgm_ =
            core::utility::make_glowl_shader("fsr_upscale_easu", shader_options, "ResolutionScaler/fsr_upscale_easu.comp.glsl");

        fsr_rcas_upsample_prgm_ = core::utility::make_glowl_shader(
            "fsr_upscale_rcas", shader_options, "ResolutionScaler/fsr_upscale_rcas.comp.glsl");
    }
    catch (std::exception& e){
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            ("ResolutionScaler: " + std::string(e.what())).c_str());
    }

    scaled_fbo_ = std::make_shared<glowl::FramebufferObject>(1,1);

    inter_tl_ = glowl::TextureLayout(GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1);
    intermediary_rcas_tx2D_ = std::make_unique<glowl::Texture2D>("intermediary_rcas_tx2D", inter_tl_, nullptr);

    // params according to AMD, see FSR_Filter.cpp line 48 onwards
    std::vector<std::pair<GLenum, GLint>> fsr_int_params = {{GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST},
        {GL_TEXTURE_MAG_FILTER, GL_LINEAR}, {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE}};

    fsr_input_sampler_ = std::make_unique<glowl::Sampler>("fsr_input_sampler", fsr_int_params);

    fsr_consts_ssbo_ = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    return true;
}


void ResolutionScaler::release(void) {}

void ResolutionScaler::calcConstants(
    glm::uvec4& con0, glm::uvec4& con1,
    glm::uvec4& con2, glm::uvec4& con3,
    float inputSizeX, float inputSizeY,
    float outputSizeX, float outputSizeY)
{
    // Output integer position to a pixel position in viewport.
    con0[0] = AU1_AF1(inputSizeX * (1.f /  outputSizeX));
    con0[1] = AU1_AF1(inputSizeY * (1.f /  outputSizeY));
    con0[2] = AU1_AF1(0.5f * inputSizeX * (1.f / outputSizeX) - 0.5f);
    con0[3] = AU1_AF1(0.5f * inputSizeY * (1.f / outputSizeY) - 0.5f);
    // Viewport pixel position to normalized image space.
    // This is used to get upper-left of 'F' tap.
    con1[0] = AU1_AF1(1.f / inputSizeX);
    con1[1] = AU1_AF1(1.f / inputSizeY);
    // Centers of gather4, first offset from upper-left of 'F'.
    //      +---+---+
    //      |   |   |
    //      +--(0)--+
    //      | b | c |
    //  +---F---+---+---+
    //  | e | f | g | h |
    //  +--(1)--+--(2)--+
    //  | i | j | k | l |
    //  +---+---+---+---+
    //      | n | o |
    //      +--(3)--+
    //      |   |   |
    //      +---+---+
    // These are from (0) instead of 'F'.
    con1[2] = AU1_AF1(1.f  * (1.f / inputSizeX));
    con1[3] = AU1_AF1(-1.f * (1.f / inputSizeY));
    con2[0] = AU1_AF1(-1.f * (1.f / inputSizeX));
    con2[1] = AU1_AF1(2.f  * (1.f / inputSizeY));
    con2[2] = AU1_AF1(1.f  * (1.f / inputSizeX));
    con2[3] = AU1_AF1(2.f  * (1.f / inputSizeY));
    con3[0] = AU1_AF1(0.f  * (1.f / inputSizeX));
    con3[1] = AU1_AF1(4.f  * (1.f / inputSizeY));
    con3[2] = con3[3] = 0;
}

void ResolutionScaler::FsrRcasCon(glm::uvec4& con,
    float sharpness) {
    // Transform from stops to linear value.
    sharpness = exp2f(-sharpness);
    float hSharp[2] = {sharpness, sharpness};
    con[0] = AU1_AF1(sharpness);
    con[1] = AU1_AH2_AF2(hSharp);
    con[2] = 0;
    con[3] = 0;
}

bool ResolutionScaler::GetExtents(core_gl::view::CallRender3DGL& call) {
    core_gl::view::CallRender3DGL* chainedCall = this->chainRenderSlot.CallAs<core_gl::view::CallRender3DGL>();
    if (chainedCall != nullptr) {
        *chainedCall = call;
        bool retVal = (*chainedCall)(core::view::AbstractCallRender::FnGetExtents);
        call = *chainedCall;
    }
    return true;
}

bool ResolutionScaler::Render(view::CallRender3DGL& call) {
    view::CallRender3DGL* rhs_chained_call = this->chainRenderSlot.CallAs<view::CallRender3DGL>();
    if (rhs_chained_call == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "The ResolutionScaler does not work without a renderer attached to its right");
        return false;
    }

    auto lhs_input_fbo = call.GetFramebuffer();
    int fbo_width = lhs_input_fbo->getWidth();
    int fbo_height = lhs_input_fbo->getHeight();
    int downsampled_width = static_cast<int>(fbo_width / scale_factor_);
    int downsampled_height = static_cast<int>(fbo_height / scale_factor_);

    int mode = this->scale_mode_.Param<core::param::EnumParam>()->Value();

    // prepare fbo and intermediary texture
    if (mode != 0) {
        // bind the newly scaled fbo that should be used by the rhs renderers
        scaled_fbo_->bind();

        if (scaled_fbo_->getWidth() != downsampled_width || scaled_fbo_->getHeight() != downsampled_height) {
            scaled_fbo_->resize(downsampled_width, downsampled_height);
        }

        auto input_fbo_tl = lhs_input_fbo->getColorAttachment(0)->getTextureLayout();

        bool fbo_update = input_fbo_tl.internal_format != inter_tl_.internal_format ||
                          input_fbo_tl.format != inter_tl_.format || input_fbo_tl.type != inter_tl_.type ||
                          scaled_fbo_->getNumColorAttachments() == 0;
        bool tx2D_update =
            fbo_update || input_fbo_tl.width != inter_tl_.width || input_fbo_tl.height != inter_tl_.height;

        // prepare color attachment of fbo and intermediate texture for rcas
        if (fbo_update)
        {
            scaled_fbo_->createColorAttachment(input_fbo_tl.internal_format, input_fbo_tl.format, input_fbo_tl.type);

            // prepare intermediate texture used in rcas pass
            if (tx2D_update) {
                intermediary_rcas_tx2D_->reload(input_fbo_tl, nullptr);
            }

            inter_tl_ = input_fbo_tl;
        }
        
        if (scaled_fbo_->checkStatus(0) != GL_FRAMEBUFFER_COMPLETE) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "The scaled_fbo_ in ResolutionScaler did not return GL_FRAMEBUFFER_COMPLETE");
            return false;
        }
    }

    {
        scaled_fbo_->bind();
        glm::vec4 bgc = call.BackgroundColor();
        glClearColor(bgc.r, bgc.g, bgc.b, bgc.a);
        glClearDepth(1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    *rhs_chained_call = call;
    if (mode != 0) rhs_chained_call->SetFramebuffer(scaled_fbo_);

    // call rhs
    if (!(*rhs_chained_call)(core::view::AbstractCallRender::FnRender)) {
        return false;
    }

    
    // we assume only one color attachment in the fbo
    // naive upsample
    if (mode == 1) {
        naive_upsample_prgm_->use();

        // downsample color buffer
        glActiveTexture(GL_TEXTURE0);
        scaled_fbo_->getColorAttachment(0)->bindTexture();
        glUniform1i(naive_upsample_prgm_->getUniformLocation("g_input_tx2D"), 0);

        lhs_input_fbo->getColorAttachment(0)->bindImage(0, GL_WRITE_ONLY);

        glDispatchCompute(static_cast<int>(std::ceil(downsampled_width / 8.0f)),
            static_cast<int>(std::ceil(downsampled_height / 8.0f)), 1);
        ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        resetGLStates();
    }

    FSRConstants fsr;
    if (mode > 1) {
        // TODO: dynamic resolution
        // TODO: are the const values actually correct in shaders?
        calcConstants(fsr.const0, fsr.const1, fsr.const2, fsr.const3, downsampled_width, downsampled_height, fbo_width,
            fbo_height);
        // TODO: kick this part in shaders out, since we dont do hdr
        fsr.Sample = glm::uvec4(0, 0, 0, 0); // always 0 because no hdr is used
        fsr_consts_ssbo_->rebuffer(&fsr, sizeof(fsr));
    }

    // bilinear upsample
    if (mode == 2) {
        fsr_bilinear_upsample_prgm_->use();

        // upample color buffer
        fsr_consts_ssbo_->bind(0);

        glActiveTexture(GL_TEXTURE0);
        scaled_fbo_->getColorAttachment(0)->bindTexture();
        fsr_input_sampler_->bindSampler(0);
        glUniform1i(fsr_bilinear_upsample_prgm_->getUniformLocation("InputTexture"), 0);

        lhs_input_fbo->getColorAttachment(0)->bindImage(1, GL_WRITE_ONLY);

        // TODO: calc group sizes better (see FSR_Filter.cpp in amd)
        glDispatchCompute(static_cast<int>(std::ceil(downsampled_width / 8.0f)),
            static_cast<int>(std::ceil(downsampled_height / 8.0f)), 1);
        ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        resetGLStates();
    }

    // fsr upsample easu
    if (mode > 2) {
        fsr_easu_upsample_prgm_->use();

        // upsample color buffers
        fsr_consts_ssbo_->bind(0);

        glActiveTexture(GL_TEXTURE0);
        scaled_fbo_->getColorAttachment(0)->bindTexture();
        fsr_input_sampler_->bindSampler(0);
        glUniform1i(fsr_easu_upsample_prgm_->getUniformLocation("InputTexture"), 0);

        if (mode == 3) lhs_input_fbo->getColorAttachment(0)->bindImage(1, GL_WRITE_ONLY);
        if (mode == 4) intermediary_rcas_tx2D_->bindImage(1, GL_WRITE_ONLY);

        // TODO: calc group sizes better (see FSR_Filter.cpp in amd)
        glDispatchCompute(static_cast<int>(std::ceil(downsampled_width / 8.0f)),
            static_cast<int>(std::ceil(downsampled_height / 8.0f)), 1);
        ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        resetGLStates();
    }

    // fsr upsample rcas
    if (mode == 4) {
        // re-calculate constants for rcas
        FsrRcasCon(fsr.const0, rcas_sharpness_attenuation_.Param<core::param::FloatParam>()->Value());
        fsr.Sample = glm::uvec4(0, 0, 0, 0); // always 0 because no hdr is used
        fsr_consts_ssbo_->rebuffer(&fsr, sizeof(fsr));


        fsr_rcas_upsample_prgm_->use();

        // upsample color buffers
        fsr_consts_ssbo_->bind(0);

        glActiveTexture(GL_TEXTURE0);
        intermediary_rcas_tx2D_->bindTexture();
        fsr_input_sampler_->bindSampler(0);
        glUniform1i(fsr_rcas_upsample_prgm_->getUniformLocation("InputTexture"), 0);

        lhs_input_fbo->getColorAttachment(0)->bindImage(1, GL_WRITE_ONLY);

        // TODO: calc group sizes better (see FSR_Filter.cpp in amd)
        glDispatchCompute(static_cast<int>(std::ceil(downsampled_width / 8.0f)),
            static_cast<int>(std::ceil(downsampled_height / 8.0f)), 1);
        ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        resetGLStates();
    }

    call.SetFramebuffer(lhs_input_fbo);

    return true;
}

