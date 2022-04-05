/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore_gl/ResolutionScaler.h"
#include "mmcore/CoreInstance.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"

#include "glowl/Texture2D.hpp"


using namespace megamol;
using namespace megamol::core_gl;


ResolutionScaler::ResolutionScaler(void)
        : core::view::RendererModule<core_gl::view::CallRender3DGL, ModuleGL>()
        , scale_mode_("Scale Mode", "Sets the scale mode for the input fbo, e.g. no scale, bilinear, FSR")
{
    auto scale_modes = new core::param::EnumParam(1);
    scale_modes->SetTypePair(0, "None");
    scale_modes->SetTypePair(1, "Naive");
    scale_modes->SetTypePair(2, "Bilinear");
    scale_modes->SetTypePair(3, "FSR");
    this->scale_mode_.SetParameter(scale_modes);
    // TODO: set callback function here for defines
    // --> re-compile shader based on new defines (iff there are new ones set)
    // same TODO: what about RCAS (sharpening)
    this->MakeSlotAvailable(&scale_mode_);

    this->MakeSlotAvailable(&this->chainRenderSlot);
    this->MakeSlotAvailable(&this->renderSlot);
}


ResolutionScaler::~ResolutionScaler(void) {
    this->Release();
}


bool ResolutionScaler::create(void) {
    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());

    try {
        naive_downsample_prgm_ =
            core::utility::make_glowl_shader("naive_downscale", shader_options, "ResolutionScaler/naive_downscale.comp.glsl");

        naive_upsample_prgm_ =
            core::utility::make_glowl_shader("naive_upscale", shader_options, "ResolutionScaler/naive_upscale.comp.glsl");

        fsr_downsample_prgm_ =
            core::utility::make_glowl_shader("fsr_downscale", shader_options, "ResolutionScaler/fsr_downscale.comp.glsl");

        fsr_upsample_prgm_ =
            core::utility::make_glowl_shader("fsr_upscale", shader_options, "ResolutionScaler/fsr_upscale.comp.glsl");
    }
    catch (std::exception& e){
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            ("ResolutionScaler: " + std::string(e.what())).c_str());
    }

    scaled_fbo_ = std::make_shared<glowl::FramebufferObject>(1,1);

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
    int downsampled_width = fbo_width / 2;
    int downsampled_height = fbo_height / 2;

    int mode = this->scale_mode_.Param<core::param::EnumParam>()->Value();

    // TODO: what about mipmap down- and upscaling?
    // TODO: need a new way of downscaling
    // currently it is only possible to downscale by factor 2 in each direction
    // but what about 1.3, 1.5, or 1.7?
    // AMD launches vertex/fragment shader to do that
    // TODO: do we actually need downsampling anyway?

    // prepare fbo
    if (mode != 0) {
        // bind the newly scaled fbo that should be used by the rhs renderers
        scaled_fbo_->bind();

        scaled_fbo_->resize(downsampled_width, downsampled_height);
        
        // scale down fbo here
        // the downscaled version is the current fbo that the rhs modules automatically use
        // so: downscale --> call rhs with (1) so it uses the downscaled version
        // and after the (1) call, re-scale the fbo and set with call.SetFramebuffer(scaled_fbo_);

        // create color attachments for new scaled fbo
        if (scaled_fbo_->getNumColorAttachments() == 0) {
            for (int i = 0; i < lhs_input_fbo->getNumColorAttachments(); ++i) {
                // use temporary Texture2D as the downsampled color attachment
                auto col_tl = lhs_input_fbo->getColorAttachment(i)->getTextureLayout();
                // TODO: adjust int_params (e.g. fsr gaussian like downscaling requires linear mirror --> see 
                scaled_fbo_->createColorAttachment(col_tl.internal_format, col_tl.format, col_tl.type);
            }
        }

        if (!scaled_fbo_->checkStatus(0)) {
            std::cout << "scaled_fbo_ is broken\n";
        }
    }

    scaled_fbo_->bind();
    glm::vec4 bgc = call.BackgroundColor();
    glClearColor(bgc.r, bgc.g, bgc.b, bgc.a);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    *rhs_chained_call = call;
    if(mode != 0) rhs_chained_call->SetFramebuffer(scaled_fbo_);
    
    // call rhs
    if (!(*rhs_chained_call)(core::view::AbstractCallRender::FnRender)) {
        return false;
    }
    
    // naive upsample
    if (mode == 1) {
        naive_upsample_prgm_->use();

        // downsample color buffers
        for (int i = 0; i < lhs_input_fbo->getNumColorAttachments(); ++i) {
            glActiveTexture(GL_TEXTURE0);
            scaled_fbo_->getColorAttachment(i)->bindTexture();
            glUniform1i(naive_upsample_prgm_->getUniformLocation("g_input_tx2D"), 0);

            lhs_input_fbo->getColorAttachment(i)->bindImage(0, GL_WRITE_ONLY);

            // TODO: calc group sizes better (see parallelcoordinatesrenderer2d)
            glDispatchCompute(static_cast<int>(std::ceil(downsampled_width / 8.0f)),
                static_cast<int>(std::ceil(downsampled_height / 8.0f)), 1);
            ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        // TODO: depth buffer

        glUseProgram(0);
    }

    // fsr upsample
    if (mode == 3) {
        FSRConstants fsr;
        calcConstants(fsr.const0, fsr.const1, fsr.const2, fsr.const3,
            downsampled_width, downsampled_height, fbo_width, fbo_height);
        fsr_consts_ssbo_->rebuffer(&fsr, sizeof(fsr));

        fsr_upsample_prgm_->use();

        // upsample color buffers
        for (int i = 0; i < lhs_input_fbo->getNumColorAttachments(); ++i) {
            fsr_consts_ssbo_->bind(0);

            glActiveTexture(GL_TEXTURE0);
            scaled_fbo_->getColorAttachment(i)->bindTexture();
            glUniform1i(fsr_upsample_prgm_->getUniformLocation("InputTexture"), 0);

            lhs_input_fbo->getColorAttachment(i)->bindImage(1, GL_WRITE_ONLY);

            // TODO: calc group sizes better (see parallelcoordinatesrenderer2d)
            /*glDispatchCompute(static_cast<int>(std::ceil(downsampled_width / 8.0f)),
                static_cast<int>(std::ceil(downsampled_height / 8.0f)), 1);*/
            glDispatchCompute(static_cast<int>(std::ceil(downsampled_width / 8.0f)),
                static_cast<int>(std::ceil(downsampled_height / 8.0f)), 1);
            ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        // TODO: different shaders for signle easu and easu + rcas

        glUseProgram(0);
    }

    call.SetFramebuffer(lhs_input_fbo);

    return true;
}

