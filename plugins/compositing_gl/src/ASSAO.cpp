/*
 * ASSAO.cpp
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016, Intel Corporation
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "ASSAO.h"

#include <array>
#include <random>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "vislib/graphics/gl/ShaderSource.h"

#include "compositing/CompositingCalls.h"


/////////////////////////////////////////////////////////////////////////
// CONSTANTS
/////////////////////////////////////////////////////////////////////////
#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)     \
    {                       \
        if (p) {            \
            (p)->Release(); \
            (p) = NULL;     \
        }                   \
    }
#endif
#ifndef SAFE_RELEASE_ARRAY
#define SAFE_RELEASE_ARRAY(p)                 \
    {                                         \
        for (int i = 0; i < _countof(p); i++) \
            if (p[i]) {                       \
                (p[i])->Release();            \
                (p[i]) = NULL;                \
            }                                 \
    }
#endif

#define SSA_STRINGIZIZER(x) SSA_STRINGIZIZER_(x)
#define SSA_STRINGIZIZER_(x) #x

#define MEGAMOL_ASSAO_MANUAL_MIPS

#define SSAO_MAX_TAPS 32
#define SSAO_MAX_REF_TAPS 512
#define SSAO_ADAPTIVE_TAP_BASE_COUNT 5
#define SSAO_ADAPTIVE_TAP_FLEXIBLE_COUNT (SSAO_MAX_TAPS - SSAO_ADAPTIVE_TAP_BASE_COUNT)
#define SSAO_DEPTH_MIP_LEVELS 4

#ifdef INTEL_SSAO_ENABLE_NORMAL_WORLD_TO_VIEW_CONVERSION
#define SSAO_ENABLE_NORMAL_WORLD_TO_VIEW_CONVERSION 1
#else
#define SSAO_ENABLE_NORMAL_WORLD_TO_VIEW_CONVERSION 0
#endif
/////////////////////////////////////////////////////////////////////////

megamol::compositing::ASSAO::ASSAO()
    : core::Module()
    , m_version(0)
    , m_output_texture_hash(0)
    , m_output_tex_slot("OutputTexture", "Gives access to resulting output texture")
    , m_normals_tex_slot("NormalTexture", "Connects the normals render target texture")
    , m_depth_tex_slot("DepthTexture", "Connects the depth render target texture")
    , m_camera_slot("Camera", "Connects a (copy of) camera state")
    , m_halfDepths{nullptr, nullptr, nullptr, nullptr}
    , m_halfDepthsMipViews{}
    , m_pingPongHalfResultA(nullptr)
    , m_pingPongHalfResultB(nullptr)
    , m_finalResults(nullptr)
    , m_finalResultsArrayViews{nullptr, nullptr, nullptr, nullptr}
    , m_normals(nullptr)
    , m_finalOutput(nullptr)
    , m_samplerStatePointClamp()
    , m_samplerStatePointMirror()
    , m_samplerStateLinearClamp()
    , m_samplerStateViewspaceDepthTap()
    , m_depthBufferViewspaceLinearLayout()
    , m_AOResultLayout()
    , m_size(0, 0)
    , m_halfSize(0, 0)
    , m_quarterSize(0, 0)
    , m_fullResOutScissorRect(0, 0, 0, 0)
    , m_halfResOutScissorRect(0, 0, 0, 0)
    , m_depthMipLevels(0)
    , m_inputs(nullptr)
    , m_max_blur_pass_count(6)
    , m_ssbo_constants(nullptr)
    , m_settings()
    , m_psRadius("Radius", "Specifies world (view) space size of the occlusion sphere")
    , m_psShadowMultiplier("ShadowMultiplier", "Specifies effect strength linear multiplier")
    , m_psShadowPower("ShadowPower", "Specifies the effect strength pow modifier")
    , m_psShadowClamp("ShadowClamp", "Specifies the effect max limit")
    , m_psHorizonAngleThreshold("HorizonAngleThreshold", "Specifies the self-shadowing limit")
    , m_psFadeOutFrom("FadeOutFrom", "Specifies the distance to start fading out the effect")
    , m_psFadeOutTo("FadeOutTo", "Specifies the distance at which the effect is faded out")
    , m_psQualityLevel("QualityLevel", "Specifies the ssao effect quality level")
    , m_psAdaptiveQualityLimit("AdaptiveQualityLimit", "Specifies the adaptive quality limit (only for quality level 3)")
    , m_psBlurPassCount("BlurPassCount", "Specifies the number of edge-sensitive smart blur passes to apply")
    , m_psSharpness("Sharpness", "Specifies how much to bleed over edges")
    , m_psTemporalSupersamplingAngleOffset("TemporalSupersamplingAngleOffset", "Specifies the rotating of the sampling kernel if temporal AA / supersampling is used")
    , m_psTemporalSupersamplingRadiusOffset("TemporalSupersamplingRadiusOffset", "Specifies the scaling of the sampling kernel if temporal AA / supersampling is used")
    , m_psDetailShadowStrength("DetailShadowStrength", "Specifies the high-res detail AO using neighboring depth pixels")
    , m_settingsHaveChanged(false)
{
    this->m_output_tex_slot.SetCallback(CallTexture2D::ClassName(), "GetData", &ASSAO::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &ASSAO::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_normals_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_normals_tex_slot);

    this->m_depth_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_depth_tex_slot);

    this->m_camera_slot.SetCompatibleCall<CallCameraDescription>();
    this->MakeSlotAvailable(&this->m_camera_slot);

    // settings
    this->m_psRadius.SetParameter(new core::param::FloatParam(1.2f, 0.f));
    this->m_psRadius.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psRadius.SetUpdateCallback(&ASSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psRadius);

    this->m_psShadowMultiplier.SetParameter(new core::param::FloatParam(1.f, 0.f, 5.f));
    this->m_psShadowMultiplier.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psShadowMultiplier.SetUpdateCallback(&ASSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psShadowMultiplier);

    this->m_psShadowPower.SetParameter(new core::param::FloatParam(1.5f, 0.5f, 5.f));
    this->m_psShadowPower.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psShadowPower.SetUpdateCallback(&ASSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psShadowPower);

    this->m_psShadowClamp.SetParameter(new core::param::FloatParam(0.98f, 0.f, 1.f));
    this->m_psShadowClamp.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psShadowClamp.SetUpdateCallback(&ASSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psShadowClamp);

    this->m_psHorizonAngleThreshold.SetParameter(new core::param::FloatParam(0.06f, 0.f, 0.2f));
    this->m_psHorizonAngleThreshold.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psHorizonAngleThreshold.SetUpdateCallback(&ASSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psHorizonAngleThreshold);

    this->m_psFadeOutFrom.SetParameter(new core::param::FloatParam(50.f, 0.f));
    this->m_psFadeOutFrom.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psFadeOutFrom.SetUpdateCallback(&ASSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psFadeOutFrom);

    this->m_psFadeOutTo.SetParameter(new core::param::FloatParam(300.f, 0.f));
    this->m_psFadeOutTo.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psFadeOutTo.SetUpdateCallback(&ASSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psFadeOutTo);

    // generally there are quality levels from -1 (lowest) to 3 (highest, adaptive), but 3 (adaptive) is not implemented yet
    this->m_psQualityLevel.SetParameter(new core::param::EnumParam(2));
    this->m_psQualityLevel.Param<core::param::EnumParam>()->SetTypePair(-1, "Lowest");
    this->m_psQualityLevel.Param<core::param::EnumParam>()->SetTypePair( 0, "Low");
    this->m_psQualityLevel.Param<core::param::EnumParam>()->SetTypePair( 1, "Medium");
    this->m_psQualityLevel.Param<core::param::EnumParam>()->SetTypePair( 2, "High");
    this->m_psQualityLevel.SetUpdateCallback(&ASSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psQualityLevel);

    this->m_psAdaptiveQualityLimit.SetParameter(new core::param::FloatParam(0.45f, 0.f, 1.f));
    this->m_psAdaptiveQualityLimit.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psAdaptiveQualityLimit.SetUpdateCallback(&ASSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psAdaptiveQualityLimit);

    this->m_psBlurPassCount.SetParameter(new core::param::IntParam(2, 0, 6));
    this->m_psBlurPassCount.SetUpdateCallback(&ASSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psBlurPassCount);

    this->m_psSharpness.SetParameter(new core::param::FloatParam(0.98f, 0.f, 1.f));
    this->m_psSharpness.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psSharpness.SetUpdateCallback(&ASSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psSharpness);

    this->m_psTemporalSupersamplingAngleOffset.SetParameter(new core::param::FloatParam(0.f, 0.f, 3.141592653589f));
    this->m_psTemporalSupersamplingAngleOffset.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psTemporalSupersamplingAngleOffset.SetUpdateCallback(&ASSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psTemporalSupersamplingAngleOffset);

    this->m_psTemporalSupersamplingRadiusOffset.SetParameter(new core::param::FloatParam(1.f, 0.f, 2.f));
    this->m_psTemporalSupersamplingRadiusOffset.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psTemporalSupersamplingRadiusOffset.SetUpdateCallback(&ASSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psTemporalSupersamplingRadiusOffset);

    this->m_psDetailShadowStrength.SetParameter(new core::param::FloatParam(0.5f, 0.f, 5.f));
    this->m_psDetailShadowStrength.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psDetailShadowStrength.SetUpdateCallback(&ASSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psDetailShadowStrength);
}

bool megamol::compositing::ASSAO::settingsCallback(core::param::ParamSlot& slot) {
    m_settings.Radius = m_psRadius.Param<core::param::FloatParam>()->Value();
    m_settings.ShadowMultiplier = m_psShadowMultiplier.Param<core::param::FloatParam>()->Value();
    m_settings.ShadowPower = m_psShadowPower.Param<core::param::FloatParam>()->Value();
    m_settings.ShadowClamp = m_psShadowClamp.Param<core::param::FloatParam>()->Value();
    m_settings.HorizonAngleThreshold = m_psHorizonAngleThreshold.Param<core::param::FloatParam>()->Value();
    m_settings.FadeOutFrom = m_psFadeOutFrom.Param<core::param::FloatParam>()->Value();
    m_settings.FadeOutTo = m_psFadeOutTo.Param<core::param::FloatParam>()->Value();
    m_settings.QualityLevel = m_psQualityLevel.Param<core::param::EnumParam>()->Value();
    m_settings.AdaptiveQualityLimit = m_psAdaptiveQualityLimit.Param<core::param::FloatParam>()->Value();
    m_settings.BlurPassCount = m_psBlurPassCount.Param<core::param::IntParam>()->Value();
    m_settings.Sharpness = m_psSharpness.Param<core::param::FloatParam>()->Value();
    m_settings.TemporalSupersamplingAngleOffset =
        m_psTemporalSupersamplingAngleOffset.Param<core::param::FloatParam>()->Value();
    m_settings.TemporalSupersamplingRadiusOffset =
        m_psTemporalSupersamplingRadiusOffset.Param<core::param::FloatParam>()->Value();
    m_settings.DetailShadowStrength = m_psDetailShadowStrength.Param<core::param::FloatParam>()->Value();

    m_settingsHaveChanged = true;

    return true;
}

// TODO: DELETE ALL MEMORY!
megamol::compositing::ASSAO::~ASSAO() { this->Release(); }

bool megamol::compositing::ASSAO::create() {
    try {
        {
            // create shader program
            m_prepare_depths_prgm = std::make_unique<GLSLComputeShader>();
            m_prepare_depths_half_prgm = std::make_unique<GLSLComputeShader>();
            m_prepare_depths_and_normals_prgm = std::make_unique<GLSLComputeShader>();
            m_prepare_depths_and_normals_half_prgm = std::make_unique<GLSLComputeShader>();
            m_prepare_depth_mip_prgms.resize(SSAO_DEPTH_MIP_LEVELS - 1);
            for (auto& cs : m_prepare_depth_mip_prgms) {
                cs = std::make_unique<GLSLComputeShader>();
            }
            for (auto& cs : m_generate_prgms) {
                cs = std::make_unique<GLSLComputeShader>();
            }
            m_smart_blur_prgm = std::make_unique<GLSLComputeShader>();
            m_smart_blur_wide_prgm = std::make_unique<GLSLComputeShader>();
            m_apply_prgm = std::make_unique<GLSLComputeShader>();
            m_non_smart_blur_prgm = std::make_unique<GLSLComputeShader>();
            m_non_smart_apply_prgm = std::make_unique<GLSLComputeShader>();
            m_non_smart_half_apply_prgm = std::make_unique<GLSLComputeShader>();

            vislib::graphics::gl::ShaderSource cs_prepare_depths;
            vislib::graphics::gl::ShaderSource cs_prepare_depths_half;
            vislib::graphics::gl::ShaderSource cs_prepare_depths_and_normals;
            vislib::graphics::gl::ShaderSource cs_prepare_depths_and_normals_half;
            std::vector<vislib::graphics::gl::ShaderSource> cs_prepare_depth_mip(SSAO_DEPTH_MIP_LEVELS - 1);
            std::vector<vislib::graphics::gl::ShaderSource> cs_generate(5);
            vislib::graphics::gl::ShaderSource cs_smart_blur;
            vislib::graphics::gl::ShaderSource cs_smart_blur_wide;
            vislib::graphics::gl::ShaderSource cs_apply;
            vislib::graphics::gl::ShaderSource cs_non_smart_blur;
            vislib::graphics::gl::ShaderSource cs_non_smart_apply;
            vislib::graphics::gl::ShaderSource cs_non_smart_half_apply;

            std::cout << "Compiling: Compositing::assao::CSPrepareDepths\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSPrepareDepths", cs_prepare_depths))
                return false;
            if (!m_prepare_depths_prgm->Compile(cs_prepare_depths.Code(), cs_prepare_depths.Count()))
                return false;
            if (!m_prepare_depths_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSPrepareDepthsHalf\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSPrepareDepthsHalf", cs_prepare_depths_half))
                return false;
            if (!m_prepare_depths_half_prgm->Compile(cs_prepare_depths_half.Code(), cs_prepare_depths_half.Count()))
                return false;
            if (!m_prepare_depths_half_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSPrepareDepthsAndNormals\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSPrepareDepthsAndNormals", cs_prepare_depths_and_normals))
                return false;
            if (!m_prepare_depths_and_normals_prgm->Compile(
                    cs_prepare_depths_and_normals.Code(), cs_prepare_depths_and_normals.Count()))
                return false;
            if (!m_prepare_depths_and_normals_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSPrepareDepthsAndNormalsHalf\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSPrepareDepthsAndNormalsHalf", cs_prepare_depths_and_normals_half))
                return false;
            if (!m_prepare_depths_and_normals_half_prgm->Compile(
                    cs_prepare_depths_and_normals_half.Code(), cs_prepare_depths_and_normals_half.Count()))
                return false;
            if (!m_prepare_depths_and_normals_half_prgm->Link())
                return false;

            for (int i = 0; i < SSAO_DEPTH_MIP_LEVELS - 1; ++i) {
                std::cout << "Compiling: Compositing::assao::CSPrepareDepthMip" << i + 1 << "\n";
                std::string identifier = "Compositing::assao::CSPrepareDepthMip" + std::to_string(i + 1);
                if (!instance()->ShaderSourceFactory().MakeShaderSource(
                        identifier.c_str(), cs_prepare_depth_mip[i]))
                    return false;
                if (!m_prepare_depth_mip_prgms[i]->Compile(
                        cs_prepare_depth_mip[i].Code(), cs_prepare_depth_mip[i].Count()))
                    return false;
                if (!m_prepare_depth_mip_prgms[i]->Link())
                    return false;
            }

            // one less than cs_generate.size() because the adaptive quality level is not implemented (yet)
            for (int i = 0; i < 4; ++i) { 
                std::cout << "Compiling: Compositing::assao::CSGenerateQ" << i << "\n";
                std::string identifier = "Compositing::assao::CSGenerateQ" + std::to_string(i);
                if (i >= 4)
                    identifier = "Compositing::assao::CSGenerateQ3Base";

                if (!instance()->ShaderSourceFactory().MakeShaderSource(identifier.c_str(), cs_generate[i]))
                    return false;
                if (!m_generate_prgms[i]->Compile(cs_generate[i].Code(), cs_generate[i].Count()))
                    return false;
                if (!m_generate_prgms[i]->Link())
                    return false;
            }

            std::cout << "Compiling: Compositing::assao::CSSmartBlur\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::assao::CSSmartBlur", cs_smart_blur))
                return false;
            if (!m_smart_blur_prgm->Compile(cs_smart_blur.Code(), cs_smart_blur.Count()))
                return false;
            if (!m_smart_blur_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSSmartBlurWide\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSSmartBlurWide", cs_smart_blur_wide))
                return false;
            if (!m_smart_blur_wide_prgm->Compile(cs_smart_blur_wide.Code(), cs_smart_blur_wide.Count()))
                return false;
            if (!m_smart_blur_wide_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSNonSmartBlur\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSNonSmartBlur", cs_non_smart_blur))
                return false;
            if (!m_non_smart_blur_prgm->Compile(cs_non_smart_blur.Code(), cs_non_smart_blur.Count()))
                return false;
            if (!m_non_smart_blur_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSApply\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::assao::CSApply", cs_apply))
                return false;
            if (!m_apply_prgm->Compile(cs_apply.Code(), cs_apply.Count()))
                return false;
            if (!m_apply_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSNonSmartApply\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSNonSmartApply", cs_non_smart_apply))
                return false;
            if (!m_non_smart_apply_prgm->Compile(cs_non_smart_apply.Code(), cs_non_smart_apply.Count()))
                return false;
            if (!m_non_smart_apply_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSNonSmartHalfApply\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSNonSmartHalfApply", cs_non_smart_half_apply))
                return false;
            if (!m_non_smart_half_apply_prgm->Compile(cs_non_smart_half_apply.Code(), cs_non_smart_half_apply.Count()))
                return false;
            if (!m_non_smart_half_apply_prgm->Link())
                return false;

            std::cout << "Done Compiling\n";
        }

    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile shader: Unknown exception\n");
        return false;
    }

    m_depthBufferViewspaceLinearLayout = glowl::TextureLayout(GL_R16F, 1, 1, 1, GL_RED, GL_HALF_FLOAT, 1);
    m_AOResultLayout = glowl::TextureLayout(GL_RG8, 1, 1, 1, GL_RG, GL_UNSIGNED_BYTE, 1);
    m_halfDepths[0] = std::make_shared<glowl::Texture2D>("m_halfDepths0", m_depthBufferViewspaceLinearLayout, nullptr);
    m_halfDepths[1] = std::make_shared<glowl::Texture2D>("m_halfDepths1", m_depthBufferViewspaceLinearLayout, nullptr);
    m_halfDepths[2] = std::make_shared<glowl::Texture2D>("m_halfDepths2", m_depthBufferViewspaceLinearLayout, nullptr);
    m_halfDepths[3] = std::make_shared<glowl::Texture2D>("m_halfDepths3", m_depthBufferViewspaceLinearLayout, nullptr);
    m_halfDepthsMipViews.resize(4);
    for (int j = 0; j < 4; ++j) {
        m_halfDepthsMipViews[j].resize(SSAO_DEPTH_MIP_LEVELS);
        for (int i = 0; i < m_halfDepthsMipViews[j].size(); ++i) {
            m_halfDepthsMipViews[j][i] =
                std::make_shared<glowl::Texture2DView>(
                "m_halfDepthsMipViews" + std::to_string(i), *m_halfDepths[j], m_depthBufferViewspaceLinearLayout, 0, 1, 0, 1);
        }
    }
    m_finalOutput = std::make_shared<glowl::Texture2D>("m_finalOutput", m_depthBufferViewspaceLinearLayout, nullptr);
    m_pingPongHalfResultA = std::make_shared<glowl::Texture2D>("m_pingPongHalfResultA", m_AOResultLayout, nullptr);
    m_pingPongHalfResultB = std::make_shared<glowl::Texture2D>("m_pingPongHalfResultB", m_AOResultLayout, nullptr);
    m_finalResults = std::make_shared<glowl::Texture2DArray>("m_finalResults", m_AOResultLayout, nullptr);
    m_finalResultsArrayViews[0] = std::make_shared<glowl::Texture2DView>(
        "m_finalResultsArrayViews0", *m_finalResults, m_AOResultLayout, 0, 1, 0, 1);
    m_finalResultsArrayViews[1] = std::make_shared<glowl::Texture2DView>(
        "m_finalResultsArrayViews1", *m_finalResults, m_AOResultLayout, 0, 1, 0, 1);
    m_finalResultsArrayViews[2] = std::make_shared<glowl::Texture2DView>(
        "m_finalResultsArrayViews2", *m_finalResults, m_AOResultLayout, 0, 1, 0, 1);
    m_finalResultsArrayViews[3] = std::make_shared<glowl::Texture2DView>(
        "m_finalResultsArrayViews3", *m_finalResults, m_AOResultLayout, 0, 1, 0, 1);
    // #ifdef generatenormals
    //m_normals = std::make_shared<glowl::Texture2D>("m_normals", tx_layout, nullptr);
    //#endif

    m_inputs = std::make_shared<ASSAO_Inputs>();

    m_ssbo_constants = std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    std::vector<std::pair<GLenum, GLint>> int_params = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST},
        {GL_TEXTURE_MAG_FILTER, GL_NEAREST}, {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE}};

    m_samplerStatePointClamp = std::make_shared<glowl::Sampler>("samplerStatePointClamp", int_params);

    int_params.clear();
    int_params = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST}, {GL_TEXTURE_MAG_FILTER, GL_NEAREST},
        {GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT}, {GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT}};

    m_samplerStatePointMirror = std::make_shared<glowl::Sampler>("samplerStatePointMirror", int_params);

    int_params.clear();
    int_params = {{GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR}, {GL_TEXTURE_MAG_FILTER, GL_LINEAR},
        {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}};

    m_samplerStateLinearClamp = std::make_shared<glowl::Sampler>("samplerStateLinearClamp", int_params);

    int_params.clear();
    int_params = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST},
        {GL_TEXTURE_MAG_FILTER, GL_NEAREST}, {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}};

    m_samplerStateViewspaceDepthTap =
        std::make_shared<glowl::Sampler>("samplerStateViewspaceDepthTap", int_params);

    return true;
}

void megamol::compositing::ASSAO::release() {}

bool megamol::compositing::ASSAO::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    auto call_normal = m_normals_tex_slot.CallAs<CallTexture2D>();
    auto call_depth = m_depth_tex_slot.CallAs<CallTexture2D>();
    auto call_camera = m_camera_slot.CallAs<CallCamera>();

    if (lhs_tc == NULL) return false;
    
    if ((call_normal != NULL) && (call_depth != NULL) && (call_camera != NULL)) {

        if (!(*call_normal)(0))
            return false;

         if (!(*call_depth)(0))
            return false;

         if (!(*call_camera)(0))
             return false;

        // something has changed in the neath...
        bool normal_update = call_normal->hasUpdate();
        bool depth_update = call_depth->hasUpdate();
        bool camera_update = call_camera->hasUpdate();

        bool something_has_changed =
            (call_normal != NULL ? normal_update : false) || 
            (call_depth != NULL ? depth_update : false) || 
            (call_camera != NULL ? camera_update : false) ||
            m_settingsHaveChanged;

        if (something_has_changed) {
            ++m_version;

            if (call_normal == NULL)
                return false;
            if (call_depth == NULL)
                return false;
            if (call_camera == NULL)
                return false;

            auto normal_tx2D = call_normal->getData();
            auto depth_tx2D = call_depth->getData();
            std::array<int, 2> tx_res_normal = {(int) normal_tx2D->getWidth(), (int) normal_tx2D->getHeight()};
            std::array<int, 2> tx_res_depth = {(int) depth_tx2D->getWidth(), (int) depth_tx2D->getHeight()};

            {
                glowl::TextureLayout finalLy = glowl::TextureLayout(
                    GL_RGBA16F,
                    tx_res_depth[0],
                    tx_res_depth[1],
                    1,
                    GL_RGBA,
                    GL_HALF_FLOAT,
                    1);
                m_finalOutput->reload(finalLy, nullptr);
            }

            // obtain camera information
            core::view::Camera_2 cam = call_camera->getData();
            cam_type::snapshot_type snapshot;
            cam_type::matrix_type view_tmp, proj_tmp;
            cam.calc_matrices(snapshot, view_tmp, proj_tmp, core::thecam::snapshot_content::all);
            glm::mat4 view_mx = view_tmp;
            glm::mat4 proj_mx = proj_tmp;
            

            if (normal_update || depth_update || m_settingsHaveChanged) {
                
                m_inputs->ViewportWidth = tx_res_normal[0];
                m_inputs->ViewportHeight = tx_res_normal[1];
                m_inputs->generateNormals = normal_tx2D == nullptr;
                // for now we won't use scissortests
                // but the implementation still stays for a more easy migration
                // can be removed or used if (not) needed
                // scissor rectangle stays constant for now
                m_inputs->ScissorLeft = 0;
                m_inputs->ScissorRight = tx_res_normal[0];
                m_inputs->ScissorTop = tx_res_normal[1];
                m_inputs->ScissorBottom = 0;
                m_inputs->ProjectionMatrix = proj_mx;
                m_inputs->ViewMatrix = view_mx;

                updateTextures(m_inputs);

                updateConstants(m_settings, m_inputs, 0);
            }

            {
                // only required when scissors are used
                /*if (m_requiresClear) {
                m_halfDepths[0]->reload(m_halfDepths[0]->getTextureLayout(), nullptr);
                m_halfDepths[1]->reload(m_halfDepths[1]->getTextureLayout(), nullptr);
                m_halfDepths[2]->reload(m_halfDepths[2]->getTextureLayout(), nullptr);
                m_halfDepths[3]->reload(m_halfDepths[3]->getTextureLayout(), nullptr);
                m_pingPongHalfResultA->reload(m_pingPongHalfResultA->getTextureLayout(), nullptr);
                m_pingPongHalfResultB->reload(m_pingPongHalfResultB->getTextureLayout(), nullptr);
                m_finalResultsArrayViews[0]->reload(m_finalResultsArrayViews[0]->getTextureLayout(), nullptr);
                m_finalResultsArrayViews[1]->reload(m_finalResultsArrayViews[1]->getTextureLayout(), nullptr);
                m_finalResultsArrayViews[2]->reload(m_finalResultsArrayViews[2]->getTextureLayout(), nullptr);
                m_finalResultsArrayViews[3]->reload(m_finalResultsArrayViews[3]->getTextureLayout(), nullptr);
                if (m_normals != nullptr)
                    m_normals->reload(m_normals->getTextureLayout(), nullptr);

                m_requiresClear = false;
                }*/

                prepareDepths(m_settings, m_inputs, depth_tx2D, normal_tx2D);
                 
                generateSSAO(m_settings, m_inputs, false, depth_tx2D, normal_tx2D);

                std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTextures = {
                    {m_finalOutput, 0}
                };

                // Apply
                {
                    TextureArraySamplerTuple inputFinals =
                        {m_finalResults, "g_FinalSSAO", m_samplerStateLinearClamp};
                
                    // TODO: blending states
                
                    if (m_settings.QualityLevel < 0)
                        fullscreenPassDraw <TextureSamplerTuple, glowl::Texture2D>(m_non_smart_half_apply_prgm, {}, outputTextures, true, inputFinals);
                    else if (m_settings.QualityLevel == 0)
                        fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                            m_non_smart_apply_prgm, {}, outputTextures, true, inputFinals);
                    else
                        fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                            m_apply_prgm, {}, outputTextures, true, inputFinals);
                }
            }
        }

    }

    if (lhs_tc->version() < m_version) {
        m_settingsHaveChanged = false;
        lhs_tc->setData(m_finalOutput, m_version);
    }

    return true;
}

void megamol::compositing::ASSAO::prepareDepths(
    const ASSAO_Settings& settings, const std::shared_ptr<ASSAO_Inputs> inputs, std::shared_ptr<glowl::Texture2D> depthTexture,
    std::shared_ptr<glowl::Texture2D> normalTexture) {
    bool generateNormals = inputs->generateNormals;

    std::vector<TextureSamplerTuple> inputTextures(1);
    inputTextures[0] = { depthTexture, (std::string) "g_DepthSource", nullptr /*m_samplerStatePointClamp*/ };

    std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputFourDepths = {
        {m_halfDepths[0], 0}, {m_halfDepths[1], 1}, {m_halfDepths[2], 2}, {m_halfDepths[3], 3}};
    std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTwoDepths = {
        {m_halfDepths[0], 0}, {m_halfDepths[3], 3}};

    if (!generateNormals) {
        if (settings.QualityLevel < 0) {
            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                m_prepare_depths_half_prgm, inputTextures, outputTwoDepths);
        } else {
            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                m_prepare_depths_prgm, inputTextures, outputFourDepths);
        }
    } else {
        if (settings.QualityLevel < 0) {
            // TODO: binding not up-to-date
            outputTwoDepths.push_back({normalTexture, 7});
            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                m_prepare_depths_and_normals_half_prgm, inputTextures, outputTwoDepths);
        } else {
            // TODO: binding not up-to-date
            outputFourDepths.push_back({normalTexture, 7});
            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                m_prepare_depths_and_normals_prgm, inputTextures, outputFourDepths);
        }
    }

    // only do mipmaps for higher quality levels (not beneficial on quality level 1, and detrimental on quality level 0)
    if (settings.QualityLevel > 1) {

        #ifdef MEGAMOL_ASSAO_MANUAL_MIPS
        for (int i = 1; i < m_depthMipLevels; ++i) {
            std::vector<TextureViewSamplerTuple> inputFourDepthMipsM1 = {
                {m_halfDepthsMipViews[0][i - 1LL], (std::string) "g_ViewspaceDepthSource" , nullptr},
                {m_halfDepthsMipViews[1][i - 1LL], (std::string) "g_ViewspaceDepthSource1", nullptr},
                {m_halfDepthsMipViews[2][i - 1LL], (std::string) "g_ViewspaceDepthSource2", nullptr},
                {m_halfDepthsMipViews[3][i - 1LL], (std::string) "g_ViewspaceDepthSource3", nullptr}};
            
            std::vector<std::pair<std::shared_ptr<glowl::Texture2DView>, GLuint>> outputFourDepthMips = {
                {m_halfDepthsMipViews[0][i], 0}, {m_halfDepthsMipViews[1][i], 1}, {m_halfDepthsMipViews[2][i], 2},
                {m_halfDepthsMipViews[3][i], 3}};

            fullscreenPassDraw<TextureViewSamplerTuple, glowl::Texture2DView>(
                m_prepare_depth_mip_prgms[i - 1LL], inputFourDepthMipsM1, outputFourDepthMips);
        }
        #else
        for (int i = 0; i < 4; ++i) {
            m_halfDepths[i]->bindTexture();
            m_halfDepths[i]->updateMipmaps();
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        #endif
    }
}

void megamol::compositing::ASSAO::generateSSAO(const ASSAO_Settings& settings,
    const std::shared_ptr<ASSAO_Inputs> inputs, bool adaptiveBasePass, std::shared_ptr<glowl::Texture2D> depthTexture,
    std::shared_ptr<glowl::Texture2D> normalTexture) {

    // omitted viewport and scissor code from intel here

    if (adaptiveBasePass) {
        assert(settings.QualityLevel == 3);
    }

    int passCount = 4;

    for (int pass = 0; pass < passCount; ++pass) {
        if ((settings.QualityLevel < 0) && ((pass == 1) || (pass == 2)))
            continue;

        int blurPasses = settings.BlurPassCount;
        blurPasses = std::min(blurPasses, m_max_blur_pass_count);

        // CHECK FOR ADAPTIVE SSAO
#ifdef INTEL_SSAO_ENABLE_ADAPTIVE_QUALITY
        if (settings.QualityLevel == 3) {
            // if adaptive, at least one blur pass needed as the first pass needs to read the final texture results -
            // kind of awkward
            if (adaptiveBasePass)
                blurPasses = 0;
            else
                blurPasses = Max(1, blurPasses);
        } else
#endif
        if (settings.QualityLevel <= 0) {
            // just one blur pass allowed for minimum quality
            // MM simply uses one blur pass
            blurPasses = std::min(1, settings.BlurPassCount);
        }

        updateConstants(settings, inputs, pass);

        // Generate
        {
            std::vector<TextureSamplerTuple> inputTextures(3);
            inputTextures[0] = {m_halfDepths[pass], "g_ViewSpaceDepthSource", m_samplerStatePointMirror};
            inputTextures[1] = {normalTexture, "g_NormalmapSource", nullptr};
            inputTextures[2] = {m_halfDepths[pass], "g_ViewSpaceDepthSourceDepthTapSampler", m_samplerStateViewspaceDepthTap};

            // CHECK FOR ADAPTIVE SSAO
#ifdef INTEL_SSAO_ENABLE_ADAPTIVE_QUALITY
            if (!adaptiveBasePass && (settings.QualityLevel == 3)) {
                inputTextures[3] = {m_loadCounterSRV, "g_LoadCounter"};
                inputTextures[4] = {m_importanceMap.SRV, "g_ImportanceMap"};
                inputTextures[5] = {m_finalResults.SRV, "g_FinalSSAO"};
            }
#endif
            GLuint binding = 0;
            int shaderIndex = std::max(0, !adaptiveBasePass ? settings.QualityLevel : 4);

            // no blur?
            if (blurPasses == 0) {
                std::vector<std::pair<std::shared_ptr<glowl::Texture2DView>, GLuint>> outputTextures = {
                    {m_finalResultsArrayViews[pass], binding}};
            
                fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2DView>(
                    m_generate_prgms[shaderIndex], inputTextures, outputTextures);
            } else {
                std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTextures = {
                    {m_pingPongHalfResultA, binding}};
            
                fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                    m_generate_prgms[shaderIndex], inputTextures, outputTextures);
            }

            
        }

        // Blur
        if (blurPasses > 0) {
            int wideBlursRemaining = std::max(0, blurPasses - 2);

            for (int i = 0; i < blurPasses; ++i) {
                GLuint binding = 0;

                std::vector<TextureSamplerTuple> inputTextures = {
                    {m_pingPongHalfResultA, "g_BlurInput", m_samplerStatePointMirror}};

                // TODO: re-do below code, ugly af
                if (settings.QualityLevel > 0) {
                    if (wideBlursRemaining > 0) {
                        if (i == blurPasses - 1) {
                            std::vector<std::pair<std::shared_ptr<glowl::Texture2DView>, GLuint>> outputTextures = {
                                {m_finalResultsArrayViews[pass], binding}};
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2DView>(
                                m_smart_blur_wide_prgm, inputTextures, outputTextures);
                        } else {
                            std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTextures = {
                                {m_pingPongHalfResultB, binding}};
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                                m_smart_blur_wide_prgm, inputTextures, outputTextures);
                        }
                        wideBlursRemaining--;
                    } else {
                        if (i == blurPasses - 1) {
                            std::vector<std::pair<std::shared_ptr<glowl::Texture2DView>, GLuint>> outputTextures = {
                                {m_finalResultsArrayViews[pass], binding}};
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2DView>(
                                m_smart_blur_prgm, inputTextures, outputTextures);
                        } else {
                            std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTextures = {
                                {m_pingPongHalfResultB, binding}};
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                                m_smart_blur_prgm, inputTextures, outputTextures);
                        }
                    }
                } else {
                    std::get<2>(inputTextures[0]) = m_samplerStateLinearClamp;
                    if (i == blurPasses - 1) {
                        std::vector<std::pair<std::shared_ptr<glowl::Texture2DView>, GLuint>> outputTextures = {
                            {m_finalResultsArrayViews[pass], binding}};
                        fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2DView>(
                            m_non_smart_blur_prgm, inputTextures, outputTextures);
                    } else {
                        std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTextures = {
                            {m_pingPongHalfResultB, binding}};
                        fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                            m_non_smart_blur_prgm, inputTextures, outputTextures);
                    } // just for quality level 0 (and -1)
                }

                std::swap(m_pingPongHalfResultA, m_pingPongHalfResultB);
            }
        }
    }
}

bool megamol::compositing::ASSAO::getMetaDataCallback(core::Call& caller) { return true; }

// TODO: the whole functions seems redundant.
// i think it is possible to call it only once at the beginning
// no need for further calls afterwards
void megamol::compositing::ASSAO::updateTextures(const std::shared_ptr<ASSAO_Inputs> inputs) {
    int width = inputs->ViewportWidth;
    int height = inputs->ViewportHeight;

    bool needsUpdate = (m_size.x != width) || (m_size.y != height);

    m_size.x = width;
    m_size.y = height;
    m_halfSize.x = (width + 1) / 2;
    m_halfSize.y = (height + 1) / 2;
    m_quarterSize.x = (m_halfSize.x + 1) / 2;
    m_quarterSize.y = (m_halfSize.y + 1) / 2;

    glm::ivec4 prevScissorRect = m_fullResOutScissorRect;

    if ((inputs->ScissorRight == 0) || (inputs->ScissorTop == 0)) {
        m_fullResOutScissorRect = glm::ivec4(0, 0, width, height);
    } else {
        m_fullResOutScissorRect =
            glm::ivec4(std::max(0, inputs->ScissorLeft),
            std::max(0, inputs->ScissorBottom),
            std::min(width, inputs->ScissorRight),
            std::min(0, inputs->ScissorTop));
    }

    needsUpdate |= (prevScissorRect != m_fullResOutScissorRect);
    if (!needsUpdate)
        return;

    m_halfResOutScissorRect =
        glm::ivec4(m_fullResOutScissorRect.x / 2, m_fullResOutScissorRect.y / 2,
        (m_fullResOutScissorRect.z + 1) / 2, (m_fullResOutScissorRect.w + 1) / 2);

    int blurEnlarge = m_max_blur_pass_count + std::max(0, m_max_blur_pass_count - 2); // +1 for max normal blurs, +2 for wide blurs

    m_halfResOutScissorRect =
        glm::ivec4(std::max(0, m_halfResOutScissorRect.x - blurEnlarge),
        std::max(0, m_halfResOutScissorRect.y - blurEnlarge),
        std::min(m_halfSize.x, m_halfResOutScissorRect.z + blurEnlarge),
        std::min(m_halfSize.y, m_halfResOutScissorRect.w + blurEnlarge));

    float totalSizeInMB = 0.f;

    m_depthMipLevels = SSAO_DEPTH_MIP_LEVELS;

    for (int i = 0; i < 4; i++) {
        if (reCreateIfNeeded(m_halfDepths[i], m_halfSize, m_depthBufferViewspaceLinearLayout, true)) {

            #ifdef MEGAMOL_ASSAO_MANUAL_MIPS
            for (int j = 0; j < m_depthMipLevels; j++) {
                std::cout << "mip run number: " << j << "\n";
                reCreateMIPViewIfNeeded(m_halfDepthsMipViews[i][j], m_halfDepths[i], j);
            }
            #endif
        }
    }

    reCreateIfNeeded(m_pingPongHalfResultA, m_halfSize, m_AOResultLayout);
    reCreateIfNeeded(m_pingPongHalfResultB, m_halfSize, m_AOResultLayout);
    reCreateIfNeeded(m_finalResults, m_halfSize, m_AOResultLayout);

    for (int i = 0; i < 4; ++i) {
        std::cout << "array run number: " << i << "\n";
        reCreateArrayIfNeeded(m_finalResultsArrayViews[i], m_finalResults, m_halfSize, m_AOResultLayout, i);
    }

    // trigger a full buffers clear first time; only really required when using scissor rects
    //m_requiresClear = true;
}

void megamol::compositing::ASSAO::updateConstants(
    const ASSAO_Settings& settings, const std::shared_ptr<ASSAO_Inputs> inputs, int pass) {
    bool generateNormals = inputs->generateNormals;

    // update constants
    ASSAO_Constants& consts = m_constants;// = *((ASSAOConstants*) mappedResource.pData);

    const glm::mat4& proj = inputs->ProjectionMatrix;

    consts.ViewportPixelSize = glm::vec2(1.0f / (float) m_size.x, 1.0f / (float) m_size.y);
    consts.HalfViewportPixelSize = glm::vec2(1.0f / (float) m_halfSize.x, 1.0f / (float) m_halfSize.y);

    consts.Viewport2xPixelSize = glm::vec2(consts.ViewportPixelSize.x * 2.0f, consts.ViewportPixelSize.y * 2.0f);
    consts.Viewport2xPixelSize_x_025 =
        glm::vec2(consts.Viewport2xPixelSize.x * 0.25f, consts.Viewport2xPixelSize.y * 0.25f);

    // requires proj matrix to be in column-major order
    float depthLinearizeMul = proj[3][2]; // float depthLinearizeMul = -( 2.0 * clipFar * clipNear ) / ( clipFar - clipNear );
    float depthLinearizeAdd = proj[2][2];  // float depthLinearizeAdd = -(clipFar + clipNear) / ( clipFar - clipNear );
    
    // correct the handedness issue. need to make sure this below is correct, but I think it is.
    if (depthLinearizeMul * depthLinearizeAdd < 0)
        depthLinearizeAdd = -depthLinearizeAdd;
    //consts.DepthUnpackConsts = glm::vec2(depthLinearizeMul, depthLinearizeAdd);
    consts.DepthUnpackConsts = glm::vec2(depthLinearizeMul, depthLinearizeAdd);
    
    float tanHalfFOVX = 1.0f / proj[0][0]; // = tanHalfFOVY * drawContext.Camera.GetAspect( );
    float tanHalfFOVY = 1.0f / proj[1][1]; // = tanf( drawContext.Camera.GetYFOV( ) * 0.5f );

    consts.CameraTanHalfFOV = glm::vec2(tanHalfFOVX, tanHalfFOVY);

    consts.NDCToViewMul = glm::vec2(consts.CameraTanHalfFOV.x * 2.f, consts.CameraTanHalfFOV.y * 2.f);
    consts.NDCToViewAdd = glm::vec2(consts.CameraTanHalfFOV.x * -1.f, consts.CameraTanHalfFOV.y * -1.f);

    consts.EffectRadius = std::clamp(settings.Radius, 0.0f, 100000.0f);
    consts.EffectShadowStrength = std::clamp(settings.ShadowMultiplier * 4.3f, 0.0f, 10.0f);
    consts.EffectShadowPow = std::clamp(settings.ShadowPower, 0.0f, 10.0f);
    consts.EffectShadowClamp = std::clamp(settings.ShadowClamp, 0.0f, 1.0f);
    consts.EffectFadeOutMul = -1.0f / (settings.FadeOutTo - settings.FadeOutFrom);
    consts.EffectFadeOutAdd = settings.FadeOutFrom / (settings.FadeOutTo - settings.FadeOutFrom) + 1.0f;
    consts.EffectHorizonAngleThreshold = std::clamp(settings.HorizonAngleThreshold, 0.0f, 1.0f);

    // 1.2 seems to be around the best trade off - 1.0 means on-screen radius will stop/slow growing when the camera
    // is at 1.0 distance, so, depending on FOV, basically filling up most of the screen This setting is
    // viewspace-dependent and not screen size dependent intentionally, so that when you change FOV the effect stays
    // (relatively) similar.
    float effectSamplingRadiusNearLimit = (settings.Radius * 1.2f);

    // if the depth precision is switched to 32bit float, this can be set to something closer to 1 (0.9999 is fine)
    consts.DepthPrecisionOffsetMod = 0.9992f; // TODO: is this correct? or does it need to be slightly > 1?

    // consts.RadiusDistanceScalingFunctionPow     = 1.0f - Clamp( settings.RadiusDistanceScalingFunction,
    // 0.0f, 1.0f );

    int lastHalfDepthMipX = m_halfDepthsMipViews[0][SSAO_DEPTH_MIP_LEVELS - 1]->getWidth();
    int lastHalfDepthMipY = m_halfDepthsMipViews[0][SSAO_DEPTH_MIP_LEVELS - 1]->getHeight();

    // used to get average load per pixel; 9.0 is there to compensate for only doing every 9th InterlockedAdd in
    // PSPostprocessImportanceMapB for performance reasons
    consts.LoadCounterAvgDiv = 9.0f / (float) (m_quarterSize.x * m_quarterSize.y * 255.0);

    // Special settings for lowest quality level - just nerf the effect a tiny bit
    if (settings.QualityLevel <= 0) {
        // consts.EffectShadowStrength     *= 0.9f;
        effectSamplingRadiusNearLimit *= 1.50f;

        if (settings.QualityLevel < 0)
            consts.EffectRadius *= 0.8f;
    }
    effectSamplingRadiusNearLimit /= tanHalfFOVY; // to keep the effect same regardless of FOV

    consts.EffectSamplingRadiusNearLimitRec = 1.0f / effectSamplingRadiusNearLimit;

    consts.AdaptiveSampleCountLimit = settings.AdaptiveQualityLimit;

    consts.NegRecEffectRadius = -1.0f / consts.EffectRadius;

    consts.PerPassFullResCoordOffset = glm::ivec2(pass % 2, pass / 2);
    consts.PerPassFullResUVOffset = glm::vec2(((pass % 2) - 0.0f) / m_size.x, ((pass / 2) - 0.0f) / m_size.y);

    consts.InvSharpness = std::clamp(1.0f - settings.Sharpness, 0.0f, 1.0f);
    consts.PassIndex = pass;
    consts.QuarterResPixelSize = glm::vec2(1.0f / (float) m_quarterSize.x, 1.0f / (float) m_quarterSize.y);

    float additionalAngleOffset =
        settings.TemporalSupersamplingAngleOffset; // if using temporal supersampling approach (like "Progressive
                                                    // Rendering Using Multi-frame Sampling" from GPU Pro 7, etc.)
    float additionalRadiusScale =
        settings.TemporalSupersamplingRadiusOffset; // if using temporal supersampling approach (like "Progressive
                                                    // Rendering Using Multi-frame Sampling" from GPU Pro 7, etc.)
    const int subPassCount = 5;
    for (int subPass = 0; subPass < subPassCount; subPass++) {
        int a = pass;
        int b = subPass;

        int spmap[5]{0, 1, 4, 3, 2};
        b = spmap[subPass];

        float ca, sa;
        float angle0 = ((float) a + (float) b / (float) subPassCount) * (3.1415926535897932384626433832795f) * 0.5f;
        angle0 += additionalAngleOffset;

        ca = ::cosf(angle0);
        sa = ::sinf(angle0);

        float scale = 1.0f + (a - 1.5f + (b - (subPassCount - 1.0f) * 0.5f) / (float) subPassCount) * 0.07f;
        scale *= additionalRadiusScale;

        // all values are within [-1, 1]
        consts.PatternRotScaleMatrices[subPass] = glm::vec4(scale * ca, scale * -sa, -scale * sa, -scale * ca);
    }

    // TODO: check if good
    if (!generateNormals) {
        consts.NormalsUnpackMul = inputs->NormalsUnpackMul;
        consts.NormalsUnpackAdd = inputs->NormalsUnpackAdd;
    } else {
        consts.NormalsUnpackMul = 2.0f;
        consts.NormalsUnpackAdd = -1.0f;
    }
    consts.DetailAOStrength = settings.DetailShadowStrength;
    consts.Dummy0 = 0.0f;

    consts.viewMX = inputs->ViewMatrix;

#if SSAO_ENABLE_NORMAL_WORLD_TO_VIEW_CONVERSION
    if (!generateNormals) {
        consts.NormalsWorldToViewspaceMatrix = inputs->NormalsWorldToViewspaceMatrix;
        if (!inputs->MatricesRowMajorOrder)
            consts.NormalsWorldToViewspaceMatrix.Transpose();
    } else {
        consts.NormalsWorldToViewspaceMatrix.SetIdentity();
    }
#endif

    // probably do something with the ssbo? but could also just be done at this point
    m_ssbo_constants->rebuffer(&m_constants, sizeof(m_constants));
}

// only resets textures if needed
bool megamol::compositing::ASSAO::reCreateIfNeeded(
    std::shared_ptr<glowl::Texture2D> tex, glm::ivec2 size, const glowl::TextureLayout& ly, bool generateMipMaps) {
    if ((size.x == 0) || (size.y == 0)) {
        // reset object
    } else {
        if (tex != nullptr) {
            glowl::TextureLayout desc = tex->getTextureLayout();
            if (equalLayoutsWithoutSize(desc, ly) && (desc.width == size.x) && (desc.height == size.y))
                return false;
        }

        glowl::TextureLayout desc = ly;
        desc.width = size.x;
        desc.height = size.y;
        if (generateMipMaps) {
            desc.levels = SSAO_DEPTH_MIP_LEVELS;
            tex->reload(desc, nullptr, true, true);
        }
        else
            tex->reload(desc, nullptr);
    }

    return true;
}

bool megamol::compositing::ASSAO::reCreateIfNeeded(
    std::shared_ptr<glowl::Texture2DArray> tex, glm::ivec2 size, const glowl::TextureLayout& ly) {
    if ((size.x == 0) || (size.y == 0)) {
        // reset object
    } else {
        if (tex != nullptr) {
            glowl::TextureLayout desc = tex->getTextureLayout();
            if (equalLayoutsWithoutSize(desc, ly) && (desc.width == size.x) && (desc.height == size.y))
                return false;
        }

        glowl::TextureLayout desc = ly;
        desc.width = size.x;
        desc.height = size.y;
        desc.depth = 4;
        tex->reload(desc, nullptr);
    }

    return true;
}

// TODO: make checks for euqality in reCreateArrayIfNeeded
bool megamol::compositing::ASSAO::reCreateArrayIfNeeded(std::shared_ptr<glowl::Texture2DView> tex,
    std::shared_ptr<glowl::Texture2DArray> original, glm::ivec2 size, const glowl::TextureLayout& ly, int arraySlice) {
    if ((size.x == 0) || (size.y == 0)) {
        //tex->~Texture2D();
    } else {
        /*if (tex != nullptr) {
            glowl::TextureLayout desc = tex->getTextureLayout();
            if (equalLayoutsWithoutSize(desc, ly) && (desc.width == size.x) && (desc.height == size.y))
                return false;
        }*/

        // not needed, glTextureView does this automatically
        /*glowl::TextureLayout desc = ly;
        desc.width = size.x;
        desc.height = size.y;
        tex->reload(desc, nullptr);*/

        // make to one function "makeTextureView" for error handling in texture2d.hpp
        tex->reload(*original, original->getTextureLayout(), 0, 1, arraySlice, 1);
        /*glTextureView(
            tex->getName(), GL_TEXTURE_2D, original->getName(), original->getInternalFormat(), 0, 1, arraySlice, 1);*/

        // debug
        /*tex->bindTexture();
        int w, h;
        int miplevel = 0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, miplevel, GL_TEXTURE_WIDTH, &w);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, miplevel, GL_TEXTURE_HEIGHT, &h);

        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cout << "gltextureview failed\n";
        }*/
    }

    return true;
}


// TODO: make checks for euqality in reCreateMIPViewIfNeeded
bool megamol::compositing::ASSAO::reCreateMIPViewIfNeeded(
    std::shared_ptr<glowl::Texture2DView> current, std::shared_ptr<glowl::Texture2D> original, int mipViewSlice) {
    //glowl::TextureLayout current_layout = current->getTextureLayout();
    glowl::TextureLayout original_layout = original->getTextureLayout();

    // TODO: check if they are equal, i.e. if current is already textureview of original at mipviewslice
    /*if (current == original)
        return true;*/

    current->reload(*original, original->getTextureLayout(), mipViewSlice, 1, 0, 1);
    /*current = std::make_shared<glowl::Texture2DView>(
        id, *original, original->getTextureLayout(), mipViewSlice, 1, 0, 1);*/

    // not needed, glTextureView does this automatically
    /*int new_width = original_layout.width;
    int new_height = original_layout.height;

    for (int i = 0; i < mipViewSlice; ++i) {
        new_width = (new_width + 1) / 2;
        new_height = (new_height + 1) / 2;
    }
    new_width = std::max( new_width, 1 );
    new_height = std::max( new_height, 1 );

    current_layout.width = new_width;
    current_layout.height = new_height;*/

    //current->reload(current_layout, nullptr);

    // make to one function "makeTextureView" for error handling in texture2d.hpp
    /*current->deleteTexture();
    current->genTexture();
    glTextureView(
        current->getName(), GL_TEXTURE_2D, original->getName(), original->getInternalFormat(), mipViewSlice, 1, 0, 1);*/

    /*std::vector<float> check(new_width * new_height);
    for (int i = 0; i < check.size(); ++i) {
        check[i] = i;
    }
    glTextureSubImage2D(current->getName(), 0, 0, 0, new_width, new_height, original->getFormat(),
        original->getType(), check.data());*/

    // debug
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cout << "error after textureview\n";
    }
    /*
    current->bindTexture();
    int w, h;
    int miplevel = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, miplevel, GL_TEXTURE_WIDTH, &w);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, miplevel, GL_TEXTURE_HEIGHT, &h);*/

    return true;
}

bool megamol::compositing::ASSAO::equalLayoutsWithoutSize(const glowl::TextureLayout& lhs, const glowl::TextureLayout& rhs) {
    bool depth            = lhs.depth == rhs.depth;
    bool float_parameters = lhs.float_parameters == rhs.float_parameters;
    bool format           = lhs.format == rhs.format;
    //bool height           = lhs.height == rhs.height;
    bool internal_format  = lhs.internal_format == rhs.internal_format;
    bool int_parameters   = lhs.int_parameters == rhs.int_parameters;
    bool levels           = lhs.levels == rhs.levels;
    bool type             = lhs.type == rhs.type;
    //bool width            = lhs.width == rhs.width;

    return depth && float_parameters && format /*&& height*/ && internal_format && int_parameters && levels &&
           type /*&& width*/;
}

bool megamol::compositing::ASSAO::equalLayouts(
    const glowl::TextureLayout& lhs, const glowl::TextureLayout& rhs) {
    bool depth            = lhs.depth == rhs.depth;
    bool float_parameters = lhs.float_parameters == rhs.float_parameters;
    bool format           = lhs.format == rhs.format;
    bool height           = lhs.height == rhs.height;
    bool internal_format  = lhs.internal_format == rhs.internal_format;
    bool int_parameters   = lhs.int_parameters == rhs.int_parameters;
    bool levels           = lhs.levels == rhs.levels;
    bool type             = lhs.type == rhs.type;
    bool width            = lhs.width == rhs.width;

    return depth && float_parameters && format && height && internal_format && int_parameters && levels &&
           type && width;
}
