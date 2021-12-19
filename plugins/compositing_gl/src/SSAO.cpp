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
#include "SSAO.h"

#include <array>
#include <random>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "compositing_gl/CompositingCalls.h"
#include "mmcore_gl/utility/ShaderSourceFactory.h"

/////////////////////////////////////////////////////////////////////////
// CONSTANTS
/////////////////////////////////////////////////////////////////////////
#define MEGAMOL_ASSAO_MANUAL_MIPS

#define SSAO_MAX_TAPS 32
#define SSAO_MAX_REF_TAPS 512
#define SSAO_ADAPTIVE_TAP_BASE_COUNT 5
#define SSAO_ADAPTIVE_TAP_FLEXIBLE_COUNT (SSAO_MAX_TAPS - SSAO_ADAPTIVE_TAP_BASE_COUNT)
#define SSAODepth_MIP_LEVELS 4
/////////////////////////////////////////////////////////////////////////

/*
 * @megamol::compositing::SSAO::SSAO
 */
megamol::compositing::SSAO::SSAO()
    : core::Module()
    , m_version(0)
    , m_outputTexSlot("OutputTexture", "Gives access to resulting output texture")
    , m_normalsTexSlot("NormalTexture", "Connects the normals render target texture")
    , m_depthTexSlot("DepthTexture", "Connects the depth render target texture")
    , m_cameraSlot("Camera", "Connects a (copy of) camera state")
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
    , m_depthMipLevels(0)
    , m_inputs(nullptr)
    , m_maxBlurPassCount(6)
    , m_ssboConstants(nullptr)
    , m_settings()
    , m_psSSAOMode("SSAO", "Specifices which SSAO technqiue should be used: naive SSAO or ASSAO")
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
    , m_psSSAORadius("SSAO Radius", "Sets radius for SSAO")
    , m_psSSAOSampleCnt("SSAO Samples", "Sets the number of samples used SSAO")
    , m_settingsHaveChanged(false)
    , m_slotIsActive(false)
    , m_updateCausedByNormalSlotChange(false)
{
    this->m_outputTexSlot.SetCallback(CallTexture2D::ClassName(), "GetData", &SSAO::getDataCallback);
    this->m_outputTexSlot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &SSAO::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_outputTexSlot);

    this->m_normalsTexSlot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_normalsTexSlot);

    this->m_depthTexSlot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_depthTexSlot);

    this->m_cameraSlot.SetCompatibleCall<CallCameraDescription>();
    this->MakeSlotAvailable(&this->m_cameraSlot);

    this->m_psSSAOMode << new core::param::EnumParam(0);
    this->m_psSSAOMode.Param<core::param::EnumParam>()->SetTypePair(0, "ASSAO");
    this->m_psSSAOMode.Param<core::param::EnumParam>()->SetTypePair(1, "Naive");
    this->m_psSSAOMode.SetUpdateCallback(&SSAO::ssaoModeCallback);
    this->MakeSlotAvailable(&this->m_psSSAOMode);

    this->m_psSSAORadius << new megamol::core::param::FloatParam(0.5f, 0.0f);
    this->MakeSlotAvailable(&this->m_psSSAORadius);

    this->m_psSSAOSampleCnt << new megamol::core::param::IntParam(16, 0, 64);
    this->MakeSlotAvailable(&this->m_psSSAOSampleCnt);

    // settings
    this->m_psRadius << new core::param::FloatParam(1.2f, 0.f);
    this->m_psRadius.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psRadius.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psRadius);

    this->m_psShadowMultiplier << new core::param::FloatParam(1.f, 0.f, 5.f);
    this->m_psShadowMultiplier.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psShadowMultiplier.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psShadowMultiplier);

    this->m_psShadowPower << new core::param::FloatParam(1.5f, 0.5f, 5.f);
    this->m_psShadowPower.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psShadowPower.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psShadowPower);

    this->m_psShadowClamp << new core::param::FloatParam(0.98f, 0.f, 1.f);
    this->m_psShadowClamp.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psShadowClamp.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psShadowClamp);

    this->m_psHorizonAngleThreshold << new core::param::FloatParam(0.06f, 0.f, 0.2f);
    this->m_psHorizonAngleThreshold.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psHorizonAngleThreshold.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psHorizonAngleThreshold);

    this->m_psFadeOutFrom << new core::param::FloatParam(50.f, 0.f);
    this->m_psFadeOutFrom.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psFadeOutFrom.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psFadeOutFrom);

    this->m_psFadeOutTo << new core::param::FloatParam(300.f, 0.f);
    this->m_psFadeOutTo.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psFadeOutTo.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psFadeOutTo);

    // generally there are quality levels from -1 (lowest) to 3 (highest, adaptive), but 3 (adaptive) is not implemented yet
    this->m_psQualityLevel << new core::param::EnumParam(2);
    this->m_psQualityLevel.Param<core::param::EnumParam>()->SetTypePair(-1, "Lowest");
    this->m_psQualityLevel.Param<core::param::EnumParam>()->SetTypePair( 0, "Low");
    this->m_psQualityLevel.Param<core::param::EnumParam>()->SetTypePair( 1, "Medium");
    this->m_psQualityLevel.Param<core::param::EnumParam>()->SetTypePair( 2, "High");
    this->m_psQualityLevel.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psQualityLevel);

    this->m_psAdaptiveQualityLimit << new core::param::FloatParam(0.45f, 0.f, 1.f);
    this->m_psAdaptiveQualityLimit.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psAdaptiveQualityLimit.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psAdaptiveQualityLimit);

    this->m_psBlurPassCount << new core::param::IntParam(2, 0, 6);
    this->m_psBlurPassCount.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psBlurPassCount);

    this->m_psSharpness << new core::param::FloatParam(0.98f, 0.f, 1.f);
    this->m_psSharpness.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psSharpness.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psSharpness);

    this->m_psTemporalSupersamplingAngleOffset << new core::param::FloatParam(0.f, 0.f, 3.141592653589f);
    this->m_psTemporalSupersamplingAngleOffset.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psTemporalSupersamplingAngleOffset.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psTemporalSupersamplingAngleOffset);

    this->m_psTemporalSupersamplingRadiusOffset << new core::param::FloatParam(1.f, 0.f, 2.f);
    this->m_psTemporalSupersamplingRadiusOffset.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psTemporalSupersamplingRadiusOffset.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psTemporalSupersamplingRadiusOffset);

    this->m_psDetailShadowStrength << new core::param::FloatParam(0.5f, 0.f, 5.f);
    this->m_psDetailShadowStrength.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->m_psDetailShadowStrength.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->m_psDetailShadowStrength);
}


/*
 * @megamol::compositing::SSAO::ssaoModeCallback
 */
bool megamol::compositing::SSAO::ssaoModeCallback(core::param::ParamSlot& slot) {
    int mode = m_psSSAOMode.Param<core::param::EnumParam>()->Value();

    // assao
    if (mode == 0) {
        m_psSSAORadius.Param<core::param::FloatParam>()->SetGUIVisible(false);
        m_psSSAOSampleCnt.Param<core::param::IntParam>()->SetGUIVisible(false);

        m_psRadius.Param<core::param::FloatParam>()->SetGUIVisible(true);
        m_psShadowMultiplier.Param<core::param::FloatParam>()->SetGUIVisible(true);
        m_psShadowPower.Param<core::param::FloatParam>()->SetGUIVisible(true);
        m_psShadowClamp.Param<core::param::FloatParam>()->SetGUIVisible(true);
        m_psHorizonAngleThreshold.Param<core::param::FloatParam>()->SetGUIVisible(true);
        m_psFadeOutFrom.Param<core::param::FloatParam>()->SetGUIVisible(true);
        m_psFadeOutTo.Param<core::param::FloatParam>()->SetGUIVisible(true);
        m_psQualityLevel.Param<core::param::EnumParam>()->SetGUIVisible(true);
        m_psAdaptiveQualityLimit.Param<core::param::FloatParam>()->SetGUIVisible(true);
        m_psBlurPassCount.Param<core::param::IntParam>()->SetGUIVisible(true);
        m_psSharpness.Param<core::param::FloatParam>()->SetGUIVisible(true);
        m_psTemporalSupersamplingAngleOffset.Param<core::param::FloatParam>()->SetGUIVisible(true);
        m_psTemporalSupersamplingRadiusOffset.Param<core::param::FloatParam>()->SetGUIVisible(true);
        m_psDetailShadowStrength.Param<core::param::FloatParam>()->SetGUIVisible(true);
    }
    // naive
    else {
        m_psSSAORadius.Param<core::param::FloatParam>()->SetGUIVisible(true);
        m_psSSAOSampleCnt.Param<core::param::IntParam>()->SetGUIVisible(true);

        m_psRadius.Param<core::param::FloatParam>()->SetGUIVisible(false);
        m_psShadowMultiplier.Param<core::param::FloatParam>()->SetGUIVisible(false);
        m_psShadowPower.Param<core::param::FloatParam>()->SetGUIVisible(false);
        m_psShadowClamp.Param<core::param::FloatParam>()->SetGUIVisible(false);
        m_psHorizonAngleThreshold.Param<core::param::FloatParam>()->SetGUIVisible(false);
        m_psFadeOutFrom.Param<core::param::FloatParam>()->SetGUIVisible(false);
        m_psFadeOutTo.Param<core::param::FloatParam>()->SetGUIVisible(false);
        m_psQualityLevel.Param<core::param::EnumParam>()->SetGUIVisible(false);
        m_psAdaptiveQualityLimit.Param<core::param::FloatParam>()->SetGUIVisible(false);
        m_psBlurPassCount.Param<core::param::IntParam>()->SetGUIVisible(false);
        m_psSharpness.Param<core::param::FloatParam>()->SetGUIVisible(false);
        m_psTemporalSupersamplingAngleOffset.Param<core::param::FloatParam>()->SetGUIVisible(false);
        m_psTemporalSupersamplingRadiusOffset.Param<core::param::FloatParam>()->SetGUIVisible(false);
        m_psDetailShadowStrength.Param<core::param::FloatParam>()->SetGUIVisible(false);
    }

    return true;
}


/*
 * @megamol::compositing::SSAO::settingsCallback
 */
bool megamol::compositing::SSAO::settingsCallback(core::param::ParamSlot& slot) {
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


/*
 * @megamol::compositing::SSAO::~SSAO
 */
megamol::compositing::SSAO::~SSAO() { this->Release(); }


/*
 * @megamol::compositing::SSAO::create
 */
bool megamol::compositing::SSAO::create() {
    typedef megamol::core::utility::log::Log Log;

    try {
        {
            // create shader program
            m_prepareDepthsPrgm = std::make_unique<GLSLComputeShader>();
            m_prepareDepthsHalfPrgm = std::make_unique<GLSLComputeShader>();
            m_prepareDepthsAndNormalsPrgm = std::make_unique<GLSLComputeShader>();
            m_prepareDepthsAndNormalsHalfPrgm = std::make_unique<GLSLComputeShader>();
            m_prepareDepthMipPrgms.resize(SSAODepth_MIP_LEVELS - 1);
            for (auto& cs : m_prepareDepthMipPrgms) {
                cs = std::make_unique<GLSLComputeShader>();
            }
            for (auto& cs : m_generatePrgms) {
                cs = std::make_unique<GLSLComputeShader>();
            }
            m_smartBlurPrgm = std::make_unique<GLSLComputeShader>();
            m_smartBlurWidePrgm = std::make_unique<GLSLComputeShader>();
            m_applyPrgm = std::make_unique<GLSLComputeShader>();
            m_nonSmartBlurPrgm = std::make_unique<GLSLComputeShader>();
            m_nonSmartApplyPrgm = std::make_unique<GLSLComputeShader>();
            m_nonSmartHalfApplyPrgm = std::make_unique<GLSLComputeShader>();
            m_naiveSSAOPrgm = std::make_unique<GLSLComputeShader>();
            m_naiveSSAOBlurPrgm = std::make_unique<GLSLComputeShader>();

            vislib_gl::graphics::gl::ShaderSource csPrepareDepths;
            vislib_gl::graphics::gl::ShaderSource csPrepareDepthsHalf;
            vislib_gl::graphics::gl::ShaderSource csPrepareDepthsAndNormals;
            vislib_gl::graphics::gl::ShaderSource csPrepareDepthsAndNormalsHalf;
            std::vector<vislib_gl::graphics::gl::ShaderSource> csPrepareDepthMip(SSAODepth_MIP_LEVELS - 1);
            std::vector<vislib_gl::graphics::gl::ShaderSource> csGenerate(5);
            vislib_gl::graphics::gl::ShaderSource csSmartBlur;
            vislib_gl::graphics::gl::ShaderSource csSmartBlur_wide;
            vislib_gl::graphics::gl::ShaderSource csApply;
            vislib_gl::graphics::gl::ShaderSource csNonSmartBlur;
            vislib_gl::graphics::gl::ShaderSource csNonSmartApply;
            vislib_gl::graphics::gl::ShaderSource csNonSmartHalfApply;
            vislib_gl::graphics::gl::ShaderSource csNaiveSSAO;
            vislib_gl::graphics::gl::ShaderSource csNaiveSSAOBlur;

            Log::DefaultLog.WriteInfo("Compiling: Compositing::assao::CSPrepareDepths");
            auto ssf = std::make_shared<core_gl::utility::ShaderSourceFactory>(
                instance()->Configuration().ShaderDirectories());
            if (!ssf->MakeShaderSource("Compositing::assao::CSPrepareDepths", csPrepareDepths))
                return false;
            if (!m_prepareDepthsPrgm->Compile(csPrepareDepths.Code(), csPrepareDepths.Count()))
                return false;
            if (!m_prepareDepthsPrgm->Link())
                return false;

            Log::DefaultLog.WriteInfo("Compiling: Compositing::assao::CSPrepareDepthsHalf");
            if (!ssf->MakeShaderSource("Compositing::assao::CSPrepareDepthsHalf", csPrepareDepthsHalf))
                return false;
            if (!m_prepareDepthsHalfPrgm->Compile(csPrepareDepthsHalf.Code(), csPrepareDepthsHalf.Count()))
                return false;
            if (!m_prepareDepthsHalfPrgm->Link())
                return false;

            Log::DefaultLog.WriteInfo("Compiling: Compositing::assao::CSPrepareDepthsAndNormals");
            if (!ssf->MakeShaderSource("Compositing::assao::CSPrepareDepthsAndNormals", csPrepareDepthsAndNormals))
                return false;
            if (!m_prepareDepthsAndNormalsPrgm->Compile(
                    csPrepareDepthsAndNormals.Code(), csPrepareDepthsAndNormals.Count()))
                return false;
            if (!m_prepareDepthsAndNormalsPrgm->Link())
                return false;

            Log::DefaultLog.WriteInfo("Compiling: Compositing::assao::CSPrepareDepthsAndNormalsHalf");
            if (!ssf->MakeShaderSource(
                    "Compositing::assao::CSPrepareDepthsAndNormalsHalf", csPrepareDepthsAndNormalsHalf))
                return false;
            if (!m_prepareDepthsAndNormalsHalfPrgm->Compile(
                    csPrepareDepthsAndNormalsHalf.Code(), csPrepareDepthsAndNormalsHalf.Count()))
                return false;
            if (!m_prepareDepthsAndNormalsHalfPrgm->Link())
                return false;

            for (int i = 0; i < SSAODepth_MIP_LEVELS - 1; ++i) {
                Log::DefaultLog.WriteInfo("Compiling: Compositing::assao::CSPrepareDepthMip%i", i+1);
                std::string identifier = "Compositing::assao::CSPrepareDepthMip" + std::to_string(i + 1);
                if (!ssf->MakeShaderSource(identifier.c_str(), csPrepareDepthMip[i]))
                    return false;
                if (!m_prepareDepthMipPrgms[i]->Compile(csPrepareDepthMip[i].Code(), csPrepareDepthMip[i].Count()))
                    return false;
                if (!m_prepareDepthMipPrgms[i]->Link())
                    return false;
            }

            // one less than cs_generate.size() because the adaptive quality level is not implemented (yet)
            for (int i = 0; i < 4; ++i) {
                Log::DefaultLog.WriteInfo("Compiling: Compositing::assao::CSGenerateQ%i", i);
                std::string identifier = "Compositing::assao::CSGenerateQ" + std::to_string(i);
                if (i >= 4)
                    identifier = "Compositing::assao::CSGenerateQ3Base";

                if (!ssf->MakeShaderSource(identifier.c_str(), csGenerate[i]))
                    return false;
                if (!m_generatePrgms[i]->Compile(csGenerate[i].Code(), csGenerate[i].Count()))
                    return false;
                if (!m_generatePrgms[i]->Link())
                    return false;
            }

            Log::DefaultLog.WriteInfo("Compiling: Compositing::assao::CSSmartBlur");
            if (!ssf->MakeShaderSource("Compositing::assao::CSSmartBlur", csSmartBlur))
                return false;
            if (!m_smartBlurPrgm->Compile(csSmartBlur.Code(), csSmartBlur.Count()))
                return false;
            if (!m_smartBlurPrgm->Link())
                return false;

            Log::DefaultLog.WriteInfo("Compiling: Compositing::assao::CSSmartBlurWide");
            if (!ssf->MakeShaderSource("Compositing::assao::CSSmartBlurWide", csSmartBlur_wide))
                return false;
            if (!m_smartBlurWidePrgm->Compile(csSmartBlur_wide.Code(), csSmartBlur_wide.Count()))
                return false;
            if (!m_smartBlurWidePrgm->Link())
                return false;

            Log::DefaultLog.WriteInfo("Compiling: Compositing::assao::CSNonSmartBlur");
            if (!ssf->MakeShaderSource("Compositing::assao::CSNonSmartBlur", csNonSmartBlur))
                return false;
            if (!m_nonSmartBlurPrgm->Compile(csNonSmartBlur.Code(), csNonSmartBlur.Count()))
                return false;
            if (!m_nonSmartBlurPrgm->Link())
                return false;

            Log::DefaultLog.WriteInfo("Compiling: Compositing::assao::CSApply");
            if (!ssf->MakeShaderSource("Compositing::assao::CSApply", csApply))
                return false;
            if (!m_applyPrgm->Compile(csApply.Code(), csApply.Count()))
                return false;
            if (!m_applyPrgm->Link())
                return false;

            Log::DefaultLog.WriteInfo("Compiling: Compositing::assao::CSNonSmartApply");
            if (!ssf->MakeShaderSource("Compositing::assao::CSNonSmartApply", csNonSmartApply))
                return false;
            if (!m_nonSmartApplyPrgm->Compile(csNonSmartApply.Code(), csNonSmartApply.Count()))
                return false;
            if (!m_nonSmartApplyPrgm->Link())
                return false;

            Log::DefaultLog.WriteInfo("Compiling: Compositing::assao::CSNonSmartHalfApply");
            if (!ssf->MakeShaderSource("Compositing::assao::CSNonSmartHalfApply", csNonSmartHalfApply))
                return false;
            if (!m_nonSmartHalfApplyPrgm->Compile(csNonSmartHalfApply.Code(), csNonSmartHalfApply.Count()))
                return false;
            if (!m_nonSmartHalfApplyPrgm->Link())
                return false;

            Log::DefaultLog.WriteInfo("Compiling: Compositing::ssao");
            if (!ssf->MakeShaderSource("Compositing::ssao", csNaiveSSAO))
                return false;
            if (!m_naiveSSAOPrgm->Compile(csNaiveSSAO.Code(), csNaiveSSAO.Count())) return false;
            if (!m_naiveSSAOPrgm->Link()) return false;

            Log::DefaultLog.WriteInfo("Compiling: Compositing::simpleBlur");
            if (!ssf->MakeShaderSource("Compositing::blur", csNaiveSSAOBlur))
                return false;
            if (!m_naiveSSAOBlurPrgm->Compile(csNaiveSSAOBlur.Code(), csNaiveSSAOBlur.Count())) return false;
            if (!m_naiveSSAOBlurPrgm->Link()) return false;

            Log::DefaultLog.WriteInfo("Done compiling (a)ssao shaders.");
        }

    } catch (vislib_gl::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Unable to compile shader (@%s): %s\n",
            vislib_gl::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
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
    m_AOResultLayout = glowl::TextureLayout(GL_RG8, 1, 1, 1, GL_RG, GL_FLOAT, 1);
    m_normalLayout = glowl::TextureLayout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_HALF_FLOAT, 1);
    m_halfDepths[0] = std::make_shared<glowl::Texture2D>("m_halfDepths0", m_depthBufferViewspaceLinearLayout, nullptr);
    m_halfDepths[1] = std::make_shared<glowl::Texture2D>("m_halfDepths1", m_depthBufferViewspaceLinearLayout, nullptr);
    m_halfDepths[2] = std::make_shared<glowl::Texture2D>("m_halfDepths2", m_depthBufferViewspaceLinearLayout, nullptr);
    m_halfDepths[3] = std::make_shared<glowl::Texture2D>("m_halfDepths3", m_depthBufferViewspaceLinearLayout, nullptr);
    m_halfDepthsMipViews.resize(4);
    for (int j = 0; j < 4; ++j) {
        m_halfDepthsMipViews[j].resize(SSAODepth_MIP_LEVELS);
        for (int i = 0; i < m_halfDepthsMipViews[j].size(); ++i) {
            m_halfDepthsMipViews[j][i] =
                std::make_shared<glowl::Texture2DView>("m_halfDepthsMipViews" + std::to_string(i), *m_halfDepths[j],
                    m_depthBufferViewspaceLinearLayout, 0, 1, 0, 1);
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
    m_normals = std::make_shared<glowl::Texture2D>("m_normals", m_normalLayout, nullptr);

    m_inputs = std::make_shared<ASSAO_Inputs>();

    m_ssboConstants = std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    std::vector<std::pair<GLenum, GLint>> intParams = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST},
        {GL_TEXTURE_MAG_FILTER, GL_NEAREST}, {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE}};

    m_samplerStatePointClamp = std::make_shared<glowl::Sampler>("samplerStatePointClamp", intParams);

    intParams.clear();
    intParams = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST}, {GL_TEXTURE_MAG_FILTER, GL_NEAREST},
        {GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT}, {GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT}};

    m_samplerStatePointMirror = std::make_shared<glowl::Sampler>("samplerStatePointMirror", intParams);

    intParams.clear();
    intParams = {{GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR}, {GL_TEXTURE_MAG_FILTER, GL_LINEAR},
        {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}};

    m_samplerStateLinearClamp = std::make_shared<glowl::Sampler>("samplerStateLinearClamp", intParams);

    intParams.clear();
    intParams = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST}, {GL_TEXTURE_MAG_FILTER, GL_NEAREST},
        {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}};

    m_samplerStateViewspaceDepthTap = std::make_shared<glowl::Sampler>("samplerStateViewspaceDepthTap", intParams);


    // naive ssao stuff
    m_intermediateTx2D = std::make_shared<glowl::Texture2D>("screenspace_effect_intermediate", m_normalLayout, nullptr);

    // quick 'n dirty from https://learnopengl.com/Advanced-Lighting/SSAO
    std::uniform_real_distribution<float> randomFloats(0.0, 1.0); // random floats between 0.0 - 1.0
    std::default_random_engine generator;

    std::vector<float> ssaoKernel;
    for (unsigned int i = 0; i < 64; ++i) {
        glm::vec3 sample(
            randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, randomFloats(generator));
        sample = glm::normalize(sample);
        sample *= randomFloats(generator);
        float scale = (float)i / 64.0;
        ssaoKernel.push_back(sample.x);
        ssaoKernel.push_back(sample.y);
        ssaoKernel.push_back(sample.z);
    }

    m_SSAOSamples = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, ssaoKernel, GL_DYNAMIC_DRAW);

    std::vector<glm::vec3> ssaoNoise;
    for (unsigned int i = 0; i < 16; i++) {
        glm::vec3 noise(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, 0.0f);
        ssaoNoise.push_back(noise);
    }

    glowl::TextureLayout tx_layout2(GL_RGB32F, 4, 4, 1, GL_RGB, GL_FLOAT, 1);
    m_SSAOKernelRotTx2D = std::make_shared<glowl::Texture2D>("ssao_kernel_rotation", tx_layout2, ssaoNoise.data());

    return true;
}


/*
 * @megamol::compositing::SSAO::release
 */
void megamol::compositing::SSAO::release() {}


/*
 * @megamol::compositing::SSAO::getDataCallback
 */
bool megamol::compositing::SSAO::getDataCallback(core::Call& caller) {
    auto lhsTc = dynamic_cast<CallTexture2D*>(&caller);
    auto callNormal = m_normalsTexSlot.CallAs<CallTexture2D>();
    auto callDepth = m_depthTexSlot.CallAs<CallTexture2D>();
    auto callCamera = m_cameraSlot.CallAs<CallCamera>();

    if (lhsTc == NULL)
        return false;

    if ((callDepth != NULL) && (callCamera != NULL)) {

        bool generateNormals = false;
        if (callNormal == NULL) {
            if (m_slotIsActive) {
                m_slotIsActive = false;
                m_updateCausedByNormalSlotChange = true;
            }
            generateNormals = true;
        } else {
            if (!m_slotIsActive) {
                m_slotIsActive = true;
                m_updateCausedByNormalSlotChange = true;
            }
        }

        if (!generateNormals && !(*callNormal)(0))
            return false;

        if (!(*callDepth)(0))
            return false;

        if (!(*callCamera)(0))
            return false;

        // something has changed in the neath...
        bool normalUpdate = false;

        if (!generateNormals) {
            normalUpdate = callNormal->hasUpdate();
        }
        bool depthUpdate = callDepth->hasUpdate();
        bool cameraUpdate = callCamera->hasUpdate();

        bool somethingHasChanged =
            (callNormal != NULL ? normalUpdate : false) || (callDepth != NULL ? depthUpdate : false) ||
            (callCamera != NULL ? cameraUpdate : false) || m_updateCausedByNormalSlotChange || m_settingsHaveChanged;

        if (somethingHasChanged) {
            ++m_version;

            std::function<void(std::shared_ptr<glowl::Texture2D> src, std::shared_ptr<glowl::Texture2D> tgt)>
            setupOutputTexture = [](std::shared_ptr<glowl::Texture2D> src, std::shared_ptr<glowl::Texture2D> tgt) {
                // set output texture size to primary input texture
                std::array<float, 2> texture_res = {
                    static_cast<float>(src->getWidth()), static_cast<float>(src->getHeight())};

                if (tgt->getWidth() != std::get<0>(texture_res) || tgt->getHeight() != std::get<1>(texture_res)) {
                    glowl::TextureLayout tx_layout(
                        GL_RGBA16F, std::get<0>(texture_res), std::get<1>(texture_res), 1, GL_RGBA, GL_HALF_FLOAT, 1);
                    tgt->reload(tx_layout, nullptr);
                }
            };

            if (callNormal == NULL && m_slotIsActive)
                return false;
            if (callDepth == NULL)
                return false;
            if (callCamera == NULL)
                return false;

            std::array<int, 2> txResNormal;
            if (!generateNormals) {
                m_normals = callNormal->getData();
                txResNormal = {(int)m_normals->getWidth(), (int)m_normals->getHeight()};
            }

            auto depthTx2D = callDepth->getData();
            std::array<int, 2> txResDepth = {(int)depthTx2D->getWidth(), (int)depthTx2D->getHeight()};

            setupOutputTexture(depthTx2D, m_finalOutput);

            // obtain camera information
            core::view::Camera cam = callCamera->getData();
            glm::mat4 viewMx = cam.getViewMatrix();
            glm::mat4 projMx = cam.getProjectionMatrix();

            int ssaoMode = m_psSSAOMode.Param<core::param::EnumParam>()->Value();

            // assao
            if (ssaoMode == 0) {

                if (normalUpdate || depthUpdate || m_settingsHaveChanged || m_updateCausedByNormalSlotChange) {

                    // assuming a full resolution depth buffer!
                    m_inputs->ViewportWidth = txResDepth[0];
                    m_inputs->ViewportHeight = txResDepth[1];
                    m_inputs->GenerateNormals = generateNormals;
                    m_inputs->ProjectionMatrix = projMx;
                    m_inputs->ViewMatrix = viewMx;

                    updateTextures(m_inputs);

                    updateConstants(m_settings, m_inputs, 0);
                }

                {
                    prepareDepths(m_settings, m_inputs, depthTx2D, m_normals);

                    generateSSAO(m_settings, m_inputs, false, m_normals);

                    std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTextures = {
                        {m_finalOutput, 0}
                    };

                    // Apply
                    {
                        std::vector<TextureArraySamplerTuple> inputFinals = {
                            {m_finalResults, "g_FinalSSAOLinearClamp", m_samplerStateLinearClamp} };

                        if (m_settings.QualityLevel < 0)
                            fullscreenPassDraw <TextureSamplerTuple, glowl::Texture2D>(m_nonSmartHalfApplyPrgm, {}, outputTextures, true, inputFinals);
                        else if (m_settings.QualityLevel == 0)
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                                m_nonSmartApplyPrgm, {}, outputTextures, true, inputFinals);
                        else {
                            inputFinals.push_back({ m_finalResults, "g_FinalSSAO", nullptr });
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                                m_applyPrgm, {}, outputTextures, true, inputFinals);
                        }
                    }
                }
            }
            // naive
            else {
                setupOutputTexture(depthTx2D, m_intermediateTx2D);

                m_naiveSSAOPrgm->Enable();

                m_SSAOSamples->bind(1);

                glUniform1f(m_naiveSSAOPrgm->ParameterLocation("radius"), m_psSSAORadius.Param<core::param::FloatParam>()->Value());
                glUniform1i(m_naiveSSAOPrgm->ParameterLocation("sample_cnt"), m_psSSAOSampleCnt.Param<core::param::IntParam>()->Value());

                glActiveTexture(GL_TEXTURE0);
                m_normals->bindTexture();
                glUniform1i(m_naiveSSAOPrgm->ParameterLocation("normal_tx2D"), 0);
                glActiveTexture(GL_TEXTURE1);
                depthTx2D->bindTexture();
                glUniform1i(m_naiveSSAOPrgm->ParameterLocation("depth_tx2D"), 1);
                glActiveTexture(GL_TEXTURE2);
                m_SSAOKernelRotTx2D->bindTexture();
                glUniform1i(m_naiveSSAOPrgm->ParameterLocation("noise_tx2D"), 2);

                auto invViewMx = glm::inverse(viewMx);
                auto invProjMx = glm::inverse(projMx);
                glUniformMatrix4fv(m_naiveSSAOPrgm->ParameterLocation("inv_view_mx"), 1, GL_FALSE, glm::value_ptr(invViewMx));
                glUniformMatrix4fv(m_naiveSSAOPrgm->ParameterLocation("inv_proj_mx"), 1, GL_FALSE, glm::value_ptr(invProjMx));

                glUniformMatrix4fv(m_naiveSSAOPrgm->ParameterLocation("view_mx"), 1, GL_FALSE, glm::value_ptr(viewMx));
                glUniformMatrix4fv(m_naiveSSAOPrgm->ParameterLocation("proj_mx"), 1, GL_FALSE, glm::value_ptr(projMx));

                m_intermediateTx2D->bindImage(0, GL_WRITE_ONLY);

                m_naiveSSAOPrgm->Dispatch(static_cast<int>(std::ceil(m_finalOutput->getWidth() / 8.0f)),
                    static_cast<int>(std::ceil(m_finalOutput->getHeight() / 8.0f)), 1);

                m_naiveSSAOPrgm->Disable();

                glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

                m_naiveSSAOBlurPrgm->Enable();

                glActiveTexture(GL_TEXTURE0);
                m_intermediateTx2D->bindTexture();
                // test with m_NaiveSSAOPrgm, since it was (falsely) used the entire time
                glUniform1i(m_naiveSSAOBlurPrgm->ParameterLocation("src_tx2D"), 0);

                m_finalOutput->bindImage(0, GL_WRITE_ONLY);

                m_naiveSSAOBlurPrgm->Dispatch(static_cast<int>(std::ceil(m_finalOutput->getWidth() / 8.0f)),
                    static_cast<int>(std::ceil(m_finalOutput->getHeight() / 8.0f)), 1);

                m_naiveSSAOBlurPrgm->Disable();
            }
        }
    }

    if (lhsTc->version() < m_version) {
        m_settingsHaveChanged = false;
        m_updateCausedByNormalSlotChange = false;
        lhsTc->setData(m_finalOutput, m_version);
    }

    return true;
}


/*
 * @megamol::compositing::SSAO::prepareDepths
 */
void megamol::compositing::SSAO::prepareDepths(
    const ASSAO_Settings& settings,
    const std::shared_ptr<ASSAO_Inputs> inputs,
    std::shared_ptr<glowl::Texture2D> depthTexture,
    std::shared_ptr<glowl::Texture2D> normalTexture)
{
    bool generateNormals = inputs->GenerateNormals;

    std::vector<TextureSamplerTuple> inputTextures(1);
    inputTextures[0] = {depthTexture, (std::string) "g_DepthSource", nullptr};


    std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputFourDepths = {
        {m_halfDepths[0], 0}, {m_halfDepths[1], 1}, {m_halfDepths[2], 2}, {m_halfDepths[3], 3}};
    std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTwoDepths = {
        {m_halfDepths[0], 0}, {m_halfDepths[3], 3}};

    if (!generateNormals) {
        if (settings.QualityLevel < 0) {
            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                m_prepareDepthsHalfPrgm, inputTextures, outputTwoDepths);
        } else {
            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                m_prepareDepthsPrgm, inputTextures, outputFourDepths);
        }
    } else {
        inputTextures.push_back({depthTexture, (std::string) "g_DepthSourcePointClamp", m_samplerStatePointClamp});

        if (settings.QualityLevel < 0) {
            outputTwoDepths.push_back({normalTexture, 4});
            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                m_prepareDepthsAndNormalsHalfPrgm, inputTextures, outputTwoDepths);
        } else {
            outputFourDepths.push_back({normalTexture, 4});
            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                m_prepareDepthsAndNormalsPrgm, inputTextures, outputFourDepths);
        }
    }

    // only do mipmaps for higher quality levels (not beneficial on quality level 1, and detrimental on quality level 0)
    if (settings.QualityLevel > 1) {

#ifdef MEGAMOL_ASSAO_MANUAL_MIPS
        for (int i = 1; i < m_depthMipLevels; ++i) {
            std::vector<TextureViewSamplerTuple> inputFourDepthMipsM1 = {
                {m_halfDepthsMipViews[0][i - 1LL], (std::string) "g_ViewspaceDepthSource", nullptr},
                {m_halfDepthsMipViews[1][i - 1LL], (std::string) "g_ViewspaceDepthSource1", nullptr},
                {m_halfDepthsMipViews[2][i - 1LL], (std::string) "g_ViewspaceDepthSource2", nullptr},
                {m_halfDepthsMipViews[3][i - 1LL], (std::string) "g_ViewspaceDepthSource3", nullptr}};

            std::vector<std::pair<std::shared_ptr<glowl::Texture2DView>, GLuint>> outputFourDepthMips = {
                {m_halfDepthsMipViews[0][i], 0}, {m_halfDepthsMipViews[1][i], 1}, {m_halfDepthsMipViews[2][i], 2},
                {m_halfDepthsMipViews[3][i], 3}};

            fullscreenPassDraw<TextureViewSamplerTuple, glowl::Texture2DView>(
                m_prepareDepthMipPrgms[i - 1LL], inputFourDepthMipsM1, outputFourDepthMips);
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


/*
 * @megamol::compositing::SSAO::generateSSAO
 */
void megamol::compositing::SSAO::generateSSAO(
    const ASSAO_Settings& settings,
    const std::shared_ptr<ASSAO_Inputs> inputs,
    bool adaptiveBasePass,
    std::shared_ptr<glowl::Texture2D> normalTexture)
{

    // omitted viewport and scissor code from intel here

    if (adaptiveBasePass) {
        assert(settings.QualityLevel == 3);
    }

    int passCount = 4;

    for (int pass = 0; pass < passCount; ++pass) {
        if ((settings.QualityLevel < 0) && ((pass == 1) || (pass == 2)))
            continue;

        int blurPasses = std::min(settings.BlurPassCount, m_maxBlurPassCount);

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
            inputTextures[2] = {
                m_halfDepths[pass], "g_ViewSpaceDepthSourceDepthTapSampler", m_samplerStateViewspaceDepthTap};

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
                    m_generatePrgms[shaderIndex], inputTextures, outputTextures);
            } else {
                std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTextures = {
                    {m_pingPongHalfResultA, binding}};

                fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                    m_generatePrgms[shaderIndex], inputTextures, outputTextures);
            }
        }

        // Blur
        if (blurPasses > 0) {
            int wideBlursRemaining = std::max(0, blurPasses - 2);

            for (int i = 0; i < blurPasses; ++i)
            {
                GLuint binding = 0;

                std::vector<TextureSamplerTuple> inputTextures = {
                    {m_pingPongHalfResultA, "g_BlurInput", m_samplerStatePointMirror}};

                std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> intermediateOutputTexs = {
                    {m_pingPongHalfResultB, binding}};

                std::vector<std::pair<std::shared_ptr<glowl::Texture2DView>, GLuint>> finalOutputTexs = {
                    {m_finalResultsArrayViews[pass], binding}};

                if (settings.QualityLevel > 0)
                {
                    if (wideBlursRemaining > 0)
                    {
                        if (i == blurPasses - 1)
                        {
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2DView>(
                                m_smartBlurWidePrgm, inputTextures, finalOutputTexs);
                        } else
                        {
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                                m_smartBlurWidePrgm, inputTextures, intermediateOutputTexs);
                        }

                        wideBlursRemaining--;
                    } else
                    {
                        if (i == blurPasses - 1)
                        {
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2DView>(
                                m_smartBlurPrgm, inputTextures, finalOutputTexs);
                        } else
                        {
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                                m_smartBlurPrgm, inputTextures, intermediateOutputTexs);
                        }
                    }
                } else
                {
                    std::get<2>(inputTextures[0]) = m_samplerStateLinearClamp;

                    if (i == blurPasses - 1)
                    {
                        fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2DView>(
                            m_nonSmartBlurPrgm, inputTextures, finalOutputTexs);
                    } else
                    {
                        fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                            m_nonSmartBlurPrgm, inputTextures, intermediateOutputTexs);
                    }
                }

                std::swap(m_pingPongHalfResultA, m_pingPongHalfResultB);
            }
        }
    }
}


/*
 * @megamol::compositing::SSAO::getMetaDataCallback
 */
bool megamol::compositing::SSAO::getMetaDataCallback(core::Call& caller) { return true; }


/*
 * @megamol::compositing::SSAO::updateTextures
 */
void megamol::compositing::SSAO::updateTextures(
    const std::shared_ptr<ASSAO_Inputs> inputs)
{
    int width = inputs->ViewportWidth;
    int height = inputs->ViewportHeight;

    bool needsUpdate = (m_size.x != width) || (m_size.y != height);

    m_size.x = width;
    m_size.y = height;
    m_halfSize.x = (width + 1) / 2;
    m_halfSize.y = (height + 1) / 2;
    m_quarterSize.x = (m_halfSize.x + 1) / 2;
    m_quarterSize.y = (m_halfSize.y + 1) / 2;

    if (!needsUpdate)
        return;

    int blurEnlarge =
        m_maxBlurPassCount + std::max(0, m_maxBlurPassCount - 2); // +1 for max normal blurs, +2 for wide blurs

    float totalSizeInMB = 0.f;

    m_depthMipLevels = SSAODepth_MIP_LEVELS;

    for (int i = 0; i < 4; i++) {
        if (reCreateIfNeeded(m_halfDepths[i], m_halfSize, m_depthBufferViewspaceLinearLayout, true)) {

#ifdef MEGAMOL_ASSAO_MANUAL_MIPS
            for (int j = 0; j < m_depthMipLevels; j++) {
                reCreateMIPViewIfNeeded(m_halfDepthsMipViews[i][j], m_halfDepths[i], j);
            }
#endif
        }
    }

    reCreateIfNeeded(m_pingPongHalfResultA, m_halfSize, m_AOResultLayout);
    reCreateIfNeeded(m_pingPongHalfResultB, m_halfSize, m_AOResultLayout);
    reCreateIfNeeded(m_finalResults, m_halfSize, m_AOResultLayout);

    for (int i = 0; i < 4; ++i) {
        reCreateArrayIfNeeded(m_finalResultsArrayViews[i], m_finalResults, m_halfSize, i);
    }

    if (inputs->GenerateNormals) {
        reCreateIfNeeded(m_normals, m_size, m_normalLayout);
    }
}


/*
 * @megamol::compositing::SSAO::updateConstants
 */
void megamol::compositing::SSAO::updateConstants(
    const ASSAO_Settings& settings,
    const std::shared_ptr<ASSAO_Inputs> inputs,
    int pass)
{
    bool generateNormals = inputs->GenerateNormals;

    // update constants
    ASSAO_Constants& consts = m_constants; // = *((ASSAOConstants*) mappedResource.pData);

    const glm::mat4& proj = inputs->ProjectionMatrix;

    consts.ViewportPixelSize = glm::vec2(1.0f / (float)m_size.x, 1.0f / (float)m_size.y);
    consts.HalfViewportPixelSize = glm::vec2(1.0f / (float)m_halfSize.x, 1.0f / (float)m_halfSize.y);

    consts.Viewport2xPixelSize = glm::vec2(consts.ViewportPixelSize.x * 2.0f, consts.ViewportPixelSize.y * 2.0f);
    consts.Viewport2xPixelSize_x_025 =
        glm::vec2(consts.Viewport2xPixelSize.x * 0.25f, consts.Viewport2xPixelSize.y * 0.25f);

    // requires proj matrix to be in column-major order
    float depthLinearizeMul =
        proj[3][2]; // float depthLinearizeMul = -( 2.0 * clipFar * clipNear ) / ( clipFar - clipNear );
    float depthLinearizeAdd = proj[2][2]; // float depthLinearizeAdd = -(clipFar + clipNear) / ( clipFar - clipNear );

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
    consts.DepthPrecisionOffsetMod = 0.9992f;

    // consts.RadiusDistanceScalingFunctionPow     = 1.0f - Clamp( settings.RadiusDistanceScalingFunction,
    // 0.0f, 1.0f );

    int lastHalfDepthMipX = m_halfDepthsMipViews[0][SSAODepth_MIP_LEVELS - 1]->getWidth();
    int lastHalfDepthMipY = m_halfDepthsMipViews[0][SSAODepth_MIP_LEVELS - 1]->getHeight();

    // used to get average load per pixel; 9.0 is there to compensate for only doing every 9th InterlockedAdd in
    // PSPostprocessImportanceMapB for performance reasons
    consts.LoadCounterAvgDiv = 9.0f / (float)(m_quarterSize.x * m_quarterSize.y * 255.0);

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
    consts.QuarterResPixelSize = glm::vec2(1.0f / (float)m_quarterSize.x, 1.0f / (float)m_quarterSize.y);

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
        float angle0 = ((float)a + (float)b / (float)subPassCount) * (3.1415926535897932384626433832795f) * 0.5f;
        angle0 += additionalAngleOffset;

        ca = ::cosf(angle0);
        sa = ::sinf(angle0);

        float scale = 1.0f + (a - 1.5f + (b - (subPassCount - 1.0f) * 0.5f) / (float)subPassCount) * 0.07f;
        scale *= additionalRadiusScale;

        // all values are within [-1, 1]
        consts.PatternRotScaleMatrices[subPass] = glm::vec4(scale * ca, scale * -sa, -scale * sa, -scale * ca);
    }

    // not used by megamol since we transform normals differently
    if (!generateNormals) {
        consts.NormalsUnpackMul = inputs->NormalsUnpackMul;
        consts.NormalsUnpackAdd = inputs->NormalsUnpackAdd;
        consts.TransformNormalsToViewSpace = 1;
    } else {
        consts.TransformNormalsToViewSpace = 0;
        consts.NormalsUnpackMul = 2.0f;
        consts.NormalsUnpackAdd = -1.0f;
    }
    consts.DetailAOStrength = settings.DetailShadowStrength;

    consts.ViewMX = inputs->ViewMatrix;

    // probably do something with the ssbo? but could also just be done at this point
    m_ssboConstants->rebuffer(&m_constants, sizeof(m_constants));
}


/*
 * @megamol::compositing::SSAO::reCreateIfNeeded
 */
bool megamol::compositing::SSAO::reCreateIfNeeded(
    std::shared_ptr<glowl::Texture2D> tex,
    glm::ivec2 size, const glowl::TextureLayout& ly,
    bool generateMipMaps)
{
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
            desc.levels = SSAODepth_MIP_LEVELS;
            tex->reload(desc, nullptr, true, true);
        } else
            tex->reload(desc, nullptr);
    }

    return true;
}


/*
 * @megamol::compositing::SSAO::reCreateIfNeeded
 */
bool megamol::compositing::SSAO::reCreateIfNeeded(
    std::shared_ptr<glowl::Texture2DArray> tex,
    glm::ivec2 size,
    const glowl::TextureLayout& ly)
{
    if ((size.x == 0) || (size.y == 0)) {

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


/*
 * @megamol::compositing::SSAO::reCreateArrayIfNeeded
 */
bool megamol::compositing::SSAO::reCreateArrayIfNeeded(
    std::shared_ptr<glowl::Texture2DView> tex,
    std::shared_ptr<glowl::Texture2DArray> original,
    glm::ivec2 size,
    int arraySlice)
{
    if ((size.x == 0) || (size.y == 0)) {

    } else {
        if (tex != nullptr && original != nullptr) {
            glowl::TextureLayout desc = tex->getTextureLayout();
            glowl::TextureLayout originalDesc = original->getTextureLayout();
            if (equalLayouts(desc, originalDesc))
                return false;
        }

        tex->reload(*original, original->getTextureLayout(), 0, 1, arraySlice, 1);
    }

    return true;
}


/*
 * @megamol::compositing::SSAO::reCreateMIPViewIfNeeded
 */
bool megamol::compositing::SSAO::reCreateMIPViewIfNeeded(
    std::shared_ptr<glowl::Texture2DView> current,
    std::shared_ptr<glowl::Texture2D> original,
    int mipViewSlice)
{

    if (current != nullptr && original != nullptr) {
        glowl::TextureLayout desc = current->getTextureLayout();
        glowl::TextureLayout originalDesc = original->getTextureLayout();
        if (equalLayouts(desc, originalDesc))
            return false;

        current->reload(*original, original->getTextureLayout(), mipViewSlice, 1, 0, 1);
    }

    return true;
}


/*
 * @megamol::compositing::SSAO::equalLayoutsWithoutSize
 */
bool megamol::compositing::SSAO::equalLayoutsWithoutSize(
    const glowl::TextureLayout& lhs,
    const glowl::TextureLayout& rhs)
{
    bool depth = lhs.depth == rhs.depth;
    bool float_parameters = lhs.float_parameters == rhs.float_parameters;
    bool format = lhs.format == rhs.format;
    bool internal_format = lhs.internal_format == rhs.internal_format;
    bool int_parameters = lhs.int_parameters == rhs.int_parameters;
    bool levels = lhs.levels == rhs.levels;
    bool type = lhs.type == rhs.type;

    return depth && float_parameters && format && internal_format && int_parameters && levels && type;
}


/*
 * @megamol::compositing::SSAO::equalLayouts
 */
bool megamol::compositing::SSAO::equalLayouts(
    const glowl::TextureLayout& lhs,
    const glowl::TextureLayout& rhs)
{
    bool depth            = lhs.depth == rhs.depth;
    bool float_parameters = lhs.float_parameters == rhs.float_parameters;
    bool format = lhs.format == rhs.format;
    bool height = lhs.height == rhs.height;
    bool internal_format = lhs.internal_format == rhs.internal_format;
    bool int_parameters = lhs.int_parameters == rhs.int_parameters;
    bool levels = lhs.levels == rhs.levels;
    bool type = lhs.type == rhs.type;
    bool width = lhs.width == rhs.width;

    return depth && float_parameters && format && height && internal_format && int_parameters && levels && type &&
           width;
}
