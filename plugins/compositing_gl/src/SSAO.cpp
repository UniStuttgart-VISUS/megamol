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

#include "SSAO.h"
#include "stdafx.h"

#include <array>
#include <random>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "compositing_gl/CompositingCalls.h"

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
        , version_(0)
        , output_tex_slot_("OutputTexture", "Gives access to resulting output texture")
        , normals_tex_slot_("NormalTexture", "Connects the normals render target texture")
        , depth_tex_slot_("DepthTexture", "Connects the depth render target texture")
        , camera_slot_("Camera", "Connects a (copy of) camera state")
        , half_depths_{nullptr, nullptr, nullptr, nullptr}
        , half_depths_mip_views_{}
        , ping_pong_half_result_a_(nullptr)
        , ping_pong_half_result_b_(nullptr)
        , final_results_(nullptr)
        , final_results_array_views_{nullptr, nullptr, nullptr, nullptr}
        , normals_(nullptr)
        , final_output_(nullptr)
        , sampler_state_point_clamp_()
        , sampler_state_point_mirror_()
        , sampler_state_linear_clamp_()
        , sampler_state_viewspace_depth_tap_()
        , depth_buffer_viewspace_linear_layout_()
        , ao_result_layout_()
        , size_(0, 0)
        , half_size_(0, 0)
        , quarter_size_(0, 0)
        , depth_mip_levels_(0)
        , inputs_(nullptr)
        , max_blur_pass_count_(6)
        , ssbo_constants_(nullptr)
        , settings_()
        , ps_ssao_mode_("SSAO", "Specifices which SSAO technqiue should be used: naive SSAO or ASSAO")
        , ps_radius_("Radius", "Specifies world (view) space size of the occlusion sphere")
        , ps_shadow_multiplier_("ShadowMultiplier", "Specifies effect strength linear multiplier")
        , ps_shadow_power_("ShadowPower", "Specifies the effect strength pow modifier")
        , ps_shadow_clamp_("ShadowClamp", "Specifies the effect max limit")
        , ps_horizon_angle_threshold_("HorizonAngleThreshold", "Specifies the self-shadowing limit")
        , ps_fade_out_from_("FadeOutFrom", "Specifies the distance to start fading out the effect")
        , ps_fade_out_to_("FadeOutTo", "Specifies the distance at which the effect is faded out")
        , ps_quality_level_("QualityLevel", "Specifies the ssao effect quality level")
        , ps_adaptive_quality_limit_(
              "AdaptiveQualityLimit", "Specifies the adaptive quality limit (only for quality level 3)")
        , ps_blur_pass_count_("BlurPassCount", "Specifies the number of edge-sensitive smart blur passes to apply")
        , ps_sharpness_("Sharpness", "Specifies how much to bleed over edges")
        , ps_temporal_supersampling_angle_offset_("TemporalSupersamplingAngleOffset",
              "Specifies the rotating of the sampling kernel if temporal AA / supersampling is used")
        , ps_temporal_supersampling_radius_offset_("TemporalSupersamplingRadiusOffset",
              "Specifies the scaling of the sampling kernel if temporal AA / supersampling is used")
        , ps_detail_shadow_strength_(
              "DetailShadowStrength", "Specifies the high-res detail AO using neighboring depth pixels")
        , ps_ssao_radius_("SSAO Radius", "Sets radius for SSAO")
        , ps_ssao_sample_cnt_("SSAO Samples", "Sets the number of samples used SSAO")
        , settings_have_changed_(false)
        , slot_is_active_(false)
        , update_caused_by_normal_slot_change_(false) {
    this->output_tex_slot_.SetCallback(CallTexture2D::ClassName(), "GetData", &SSAO::getDataCallback);
    this->output_tex_slot_.SetCallback(CallTexture2D::ClassName(), "GetMetaData", &SSAO::getMetaDataCallback);
    this->MakeSlotAvailable(&this->output_tex_slot_);

    this->normals_tex_slot_.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->normals_tex_slot_);

    this->depth_tex_slot_.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->depth_tex_slot_);

    this->camera_slot_.SetCompatibleCall<CallCameraDescription>();
    this->MakeSlotAvailable(&this->camera_slot_);

    this->ps_ssao_mode_ << new core::param::EnumParam(0);
    this->ps_ssao_mode_.Param<core::param::EnumParam>()->SetTypePair(0, "ASSAO");
    this->ps_ssao_mode_.Param<core::param::EnumParam>()->SetTypePair(1, "Naive");
    this->ps_ssao_mode_.SetUpdateCallback(&SSAO::ssaoModeCallback);
    this->MakeSlotAvailable(&this->ps_ssao_mode_);

    this->ps_ssao_radius_ << new megamol::core::param::FloatParam(0.5f, 0.0f);
    this->ps_ssao_radius_.Parameter()->SetGUIVisible(false);
    this->MakeSlotAvailable(&this->ps_ssao_radius_);

    this->ps_ssao_sample_cnt_ << new megamol::core::param::IntParam(16, 0, 64);
    this->ps_ssao_sample_cnt_.Parameter()->SetGUIVisible(false);
    this->MakeSlotAvailable(&this->ps_ssao_sample_cnt_);

    // settings
    this->ps_radius_ << new core::param::FloatParam(1.2f, 0.f);
    this->ps_radius_.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->ps_radius_.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->ps_radius_);

    this->ps_shadow_multiplier_ << new core::param::FloatParam(1.f, 0.f, 5.f);
    this->ps_shadow_multiplier_.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->ps_shadow_multiplier_.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->ps_shadow_multiplier_);

    this->ps_shadow_power_ << new core::param::FloatParam(1.5f, 0.5f, 5.f);
    this->ps_shadow_power_.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->ps_shadow_power_.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->ps_shadow_power_);

    this->ps_shadow_clamp_ << new core::param::FloatParam(0.98f, 0.f, 1.f);
    this->ps_shadow_clamp_.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->ps_shadow_clamp_.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->ps_shadow_clamp_);

    this->ps_horizon_angle_threshold_ << new core::param::FloatParam(0.06f, 0.f, 0.2f);
    this->ps_horizon_angle_threshold_.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->ps_horizon_angle_threshold_.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->ps_horizon_angle_threshold_);

    this->ps_fade_out_from_ << new core::param::FloatParam(50.f, 0.f);
    this->ps_fade_out_from_.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->ps_fade_out_from_.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->ps_fade_out_from_);

    this->ps_fade_out_to_ << new core::param::FloatParam(300.f, 0.f);
    this->ps_fade_out_to_.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->ps_fade_out_to_.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->ps_fade_out_to_);

    // generally there are quality levels from -1 (lowest) to 3 (highest, adaptive), but 3 (adaptive) is not implemented yet
    this->ps_quality_level_ << new core::param::EnumParam(2);
    this->ps_quality_level_.Param<core::param::EnumParam>()->SetTypePair(-1, "Lowest");
    this->ps_quality_level_.Param<core::param::EnumParam>()->SetTypePair(0, "Low");
    this->ps_quality_level_.Param<core::param::EnumParam>()->SetTypePair(1, "Medium");
    this->ps_quality_level_.Param<core::param::EnumParam>()->SetTypePair(2, "High");
    this->ps_quality_level_.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->ps_quality_level_);

    this->ps_adaptive_quality_limit_ << new core::param::FloatParam(0.45f, 0.f, 1.f);
    this->ps_adaptive_quality_limit_.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->ps_adaptive_quality_limit_.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->ps_adaptive_quality_limit_);

    this->ps_blur_pass_count_ << new core::param::IntParam(2, 0, 6);
    this->ps_blur_pass_count_.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->ps_blur_pass_count_);

    this->ps_sharpness_ << new core::param::FloatParam(0.98f, 0.f, 1.f);
    this->ps_sharpness_.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->ps_sharpness_.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->ps_sharpness_);

    this->ps_temporal_supersampling_angle_offset_ << new core::param::FloatParam(0.f, 0.f, 3.141592653589f);
    this->ps_temporal_supersampling_angle_offset_.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->ps_temporal_supersampling_angle_offset_.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->ps_temporal_supersampling_angle_offset_);

    this->ps_temporal_supersampling_radius_offset_ << new core::param::FloatParam(1.f, 0.f, 2.f);
    this->ps_temporal_supersampling_radius_offset_.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->ps_temporal_supersampling_radius_offset_.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->ps_temporal_supersampling_radius_offset_);

    this->ps_detail_shadow_strength_ << new core::param::FloatParam(0.5f, 0.f, 5.f);
    this->ps_detail_shadow_strength_.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->ps_detail_shadow_strength_.SetUpdateCallback(&SSAO::settingsCallback);
    this->MakeSlotAvailable(&this->ps_detail_shadow_strength_);
}


/*
 * @megamol::compositing::SSAO::ssaoModeCallback
 */
bool megamol::compositing::SSAO::ssaoModeCallback(core::param::ParamSlot& slot) {
    int mode = ps_ssao_mode_.Param<core::param::EnumParam>()->Value();

    // assao
    if (mode == 0) {
        ps_ssao_radius_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        ps_ssao_sample_cnt_.Param<core::param::IntParam>()->SetGUIVisible(false);

        ps_radius_.Param<core::param::FloatParam>()->SetGUIVisible(true);
        ps_shadow_multiplier_.Param<core::param::FloatParam>()->SetGUIVisible(true);
        ps_shadow_power_.Param<core::param::FloatParam>()->SetGUIVisible(true);
        ps_shadow_clamp_.Param<core::param::FloatParam>()->SetGUIVisible(true);
        ps_horizon_angle_threshold_.Param<core::param::FloatParam>()->SetGUIVisible(true);
        ps_fade_out_from_.Param<core::param::FloatParam>()->SetGUIVisible(true);
        ps_fade_out_to_.Param<core::param::FloatParam>()->SetGUIVisible(true);
        ps_quality_level_.Param<core::param::EnumParam>()->SetGUIVisible(true);
        ps_adaptive_quality_limit_.Param<core::param::FloatParam>()->SetGUIVisible(true);
        ps_blur_pass_count_.Param<core::param::IntParam>()->SetGUIVisible(true);
        ps_sharpness_.Param<core::param::FloatParam>()->SetGUIVisible(true);
        ps_temporal_supersampling_angle_offset_.Param<core::param::FloatParam>()->SetGUIVisible(true);
        ps_temporal_supersampling_radius_offset_.Param<core::param::FloatParam>()->SetGUIVisible(true);
        ps_detail_shadow_strength_.Param<core::param::FloatParam>()->SetGUIVisible(true);
    }
    // naive
    else {
        ps_ssao_radius_.Param<core::param::FloatParam>()->SetGUIVisible(true);
        ps_ssao_sample_cnt_.Param<core::param::IntParam>()->SetGUIVisible(true);

        ps_radius_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        ps_shadow_multiplier_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        ps_shadow_power_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        ps_shadow_clamp_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        ps_horizon_angle_threshold_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        ps_fade_out_from_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        ps_fade_out_to_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        ps_quality_level_.Param<core::param::EnumParam>()->SetGUIVisible(false);
        ps_adaptive_quality_limit_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        ps_blur_pass_count_.Param<core::param::IntParam>()->SetGUIVisible(false);
        ps_sharpness_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        ps_temporal_supersampling_angle_offset_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        ps_temporal_supersampling_radius_offset_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        ps_detail_shadow_strength_.Param<core::param::FloatParam>()->SetGUIVisible(false);
    }

    return true;
}


/*
 * @megamol::compositing::SSAO::settingsCallback
 */
bool megamol::compositing::SSAO::settingsCallback(core::param::ParamSlot& slot) {
    settings_.Radius = ps_radius_.Param<core::param::FloatParam>()->Value();
    settings_.ShadowMultiplier = ps_shadow_multiplier_.Param<core::param::FloatParam>()->Value();
    settings_.ShadowPower = ps_shadow_power_.Param<core::param::FloatParam>()->Value();
    settings_.ShadowClamp = ps_shadow_clamp_.Param<core::param::FloatParam>()->Value();
    settings_.HorizonAngleThreshold = ps_horizon_angle_threshold_.Param<core::param::FloatParam>()->Value();
    settings_.FadeOutFrom = ps_fade_out_from_.Param<core::param::FloatParam>()->Value();
    settings_.FadeOutTo = ps_fade_out_to_.Param<core::param::FloatParam>()->Value();
    settings_.QualityLevel = ps_quality_level_.Param<core::param::EnumParam>()->Value();
    settings_.AdaptiveQualityLimit = ps_adaptive_quality_limit_.Param<core::param::FloatParam>()->Value();
    settings_.BlurPassCount = ps_blur_pass_count_.Param<core::param::IntParam>()->Value();
    settings_.Sharpness = ps_sharpness_.Param<core::param::FloatParam>()->Value();
    settings_.TemporalSupersamplingAngleOffset =
        ps_temporal_supersampling_angle_offset_.Param<core::param::FloatParam>()->Value();
    settings_.TemporalSupersamplingRadiusOffset =
        ps_temporal_supersampling_radius_offset_.Param<core::param::FloatParam>()->Value();
    settings_.DetailShadowStrength = ps_detail_shadow_strength_.Param<core::param::FloatParam>()->Value();

    settings_have_changed_ = true;

    return true;
}


/*
 * @megamol::compositing::SSAO::~SSAO
 */
megamol::compositing::SSAO::~SSAO() {
    this->Release();
}


/*
 * @megamol::compositing::SSAO::create
 */
bool megamol::compositing::SSAO::create() {
    typedef megamol::core::utility::log::Log Log;

    prepare_depth_mip_prgms_.resize(SSAODepth_MIP_LEVELS - 1);

    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());

    try {
        prepare_depths_prgm_ =
            core::utility::make_glowl_shader("prepare_depths", shader_options, "comp/assao/prepare_depths.comp.glsl");

        prepare_depths_half_prgm_ = core::utility::make_glowl_shader(
            "prepare_depths_half", shader_options, "comp/assao/prepare_depths_half.comp.glsl");

        prepare_depths_and_normals_prgm_ = core::utility::make_glowl_shader(
            "prepare_depths_and_normals", shader_options, "comp/assao/prepare_depths_and_normals.comp.glsl");

        prepare_depths_and_normals_half_prgm_ = core::utility::make_glowl_shader(
            "prepare_depths_and_normals_half", shader_options, "comp/assao/prepare_depths_and_normals_half.comp.glsl");

        prepare_depth_mip_prgms_[0] = core::utility::make_glowl_shader(
            "prepare_depth_mip1", shader_options, "comp/assao/prepare_depth_mip1.comp.glsl");

        prepare_depth_mip_prgms_[1] = core::utility::make_glowl_shader(
            "prepare_depth_mip2", shader_options, "comp/assao/prepare_depth_mip2.comp.glsl");

        prepare_depth_mip_prgms_[2] = core::utility::make_glowl_shader(
            "prepare_depth_mip3", shader_options, "comp/assao/prepare_depth_mip3.comp.glsl");

        generate_prgms_[0] =
            core::utility::make_glowl_shader("generate_q0", shader_options, "comp/assao/generate_q0.comp.glsl");

        generate_prgms_[1] =
            core::utility::make_glowl_shader("generate_q1", shader_options, "comp/assao/generate_q1.comp.glsl");

        generate_prgms_[2] =
            core::utility::make_glowl_shader("generate_q2", shader_options, "comp/assao/generate_q1.comp.glsl");

        generate_prgms_[3] =
            core::utility::make_glowl_shader("generate_q3", shader_options, "comp/assao/generate_q1.comp.glsl");

        smart_blur_prgm_ =
            core::utility::make_glowl_shader("smart_blur", shader_options, "comp/assao/smart_blur.comp.glsl");

        smart_blur_wide_prgm_ =
            core::utility::make_glowl_shader("smart_blur_wide", shader_options, "comp/assao/smart_blur_wide.comp.glsl");

        apply_prgm_ = core::utility::make_glowl_shader("apply", shader_options, "comp/assao/apply.comp.glsl");

        non_smart_blur_prgm_ =
            core::utility::make_glowl_shader("non_smart_blur", shader_options, "comp/assao/non_smart_blur.comp.glsl");

        non_smart_apply_prgm_ =
            core::utility::make_glowl_shader("non_smart_apply", shader_options, "comp/assao/non_smart_apply.comp.glsl");

        non_smart_half_apply_prgm_ = core::utility::make_glowl_shader(
            "non_smart_half_apply", shader_options, "comp/assao/non_smart_half_apply.comp.glsl");

        naive_ssao_prgm_ = core::utility::make_glowl_shader("naive_ssao", shader_options, "comp/naive_ssao.comp.glsl");

        simple_blur_prgm_ =
            core::utility::make_glowl_shader("simple_blur", shader_options, "comp/simple_blur.comp.glsl");

    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, ("SSAO: " + std::string(e.what())).c_str());
    }

    depth_buffer_viewspace_linear_layout_ = glowl::TextureLayout(GL_R16F, 1, 1, 1, GL_RED, GL_HALF_FLOAT, 1);
    ao_result_layout_ = glowl::TextureLayout(GL_RG8, 1, 1, 1, GL_RG, GL_FLOAT, 1);
    normal_layout_ = glowl::TextureLayout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_HALF_FLOAT, 1);
    half_depths_[0] =
        std::make_shared<glowl::Texture2D>("half_depths0", depth_buffer_viewspace_linear_layout_, nullptr);
    half_depths_[1] =
        std::make_shared<glowl::Texture2D>("half_depths1", depth_buffer_viewspace_linear_layout_, nullptr);
    half_depths_[2] =
        std::make_shared<glowl::Texture2D>("half_depths2", depth_buffer_viewspace_linear_layout_, nullptr);
    half_depths_[3] =
        std::make_shared<glowl::Texture2D>("half_depths3", depth_buffer_viewspace_linear_layout_, nullptr);
    half_depths_mip_views_.resize(4);
    for (int j = 0; j < 4; ++j) {
        half_depths_mip_views_[j].resize(SSAODepth_MIP_LEVELS);
        for (int i = 0; i < half_depths_mip_views_[j].size(); ++i) {
            half_depths_mip_views_[j][i] =
                std::make_shared<glowl::Texture2DView>("half_depths_mip_views" + std::to_string(i), *half_depths_[j],
                    depth_buffer_viewspace_linear_layout_, 0, 1, 0, 1);
        }
    }
    final_output_ = std::make_shared<glowl::Texture2D>("final_output", depth_buffer_viewspace_linear_layout_, nullptr);
    ping_pong_half_result_a_ =
        std::make_shared<glowl::Texture2D>("ping_pong_half_result_a", ao_result_layout_, nullptr);
    ping_pong_half_result_b_ =
        std::make_shared<glowl::Texture2D>("ping_pong_half_result_b", ao_result_layout_, nullptr);
    final_results_ = std::make_shared<glowl::Texture2DArray>("final_results", ao_result_layout_, nullptr);
    final_results_array_views_[0] = std::make_shared<glowl::Texture2DView>(
        "final_results_array_views0", *final_results_, ao_result_layout_, 0, 1, 0, 1);
    final_results_array_views_[1] = std::make_shared<glowl::Texture2DView>(
        "final_results_array_views1", *final_results_, ao_result_layout_, 0, 1, 0, 1);
    final_results_array_views_[2] = std::make_shared<glowl::Texture2DView>(
        "final_results_array_views2", *final_results_, ao_result_layout_, 0, 1, 0, 1);
    final_results_array_views_[3] = std::make_shared<glowl::Texture2DView>(
        "final_results_array_views3", *final_results_, ao_result_layout_, 0, 1, 0, 1);
    normals_ = std::make_shared<glowl::Texture2D>("normals", normal_layout_, nullptr);

    inputs_ = std::make_shared<ASSAO_Inputs>();

    ssbo_constants_ = std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    std::vector<std::pair<GLenum, GLint>> intParams = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST},
        {GL_TEXTURE_MAG_FILTER, GL_NEAREST}, {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE}};

    sampler_state_point_clamp_ = std::make_shared<glowl::Sampler>("sampler_state_point_clamp", intParams);

    intParams.clear();
    intParams = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST}, {GL_TEXTURE_MAG_FILTER, GL_NEAREST},
        {GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT}, {GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT}};

    sampler_state_point_mirror_ = std::make_shared<glowl::Sampler>("sampler_state_point_mirror", intParams);

    intParams.clear();
    intParams = {{GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR}, {GL_TEXTURE_MAG_FILTER, GL_LINEAR},
        {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}};

    sampler_state_linear_clamp_ = std::make_shared<glowl::Sampler>("sampler_state_linear_clamp", intParams);

    intParams.clear();
    intParams = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST}, {GL_TEXTURE_MAG_FILTER, GL_NEAREST},
        {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}};

    sampler_state_viewspace_depth_tap_ = std::make_shared<glowl::Sampler>("sampler_state_viewspace_depth_tap", intParams);


    // naive ssao stuff
    intermediate_tx2d_ = std::make_shared<glowl::Texture2D>("screenspace_effect_intermediate", normal_layout_, nullptr);

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

    ssao_samples_ = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, ssaoKernel, GL_DYNAMIC_DRAW);

    std::vector<glm::vec3> ssaoNoise;
    for (unsigned int i = 0; i < 16; i++) {
        glm::vec3 noise(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, 0.0f);
        ssaoNoise.push_back(noise);
    }

    glowl::TextureLayout tx_layout2(GL_RGB32F, 4, 4, 1, GL_RGB, GL_FLOAT, 1);
    ssao_kernel_rot_tx2d_ = std::make_shared<glowl::Texture2D>("ssao_kernel_rotation", tx_layout2, ssaoNoise.data());

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
    auto callNormal = normals_tex_slot_.CallAs<CallTexture2D>();
    auto callDepth = depth_tex_slot_.CallAs<CallTexture2D>();
    auto callCamera = camera_slot_.CallAs<CallCamera>();

    if (lhsTc == NULL)
        return false;

    if ((callDepth != NULL) && (callCamera != NULL)) {

        bool generateNormals = false;
        if (callNormal == NULL) {
            if (slot_is_active_) {
                slot_is_active_ = false;
                update_caused_by_normal_slot_change_ = true;
            }
            generateNormals = true;
        } else {
            if (!slot_is_active_) {
                slot_is_active_ = true;
                update_caused_by_normal_slot_change_ = true;
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

        bool somethingHasChanged = (callNormal != NULL ? normalUpdate : false) ||
                                   (callDepth != NULL ? depthUpdate : false) ||
                                   (callCamera != NULL ? cameraUpdate : false) ||
                                   update_caused_by_normal_slot_change_ || settings_have_changed_;

        if (somethingHasChanged) {
            ++version_;

            std::function<void(std::shared_ptr<glowl::Texture2D> src, std::shared_ptr<glowl::Texture2D> tgt)>
                setupOutputTexture = [](std::shared_ptr<glowl::Texture2D> src, std::shared_ptr<glowl::Texture2D> tgt) {
                    // set output texture size to primary input texture
                    std::array<float, 2> texture_res = {
                        static_cast<float>(src->getWidth()), static_cast<float>(src->getHeight())};

                    if (tgt->getWidth() != std::get<0>(texture_res) || tgt->getHeight() != std::get<1>(texture_res)) {
                        glowl::TextureLayout tx_layout(GL_RGBA16F, std::get<0>(texture_res), std::get<1>(texture_res),
                            1, GL_RGBA, GL_HALF_FLOAT, 1);
                        tgt->reload(tx_layout, nullptr);
                    }
                };

            if (callNormal == NULL && slot_is_active_)
                return false;
            if (callDepth == NULL)
                return false;
            if (callCamera == NULL)
                return false;

            std::array<int, 2> txResNormal;
            if (!generateNormals) {
                normals_ = callNormal->getData();
                txResNormal = {(int)normals_->getWidth(), (int)normals_->getHeight()};
            }

            auto depthTx2D = callDepth->getData();
            std::array<int, 2> txResDepth = {(int)depthTx2D->getWidth(), (int)depthTx2D->getHeight()};

            setupOutputTexture(depthTx2D, final_output_);

            // obtain camera information
            core::view::Camera cam = callCamera->getData();
            glm::mat4 viewMx = cam.getViewMatrix();
            glm::mat4 projMx = cam.getProjectionMatrix();

            int ssaoMode = ps_ssao_mode_.Param<core::param::EnumParam>()->Value();

            // assao
            if (ssaoMode == 0) {

                if (normalUpdate || depthUpdate || settings_have_changed_ || update_caused_by_normal_slot_change_) {

                    // assuming a full resolution depth buffer!
                    inputs_->ViewportWidth = txResDepth[0];
                    inputs_->ViewportHeight = txResDepth[1];
                    inputs_->GenerateNormals = generateNormals;
                    inputs_->ProjectionMatrix = projMx;
                    inputs_->ViewMatrix = viewMx;

                    updateTextures(inputs_);

                    updateConstants(settings_, inputs_, 0);
                }

                {
                    prepareDepths(settings_, inputs_, depthTx2D, normals_);

                    generateSSAO(settings_, inputs_, false, normals_);

                    std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTextures = {
                        {final_output_, 0}};

                    // Apply
                    {
                        std::vector<TextureArraySamplerTuple> inputFinals = {
                            {final_results_, "g_FinalSSAOLinearClamp", sampler_state_linear_clamp_}};

                        if (settings_.QualityLevel < 0)
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                                non_smart_half_apply_prgm_, {}, outputTextures, true, inputFinals);
                        else if (settings_.QualityLevel == 0)
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                                non_smart_apply_prgm_, {}, outputTextures, true, inputFinals);
                        else {
                            inputFinals.push_back({final_results_, "g_FinalSSAO", nullptr});
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                                apply_prgm_, {}, outputTextures, true, inputFinals);
                        }
                    }
                }
            }
            // naive
            else {
                setupOutputTexture(depthTx2D, intermediate_tx2d_);

                naive_ssao_prgm_->use();

                ssao_samples_->bind(1);

                glUniform1f(naive_ssao_prgm_->getUniformLocation("radius"),
                    ps_ssao_radius_.Param<core::param::FloatParam>()->Value());
                glUniform1i(naive_ssao_prgm_->getUniformLocation("sample_cnt"),
                    ps_ssao_sample_cnt_.Param<core::param::IntParam>()->Value());

                glActiveTexture(GL_TEXTURE0);
                normals_->bindTexture();
                glUniform1i(naive_ssao_prgm_->getUniformLocation("normal_tx2D"), 0);
                glActiveTexture(GL_TEXTURE1);
                depthTx2D->bindTexture();
                glUniform1i(naive_ssao_prgm_->getUniformLocation("depth_tx2D"), 1);
                glActiveTexture(GL_TEXTURE2);
                ssao_kernel_rot_tx2d_->bindTexture();
                glUniform1i(naive_ssao_prgm_->getUniformLocation("noise_tx2D"), 2);

                auto invViewMx = glm::inverse(viewMx);
                auto invProjMx = glm::inverse(projMx);
                glUniformMatrix4fv(
                    naive_ssao_prgm_->getUniformLocation("inv_view_mx"), 1, GL_FALSE, glm::value_ptr(invViewMx));
                glUniformMatrix4fv(
                    naive_ssao_prgm_->getUniformLocation("inv_proj_mx"), 1, GL_FALSE, glm::value_ptr(invProjMx));

                glUniformMatrix4fv(naive_ssao_prgm_->getUniformLocation("view_mx"), 1, GL_FALSE, glm::value_ptr(viewMx));
                glUniformMatrix4fv(naive_ssao_prgm_->getUniformLocation("proj_mx"), 1, GL_FALSE, glm::value_ptr(projMx));

                intermediate_tx2d_->bindImage(0, GL_WRITE_ONLY);

                glDispatchCompute(static_cast<int>(std::ceil(final_output_->getWidth() / 8.0f)),
                    static_cast<int>(std::ceil(final_output_->getHeight() / 8.0f)), 1);
                ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                glUseProgram(0);

                simple_blur_prgm_->use();

                glActiveTexture(GL_TEXTURE0);
                intermediate_tx2d_->bindTexture();
                // test with naive_ssao_prgm_, since it was (falsely) used the entire time
                glUniform1i(simple_blur_prgm_->getUniformLocation("src_tx2D"), 0);

                final_output_->bindImage(0, GL_WRITE_ONLY);

                glDispatchCompute(static_cast<int>(std::ceil(final_output_->getWidth() / 8.0f)),
                    static_cast<int>(std::ceil(final_output_->getHeight() / 8.0f)), 1);
                ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                glUseProgram(0);
            }
        }
    }

    if (lhsTc->version() < version_) {
        settings_have_changed_ = false;
        update_caused_by_normal_slot_change_ = false;
    }

    lhsTc->setData(final_output_, version_);

    return true;
}


/*
 * @megamol::compositing::SSAO::prepareDepths
 */
void megamol::compositing::SSAO::prepareDepths(const ASSAO_Settings& settings,
    const std::shared_ptr<ASSAO_Inputs> inputs, std::shared_ptr<glowl::Texture2D> depthTexture,
    std::shared_ptr<glowl::Texture2D> normalTexture) {
    bool generateNormals = inputs->GenerateNormals;

    std::vector<TextureSamplerTuple> inputTextures(1);
    inputTextures[0] = {depthTexture, (std::string) "g_DepthSource", nullptr};


    std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputFourDepths = {
        {half_depths_[0], 0}, {half_depths_[1], 1}, {half_depths_[2], 2}, {half_depths_[3], 3}};
    std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTwoDepths = {
        {half_depths_[0], 0}, {half_depths_[3], 3}};

    if (!generateNormals) {
        if (settings.QualityLevel < 0) {
            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                prepare_depths_half_prgm_, inputTextures, outputTwoDepths);
        } else {
            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                prepare_depths_prgm_, inputTextures, outputFourDepths);
        }
    } else {
        inputTextures.push_back({depthTexture, (std::string) "g_DepthSourcePointClamp", sampler_state_point_clamp_});

        if (settings.QualityLevel < 0) {
            outputTwoDepths.push_back({normalTexture, 4});
            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                prepare_depths_and_normals_half_prgm_, inputTextures, outputTwoDepths);
        } else {
            outputFourDepths.push_back({normalTexture, 4});
            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                prepare_depths_and_normals_prgm_, inputTextures, outputFourDepths);
        }
    }

    // only do mipmaps for higher quality levels (not beneficial on quality level 1, and detrimental on quality level 0)
    if (settings.QualityLevel > 1) {

#ifdef MEGAMOL_ASSAO_MANUAL_MIPS
        for (int i = 1; i < depth_mip_levels_; ++i) {
            std::vector<TextureViewSamplerTuple> inputFourDepthMipsM1 = {
                {half_depths_mip_views_[0][i - 1LL], (std::string) "g_ViewspaceDepthSource", nullptr},
                {half_depths_mip_views_[1][i - 1LL], (std::string) "g_ViewspaceDepthSource1", nullptr},
                {half_depths_mip_views_[2][i - 1LL], (std::string) "g_ViewspaceDepthSource2", nullptr},
                {half_depths_mip_views_[3][i - 1LL], (std::string) "g_ViewspaceDepthSource3", nullptr}};

            std::vector<std::pair<std::shared_ptr<glowl::Texture2DView>, GLuint>> outputFourDepthMips = {
                {half_depths_mip_views_[0][i], 0}, {half_depths_mip_views_[1][i], 1}, {half_depths_mip_views_[2][i], 2},
                {half_depths_mip_views_[3][i], 3}};

            fullscreenPassDraw<TextureViewSamplerTuple, glowl::Texture2DView>(
                prepare_depth_mip_prgms_[i - 1LL], inputFourDepthMipsM1, outputFourDepthMips);
        }
#else
        for (int i = 0; i < 4; ++i) {
            half_depths_[i]->bindTexture();
            half_depths_[i]->updateMipmaps();
        }
        glBindTexture(GL_TEXTURE_2D, 0);
#endif
    }
}


/*
 * @megamol::compositing::SSAO::generateSSAO
 */
void megamol::compositing::SSAO::generateSSAO(const ASSAO_Settings& settings,
    const std::shared_ptr<ASSAO_Inputs> inputs, bool adaptiveBasePass,
    std::shared_ptr<glowl::Texture2D> normalTexture) {

    // omitted viewport and scissor code from intel here

    if (adaptiveBasePass) {
        assert(settings.QualityLevel == 3);
    }

    int passCount = 4;

    for (int pass = 0; pass < passCount; ++pass) {
        if ((settings.QualityLevel < 0) && ((pass == 1) || (pass == 2)))
            continue;

        int blurPasses = std::min(settings.BlurPassCount, max_blur_pass_count_);

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
            inputTextures[0] = {half_depths_[pass], "g_ViewSpaceDepthSource", sampler_state_point_mirror_};
            inputTextures[1] = {normalTexture, "g_NormalmapSource", nullptr};
            inputTextures[2] = {
                half_depths_[pass], "g_ViewSpaceDepthSourceDepthTapSampler", sampler_state_viewspace_depth_tap_};

            // CHECK FOR ADAPTIVE SSAO
#ifdef INTEL_SSAO_ENABLE_ADAPTIVE_QUALITY
            if (!adaptiveBasePass && (settings.QualityLevel == 3)) {
                inputTextures[3] = {load_counter_srv_, "g_LoadCounter"};
                inputTextures[4] = {importance_map_.SRV, "g_ImportanceMap"};
                inputTextures[5] = {final_results_.SRV, "g_FinalSSAO"};
            }
#endif
            GLuint binding = 0;
            int shaderIndex = std::max(0, !adaptiveBasePass ? settings.QualityLevel : 4);

            // no blur?
            if (blurPasses == 0) {
                std::vector<std::pair<std::shared_ptr<glowl::Texture2DView>, GLuint>> outputTextures = {
                    {final_results_array_views_[pass], binding}};

                fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2DView>(
                    generate_prgms_[shaderIndex], inputTextures, outputTextures);
            } else {
                std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTextures = {
                    {ping_pong_half_result_a_, binding}};

                fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                    generate_prgms_[shaderIndex], inputTextures, outputTextures);
            }
        }

        // Blur
        if (blurPasses > 0) {
            int wideBlursRemaining = std::max(0, blurPasses - 2);

            for (int i = 0; i < blurPasses; ++i) {
                GLuint binding = 0;

                std::vector<TextureSamplerTuple> inputTextures = {
                    {ping_pong_half_result_a_, "g_BlurInput", sampler_state_point_mirror_}};

                std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> intermediateOutputTexs = {
                    {ping_pong_half_result_b_, binding}};

                std::vector<std::pair<std::shared_ptr<glowl::Texture2DView>, GLuint>> finalOutputTexs = {
                    {final_results_array_views_[pass], binding}};

                if (settings.QualityLevel > 0) {
                    if (wideBlursRemaining > 0) {
                        if (i == blurPasses - 1) {
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2DView>(
                                smart_blur_wide_prgm_, inputTextures, finalOutputTexs);
                        } else {
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                                smart_blur_wide_prgm_, inputTextures, intermediateOutputTexs);
                        }

                        wideBlursRemaining--;
                    } else {
                        if (i == blurPasses - 1) {
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2DView>(
                                smart_blur_prgm_, inputTextures, finalOutputTexs);
                        } else {
                            fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                                smart_blur_prgm_, inputTextures, intermediateOutputTexs);
                        }
                    }
                } else {
                    std::get<2>(inputTextures[0]) = sampler_state_linear_clamp_;

                    if (i == blurPasses - 1) {
                        fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2DView>(
                            non_smart_blur_prgm_, inputTextures, finalOutputTexs);
                    } else {
                        fullscreenPassDraw<TextureSamplerTuple, glowl::Texture2D>(
                            non_smart_blur_prgm_, inputTextures, intermediateOutputTexs);
                    }
                }

                std::swap(ping_pong_half_result_a_, ping_pong_half_result_b_);
            }
        }
    }
}


/*
 * @megamol::compositing::SSAO::getMetaDataCallback
 */
bool megamol::compositing::SSAO::getMetaDataCallback(core::Call& caller) {
    return true;
}


/*
 * @megamol::compositing::SSAO::updateTextures
 */
void megamol::compositing::SSAO::updateTextures(const std::shared_ptr<ASSAO_Inputs> inputs) {
    int width = inputs->ViewportWidth;
    int height = inputs->ViewportHeight;

    bool needsUpdate = (size_.x != width) || (size_.y != height);

    size_.x = width;
    size_.y = height;
    half_size_.x = (width + 1) / 2;
    half_size_.y = (height + 1) / 2;
    quarter_size_.x = (half_size_.x + 1) / 2;
    quarter_size_.y = (half_size_.y + 1) / 2;

    if (!needsUpdate)
        return;

    int blurEnlarge =
        max_blur_pass_count_ + std::max(0, max_blur_pass_count_ - 2); // +1 for max normal blurs, +2 for wide blurs

    float totalSizeInMB = 0.f;

    depth_mip_levels_ = SSAODepth_MIP_LEVELS;

    for (int i = 0; i < 4; i++) {
        if (reCreateIfNeeded(half_depths_[i], half_size_, depth_buffer_viewspace_linear_layout_, true)) {

#ifdef MEGAMOL_ASSAO_MANUAL_MIPS
            for (int j = 0; j < depth_mip_levels_; j++) {
                reCreateMIPViewIfNeeded(half_depths_mip_views_[i][j], half_depths_[i], j);
            }
#endif
        }
    }

    reCreateIfNeeded(ping_pong_half_result_a_, half_size_, ao_result_layout_);
    reCreateIfNeeded(ping_pong_half_result_b_, half_size_, ao_result_layout_);
    reCreateIfNeeded(final_results_, half_size_, ao_result_layout_);

    for (int i = 0; i < 4; ++i) {
        reCreateArrayIfNeeded(final_results_array_views_[i], final_results_, half_size_, i);
    }

    if (inputs->GenerateNormals) {
        reCreateIfNeeded(normals_, size_, normal_layout_);
    }
}


/*
 * @megamol::compositing::SSAO::updateConstants
 */
void megamol::compositing::SSAO::updateConstants(
    const ASSAO_Settings& settings, const std::shared_ptr<ASSAO_Inputs> inputs, int pass) {
    bool generateNormals = inputs->GenerateNormals;

    // update constants
    ASSAO_Constants& consts = constants_; // = *((ASSAOConstants*) mappedResource.pData);

    const glm::mat4& proj = inputs->ProjectionMatrix;

    consts.ViewportPixelSize = glm::vec2(1.0f / (float)size_.x, 1.0f / (float)size_.y);
    consts.HalfViewportPixelSize = glm::vec2(1.0f / (float)half_size_.x, 1.0f / (float)half_size_.y);

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

    int lastHalfDepthMipX = half_depths_mip_views_[0][SSAODepth_MIP_LEVELS - 1]->getWidth();
    int lastHalfDepthMipY = half_depths_mip_views_[0][SSAODepth_MIP_LEVELS - 1]->getHeight();

    // used to get average load per pixel; 9.0 is there to compensate for only doing every 9th InterlockedAdd in
    // PSPostprocessImportanceMapB for performance reasons
    consts.LoadCounterAvgDiv = 9.0f / (float)(quarter_size_.x * quarter_size_.y * 255.0);

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
    consts.PerPassFullResUVOffset = glm::vec2(((pass % 2) - 0.0f) / size_.x, ((pass / 2) - 0.0f) / size_.y);

    consts.InvSharpness = std::clamp(1.0f - settings.Sharpness, 0.0f, 1.0f);
    consts.PassIndex = pass;
    consts.QuarterResPixelSize = glm::vec2(1.0f / (float)quarter_size_.x, 1.0f / (float)quarter_size_.y);

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
    ssbo_constants_->rebuffer(&constants_, sizeof(constants_));
}


/*
 * @megamol::compositing::SSAO::reCreateIfNeeded
 */
bool megamol::compositing::SSAO::reCreateIfNeeded(
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
    std::shared_ptr<glowl::Texture2DArray> tex, glm::ivec2 size, const glowl::TextureLayout& ly) {
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
bool megamol::compositing::SSAO::reCreateArrayIfNeeded(std::shared_ptr<glowl::Texture2DView> tex,
    std::shared_ptr<glowl::Texture2DArray> original, glm::ivec2 size, int arraySlice) {
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
    std::shared_ptr<glowl::Texture2DView> current, std::shared_ptr<glowl::Texture2D> original, int mipViewSlice) {

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
    const glowl::TextureLayout& lhs, const glowl::TextureLayout& rhs) {
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
bool megamol::compositing::SSAO::equalLayouts(const glowl::TextureLayout& lhs, const glowl::TextureLayout& rhs) {
    bool depth = lhs.depth == rhs.depth;
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
