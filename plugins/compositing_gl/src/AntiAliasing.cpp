/*
 * AntiAliasing.cpp
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "AntiAliasing.h"

#include <array>
#include <random>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"

#include "vislib/graphics/gl/ShaderSource.h"

#include "compositing/CompositingCalls.h"

#include "SMAAAreaTex.h"
#include "SMAASearchTex.h"

megamol::compositing::AntiAliasing::AntiAliasing() : core::Module()
    , m_version(0)
    , m_output_texture(nullptr)
    , m_output_texture_hash(0)
    , m_mode("Mode", "Sets antialiasing technqiue: SMAA, FXAA, no AA")
    , m_smaa_mode("SMAA Mode", "Sets the SMAA mode: SMAA 1x or SMAA T2x")
    , m_smaa_quality("QualityLevel", "Sets smaa quality level")
    , m_smaa_threshold("Threshold", "Sets smaa threshold")
    , m_smaa_max_search_steps("MaxSearchSteps", "Sets smaa max search steps")
    , m_smaa_max_search_steps_diag("MaxDiagSearchSteps", "Sets smaa max diagonal search steps")
    , m_smaa_disable_diag_detection("DisableDiagDetection",
            "Enables/Disables diagonal detection. If set to false, diagonal detection is enabled")
    , m_smaa_disable_corner_detection("DisableCornerDetection",
            "Enables/Disables corner detection. If set to false, corner detection is enabled")
    , m_smaa_corner_rounding("CornerRounding", "Sets smaa corner rounding parameter")
    , m_smaa_detection_technique(
            "EdgeDetection", "Sets smaa edge detection base: luma, color, or depth. Use depth only when a depth "
                            "texture can be provided as it is mandatory to have one")
    , m_smaa_view("Output", "Sets the texture to view: final output, edges or weights. Edges or weights should "
                            "only be used when directly connected to the screen for debug purposes")
    , m_output_tex_slot("OutputTexture", "Gives access to the resulting output texture")
    , m_input_tex_slot("InputTexture", "Connects the input texture")
    , m_settings_have_changed(false)
{
    this->m_mode << new megamol::core::param::EnumParam(0);
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "SMAA");
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "FXAA");
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "None");
    this->m_mode.SetUpdateCallback(&megamol::compositing::AntiAliasing::visibilityCallback);
    this->MakeSlotAvailable(&this->m_mode);

    this->m_smaa_mode << new megamol::core::param::EnumParam(0);
    this->m_smaa_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "SMAA 1x");
    this->m_smaa_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "SMAA T2x");
    //this->m_smaa_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "SMAA S2x");
    //this->m_smaa_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(3, "SMAA 4x");
    this->m_smaa_mode.SetUpdateCallback(&megamol::compositing::AntiAliasing::visibilityCallback);
    this->MakeSlotAvailable(&this->m_smaa_mode);
    
    this->m_smaa_quality << new megamol::core::param::EnumParam(2);
    this->m_smaa_quality.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Low");
    this->m_smaa_quality.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "Medium");
    this->m_smaa_quality.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "High");
    this->m_smaa_quality.Param<megamol::core::param::EnumParam>()->SetTypePair(3, "Ultra");
    this->m_smaa_quality.Param<megamol::core::param::EnumParam>()->SetTypePair(4, "Custom");
    this->m_smaa_quality.SetUpdateCallback(&megamol::compositing::AntiAliasing::setSettingsCallback);
    this->MakeSlotAvailable(&this->m_smaa_quality);

    this->m_smaa_threshold << new megamol::core::param::FloatParam(0.1f, 0.f, 0.5f);
    this->m_smaa_threshold.Param<core::param::FloatParam>()->SetGUIVisible(false);
    this->m_smaa_threshold.SetUpdateCallback(&megamol::compositing::AntiAliasing::setCustomSettingsCallback);
    this->m_smaa_threshold.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->MakeSlotAvailable(&this->m_smaa_threshold);

    this->m_smaa_max_search_steps << new megamol::core::param::IntParam(16, 0, 112);
    this->m_smaa_max_search_steps.Param<core::param::IntParam>()->SetGUIVisible(false);
    this->m_smaa_max_search_steps.SetUpdateCallback(&megamol::compositing::AntiAliasing::setCustomSettingsCallback);
    this->m_smaa_max_search_steps.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->MakeSlotAvailable(&this->m_smaa_max_search_steps);

    this->m_smaa_max_search_steps_diag << new megamol::core::param::IntParam(8, 0, 20);
    this->m_smaa_max_search_steps_diag.Param<core::param::IntParam>()->SetGUIVisible(false);
    this->m_smaa_max_search_steps_diag.SetUpdateCallback(
        &megamol::compositing::AntiAliasing::setCustomSettingsCallback);
    this->m_smaa_max_search_steps_diag.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->MakeSlotAvailable(&this->m_smaa_max_search_steps_diag);

    this->m_smaa_disable_diag_detection << new megamol::core::param::BoolParam(false);
    this->m_smaa_disable_diag_detection.Param<core::param::BoolParam>()->SetGUIVisible(false);
    this->m_smaa_disable_diag_detection.SetUpdateCallback(
        &megamol::compositing::AntiAliasing::setCustomSettingsCallback);
    this->MakeSlotAvailable(&this->m_smaa_disable_diag_detection);

    this->m_smaa_disable_corner_detection << new megamol::core::param::BoolParam(false);
    this->m_smaa_disable_corner_detection.Param<core::param::BoolParam>()->SetGUIVisible(false);
    this->m_smaa_disable_corner_detection.SetUpdateCallback(
        &megamol::compositing::AntiAliasing::setCustomSettingsCallback);
    this->MakeSlotAvailable(&this->m_smaa_disable_corner_detection);

    this->m_smaa_corner_rounding << new megamol::core::param::IntParam(25, 0, 100);
    this->m_smaa_corner_rounding.Param<core::param::IntParam>()->SetGUIVisible(false);
    this->m_smaa_corner_rounding.SetUpdateCallback(&megamol::compositing::AntiAliasing::setCustomSettingsCallback);
    this->m_smaa_corner_rounding.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Drag);
    this->MakeSlotAvailable(&this->m_smaa_corner_rounding);

    this->m_smaa_detection_technique << new megamol::core::param::EnumParam(0);
    this->m_smaa_detection_technique.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Luma");
    this->m_smaa_detection_technique.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "Color");
    this->m_smaa_detection_technique.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "Depth");
    this->MakeSlotAvailable(&this->m_smaa_detection_technique);

    this->m_smaa_view << new megamol::core::param::EnumParam(0);
    this->m_smaa_view.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Result");
    this->m_smaa_view.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "Edges");
    this->m_smaa_view.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "Weights");
    this->m_smaa_view.Param<megamol::core::param::EnumParam>()->SetTypePair(3, "AreaTex");
    this->m_smaa_view.Param<megamol::core::param::EnumParam>()->SetTypePair(4, "SearchTex");
    this->MakeSlotAvailable(&this->m_smaa_view);

    this->m_output_tex_slot.SetCallback(CallTexture2D::ClassName(), "GetData", &AntiAliasing::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &AntiAliasing::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_input_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_slot);
}

megamol::compositing::AntiAliasing::~AntiAliasing() { this->Release(); }

bool megamol::compositing::AntiAliasing::create() {
    try {
        // create shader program
        m_fxaa_prgm = std::make_unique<GLSLComputeShader>();
        m_smaa_velocity_prgm = std::make_unique<GLSLComputeShader>();
        m_smaa_edge_detection_prgm = std::make_unique<GLSLComputeShader>();
        m_smaa_blending_weight_calculation_prgm = std::make_unique<GLSLComputeShader>();
        m_smaa_neighborhood_blending_prgm = std::make_unique<GLSLComputeShader>();

        vislib::graphics::gl::ShaderSource compute_fxaa_src;
        vislib::graphics::gl::ShaderSource compute_smaa_velocity_src;
        vislib::graphics::gl::ShaderSource compute_smaa_edge_detection_src;
        vislib::graphics::gl::ShaderSource compute_smaa_blending_weights_src;
        vislib::graphics::gl::ShaderSource compute_smaa_neighborhood_blending_src;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::fxaa", compute_fxaa_src))
            return false;
        if (!m_fxaa_prgm->Compile(compute_fxaa_src.Code(), compute_fxaa_src.Count()))
            return false;
        if (!m_fxaa_prgm->Link())
            return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::smaa::velocityCS", compute_smaa_velocity_src))
            return false;
        if (!m_smaa_velocity_prgm->Compile(compute_smaa_velocity_src.Code(), compute_smaa_velocity_src.Count()))
            return false;
        if (!m_smaa_velocity_prgm->Link())
            return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::smaa::edgeDetectionCS", compute_smaa_edge_detection_src))
            return false;
        if (!m_smaa_edge_detection_prgm->Compile(compute_smaa_edge_detection_src.Code(), compute_smaa_edge_detection_src.Count()))
            return false;
        if (!m_smaa_edge_detection_prgm->Link())
            return false;
        
        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::smaa::blendingWeightsCalculationCS", compute_smaa_blending_weights_src))
            return false;
        if (!m_smaa_blending_weight_calculation_prgm->Compile(compute_smaa_blending_weights_src.Code(), compute_smaa_blending_weights_src.Count()))
            return false;
        if (!m_smaa_blending_weight_calculation_prgm->Link())
            return false;
        
        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::smaa::neighborhoodBlendingCS", compute_smaa_neighborhood_blending_src))
            return false;
        if (!m_smaa_neighborhood_blending_prgm->Compile(compute_smaa_neighborhood_blending_src.Code(), compute_smaa_neighborhood_blending_src.Count()))
            return false;
        if (!m_smaa_neighborhood_blending_prgm->Link())
            return false;
        
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


    // TODO: we are using 64 bit colors here, see the note in the shader instructions about point sampling
    glowl::TextureLayout tx_layout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_HALF_FLOAT, 1);
    m_output_texture = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);

    // texture for smaa
    std::vector<std::pair<GLenum, GLint>> int_params = {
        {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST},
        {GL_TEXTURE_MAG_FILTER, GL_LINEAR} };
    m_smaa_layout = glowl::TextureLayout(GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1, int_params, {});
    m_edges_tex = std::make_shared<glowl::Texture2D>("smaa_edges_tex", m_smaa_layout, nullptr);
    m_blend_tex = std::make_shared<glowl::Texture2D>("smaa_blend_tex", m_smaa_layout, nullptr);

    glowl::TextureLayout area_layout(GL_RG8, AREATEX_WIDTH, AREATEX_HEIGHT, 1, GL_RG, GL_UNSIGNED_BYTE, 1, int_params, {});
    glowl::TextureLayout search_layout(GL_R8, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, 1, GL_RED, GL_UNSIGNED_BYTE, 1, int_params, {});
    glowl::TextureLayout velocity_layout(GL_RG8, 0, 0, 1, GL_RG, GL_UNSIGNED_BYTE, 1);


    // need to flip image around horizontal axis
    //m_area.resize(AREATEX_SIZE);
    //for (size_t y = 0; y < AREATEX_HEIGHT; ++y) {
    //    for (size_t x = 0; x < AREATEX_WIDTH; ++x) {
    //        size_t id = x + y * AREATEX_WIDTH;

    //        size_t flip_id = x + (AREATEX_HEIGHT - 1 - y) * AREATEX_WIDTH;

    //        m_area[2 * id + 0] = areaTexBytes[2 * flip_id + 0];     // R
    //        m_area[2 * id + 1] = areaTexBytes[2 * flip_id + 1]; // R
    //    }
    //}

    //m_search.resize(SEARCHTEX_SIZE);
    //for (size_t y = 0; y < SEARCHTEX_HEIGHT; ++y) {
    //    for (size_t x = 0; x < SEARCHTEX_WIDTH; ++x) {
    //        size_t id = x + y * SEARCHTEX_WIDTH;

    //        size_t flip_id = x + (SEARCHTEX_HEIGHT - 1 - y) * SEARCHTEX_WIDTH;

    //        m_search[id + 0] = searchTexBytes[flip_id + 0]; // R
    //    }
    //}

    // TODO: flip y coordinate in texture accesses in shadercode and also flip textures here?
    m_area_tex = std::make_shared<glowl::Texture2D>("smaa_area_tex", area_layout, areaTexBytes);
    m_search_tex = std::make_shared<glowl::Texture2D>("smaa_search_tex", search_layout, searchTexBytes);
    m_velocity_tex = std::make_shared<glowl::Texture2D>("smaa_input_velocity", velocity_layout, nullptr);

    m_ssbo_constants = std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    return true;
}

void megamol::compositing::AntiAliasing::release() {}

bool megamol::compositing::AntiAliasing::setSettingsCallback(core::param::ParamSlot& slot) {
    // low
    if (slot.Param<core::param::EnumParam>()->Value() == 0) {
        m_smaa_constants.Smaa_threshold = 0.15f;
        m_smaa_constants.Smaa_depth_threshold = 0.1f * m_smaa_constants.Smaa_threshold;
        m_smaa_constants.Max_search_steps = 4;
        m_smaa_constants.Max_search_steps_diag = 8;
        m_smaa_constants.Disable_diag_detection = true;
        m_smaa_constants.Disable_corner_detection = true;
        m_smaa_constants.Corner_rounding = 25;
        m_smaa_constants.Corner_rounding_norm = m_smaa_constants.Corner_rounding / 100.f;
    }
    // medium
    else if (slot.Param<core::param::EnumParam>()->Value() == 1) {
        m_smaa_constants.Smaa_threshold = 0.1f;
        m_smaa_constants.Smaa_depth_threshold = 0.1f * m_smaa_constants.Smaa_threshold;
        m_smaa_constants.Max_search_steps = 8;
        m_smaa_constants.Max_search_steps_diag = 8;
        m_smaa_constants.Disable_diag_detection = true;
        m_smaa_constants.Disable_corner_detection = true;
        m_smaa_constants.Corner_rounding = 25;
        m_smaa_constants.Corner_rounding_norm = m_smaa_constants.Corner_rounding / 100.f;
    }
    // high
    else if (slot.Param<core::param::EnumParam>()->Value() == 2) {
        m_smaa_constants.Smaa_threshold = 0.1f;
        m_smaa_constants.Smaa_depth_threshold = 0.1f * m_smaa_constants.Smaa_threshold;
        m_smaa_constants.Max_search_steps = 16;
        m_smaa_constants.Max_search_steps_diag = 8;
        m_smaa_constants.Disable_diag_detection = false;
        m_smaa_constants.Disable_corner_detection = false;
        m_smaa_constants.Corner_rounding = 25;
        m_smaa_constants.Corner_rounding_norm = m_smaa_constants.Corner_rounding / 100.f;
    }
    // ultra
    else if (slot.Param<core::param::EnumParam>()->Value() == 3) {
        m_smaa_constants.Smaa_threshold = 0.05f;
        m_smaa_constants.Smaa_depth_threshold = 0.1f * m_smaa_constants.Smaa_threshold;
        m_smaa_constants.Max_search_steps = 32;
        m_smaa_constants.Max_search_steps_diag = 16;
        m_smaa_constants.Disable_diag_detection = false;
        m_smaa_constants.Disable_corner_detection = false;
        m_smaa_constants.Corner_rounding = 25;
        m_smaa_constants.Corner_rounding_norm = m_smaa_constants.Corner_rounding / 100.f;
    }


    // custom
    if (slot.Param<core::param::EnumParam>()->Value() == 4) {
        this->m_smaa_threshold.Param<core::param::FloatParam>()->SetValue(m_smaa_custom_constants.Smaa_threshold);
        this->m_smaa_max_search_steps.Param<core::param::IntParam>()->SetValue(
            m_smaa_custom_constants.Max_search_steps);
        this->m_smaa_max_search_steps_diag.Param<core::param::IntParam>()->SetValue(
            m_smaa_custom_constants.Max_search_steps_diag);
        this->m_smaa_disable_diag_detection.Param<core::param::BoolParam>()->SetValue(
            m_smaa_custom_constants.Disable_diag_detection);
        this->m_smaa_disable_corner_detection.Param<core::param::BoolParam>()->SetValue(
            m_smaa_custom_constants.Disable_corner_detection);
        this->m_smaa_corner_rounding.Param<core::param::IntParam>()->SetValue(m_smaa_custom_constants.Corner_rounding);

        m_smaa_constants = m_smaa_custom_constants;

        this->m_smaa_threshold.Param<core::param::FloatParam>()->SetGUIVisible(true);
        this->m_smaa_max_search_steps.Param<core::param::IntParam>()->SetGUIVisible(true);
        this->m_smaa_max_search_steps_diag.Param<core::param::IntParam>()->SetGUIVisible(true);
        this->m_smaa_disable_diag_detection.Param<core::param::BoolParam>()->SetGUIVisible(true);
        this->m_smaa_disable_corner_detection.Param<core::param::BoolParam>()->SetGUIVisible(true);
        this->m_smaa_corner_rounding.Param<core::param::IntParam>()->SetGUIVisible(true);
    } else {
        this->m_smaa_threshold.Param<core::param::FloatParam>()->SetGUIVisible(false);
        this->m_smaa_max_search_steps.Param<core::param::IntParam>()->SetGUIVisible(false);
        this->m_smaa_max_search_steps_diag.Param<core::param::IntParam>()->SetGUIVisible(false);
        this->m_smaa_disable_diag_detection.Param<core::param::BoolParam>()->SetGUIVisible(false);
        this->m_smaa_disable_corner_detection.Param<core::param::BoolParam>()->SetGUIVisible(false);
        this->m_smaa_corner_rounding.Param<core::param::IntParam>()->SetGUIVisible(false);
    }

    m_settings_have_changed = true;

    return true;
}

bool megamol::compositing::AntiAliasing::setCustomSettingsCallback(core::param::ParamSlot& slot) {
    m_smaa_constants.Smaa_threshold = this->m_smaa_threshold.Param<core::param::FloatParam>()->Value();
    m_smaa_constants.Smaa_depth_threshold = 0.1f * m_smaa_constants.Smaa_threshold;
    m_smaa_constants.Max_search_steps = this->m_smaa_max_search_steps.Param<core::param::IntParam>()->Value();
    m_smaa_constants.Max_search_steps_diag = this->m_smaa_max_search_steps_diag.Param<core::param::IntParam>()->Value();
    m_smaa_constants.Disable_diag_detection =
        this->m_smaa_disable_diag_detection.Param<core::param::BoolParam>()->Value();
    m_smaa_constants.Disable_corner_detection =
        this->m_smaa_disable_corner_detection.Param<core::param::BoolParam>()->Value();
    m_smaa_constants.Corner_rounding = this->m_smaa_corner_rounding.Param<core::param::IntParam>()->Value();
    m_smaa_constants.Corner_rounding_norm = m_smaa_constants.Corner_rounding / 100.f;

    // keep a backup from the custom settings, so if custom is selected again
    // the previous values are loaded
    m_smaa_custom_constants = m_smaa_constants;

    m_settings_have_changed = true;

    return true;
}

bool megamol::compositing::AntiAliasing::visibilityCallback(core::param::ParamSlot& slot) {
    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0) {
        m_smaa_quality.Param<core::param::EnumParam>()->SetGUIVisible(true);
        m_smaa_detection_technique.Param<core::param::EnumParam>()->SetGUIVisible(true);
        m_smaa_view.Param<core::param::EnumParam>()->SetGUIVisible(true);
        m_smaa_mode.Param<core::param::EnumParam>()->SetGUIVisible(true);
    } else {
        m_smaa_quality.Param<core::param::EnumParam>()->SetGUIVisible(false);
        m_smaa_detection_technique.Param<core::param::EnumParam>()->SetGUIVisible(false);
        m_smaa_view.Param<core::param::EnumParam>()->SetGUIVisible(false);
        m_smaa_mode.Param<core::param::EnumParam>()->SetGUIVisible(false);
    }

    m_settings_have_changed = true;

    return true;
}

void megamol::compositing::AntiAliasing::dispatchComputeShader(
    const std::unique_ptr<GLSLComputeShader>& prgm,
    const std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, const char*>>& inputs,
    std::shared_ptr<glowl::Texture2D> output,
    const std::vector<std::pair<const char*, int>>& uniforms,
    const std::shared_ptr<glowl::BufferObject>& ssbo) {
    prgm->Enable();

    for (size_t i = 0; i < inputs.size(); ++i) {
        glActiveTexture(GL_TEXTURE0 + i);
        inputs[i].first->bindTexture();
        glUniform1i(prgm->ParameterLocation(inputs[i].second), i);
    }

    for (const auto& uniform : uniforms) {
        glUniform1i(prgm->ParameterLocation(uniform.first), uniform.second);
    }

    if (ssbo != nullptr) {
        ssbo->bind(0);
    }

    output->bindImage(0, GL_WRITE_ONLY);

    prgm->Dispatch(static_cast<int>(std::ceil(output->getWidth() / 8.0f)),
        static_cast<int>(std::ceil(output->getHeight() / 8.0f)), 1);

    prgm->Disable();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

bool megamol::compositing::AntiAliasing::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    auto call_input = m_input_tex_slot.CallAs<CallTexture2D>();

    if (lhs_tc == NULL) return false;
    
    if(call_input != NULL) { if (!(*call_input)(0)) return false; }

    bool something_has_changed =
        (call_input != NULL ? call_input->hasUpdate() : false)
        || m_settings_have_changed
        || this->m_smaa_detection_technique.IsDirty();

    if (something_has_changed) {
        ++m_version;
        m_settings_have_changed = false;
        this->m_smaa_detection_technique.ResetDirty();

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

        auto input_tx2D = call_input->getData();
        setupOutputTexture(input_tx2D, m_output_texture);

        int mode = this->m_mode.Param<core::param::EnumParam>()->Value();
        bool temporal = this->m_smaa_mode.Param<core::param::EnumParam>()->Value() == 1;

        // smaa
        if (mode == 0) {
            int input_width = input_tx2D->getWidth();
            int input_height = input_tx2D->getHeight();

            // init textures and clear them
            m_smaa_layout.width = input_width;
            m_smaa_layout.height = input_height;
            m_edges_tex->reload(m_smaa_layout, nullptr);
            m_blend_tex->reload(m_smaa_layout, nullptr);
            GLubyte col[4] = { 0, 0, 0, 0 };
            m_edges_tex->clearTexImage(col);
            m_blend_tex->clearTexImage(col);

            m_smaa_constants.Rt_metrics = glm::vec4(
                1.f / (float) input_width, 1.f / (float) input_height, (float) input_width, (float) input_height);
            m_ssbo_constants->rebuffer(&m_smaa_constants, sizeof(m_smaa_constants));

            GLint technique = m_smaa_detection_technique.Param<core::param::EnumParam>()->Value();
            GLint preset = m_smaa_quality.Param<core::param::EnumParam>()->Value();


            // TODO: one program for all? one mega shaders with barriers?

            if (temporal) {
                // calculate velocity tex for temporal smaa (SMAA S2x)
                glowl::TextureLayout ly = m_velocity_tex->getTextureLayout();
                ly.width = input_width;
                ly.height = input_height;
                m_velocity_tex->reload(ly, nullptr);

                m_smaa_velocity_prgm->Enable();

                glActiveTexture(GL_TEXTURE0);
                currDepth_tx2D->bindTexture();
                glUniform1i(m_smaa_velocity_prgm->ParameterLocation("g_currDepthTex"), 0);
                glActiveTexture(GL_TEXTURE1);
                prevDepth_tx2D->bindTexture();
                glUniform1i(m_smaa_velocity_prgm->ParameterLocation("g_prevDepthTex"), 1);
                //glUniformMatrix4fv(m_smaa_velocity_prgm->ParameterLocation("currViewProjMx"), 1, false, currViewProjMx);
                //glUniformMatrix4fv(m_smaa_velocity_prgm->ParameterLocation("prevViewProjMx"), 1, false, prevViewProjMx);
                //glUniform2fv(m_smaa_velocity_prgm->ParameterLocation("jitter"), 1, jitter);

                m_velocity_tex->bindImage(0, GL_WRITE_ONLY);

                m_smaa_velocity_prgm->Dispatch(static_cast<int>(std::ceil(m_velocity_tex->getWidth() / 8.0f)),
                    static_cast<int>(std::ceil(m_velocity_tex->getHeight() / 8.0f)), 1);

                m_smaa_velocity_prgm->Disable();

                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, 0);

                //dispatchComputeShader(m_smaa_velocity_prgm, { { input_tx2D, "g_depthTex" } }, m_velocity_tex, {});
            }
            // edge detection
            // TODO: the original paper uses a stencil buffer here in the first pass to optimize the next pass
            std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, const char*>> inputs = {{input_tx2D, "g_colorTex"}};
            std::vector<std::pair<const char*, int>> uniforms = {{"technique", technique}};

            dispatchComputeShader(m_smaa_edge_detection_prgm, inputs, m_edges_tex, uniforms, m_ssbo_constants);

            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

            // blending weights calculation
            inputs.clear();
            inputs = {{m_edges_tex, "g_edgesTex"}, {m_area_tex, "g_areaTex"}, {m_search_tex, "g_searchTex"}};

            dispatchComputeShader(m_smaa_blending_weight_calculation_prgm, inputs, m_blend_tex, {}, m_ssbo_constants);

            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

            // final step: neighborhood blending
            inputs.clear();
            inputs = {{input_tx2D, "g_colorTex"}, {m_blend_tex, "g_blendingWeightsTex"}};

            dispatchComputeShader(
                m_smaa_neighborhood_blending_prgm, inputs, m_output_texture, {}, m_ssbo_constants);
            // only used for temporal reprojection
            /*glActiveTexture(GL_TEXTURE2);
            m_velocity_tex->bindTexture();
            glUniform1i(m_smaa_neighborhood_blending_prgm->ParameterLocation("g_velocityTex"), 2);*/


            // TODO: in smaaneighborhoodblending the reads and writes must be in srgb (and only there!)
        }
        // fxaa
        else if (mode == 1) {
            std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, const char*>> inputs = {{input_tx2D, "src_tx2D"}};
            std::vector<std::pair<const char*, int>> uniforms = {{"disable_aa", 0}};

            dispatchComputeShader(m_fxaa_prgm, inputs, m_output_texture, uniforms);
        }
        // no aa
        else if (mode == 2) {
            // TODO: find a better solution
            std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, const char*>> inputs = {{input_tx2D, "src_tx2D"}};
            std::vector<std::pair<const char*, int>> uniforms = {{"disable_aa", 1}};

            dispatchComputeShader(m_fxaa_prgm, inputs, m_output_texture, uniforms);
        }
    }

    if (lhs_tc->version() < m_version || this->m_smaa_view.IsDirty()) {
        int view = this->m_smaa_view.Param<core::param::EnumParam>()->Value();

        switch (view) {
        case 0:
            lhs_tc->setData(m_output_texture, m_version);
            break;
        case 1:
            lhs_tc->setData(m_edges_tex, m_version);
            break;
        case 2:
            lhs_tc->setData(m_blend_tex, m_version);
            break;
        case 3:
            lhs_tc->setData(m_area_tex, m_version);
            break;
        case 4:
            lhs_tc->setData(m_search_tex, m_version);
            break;
        default:
            lhs_tc->setData(m_output_texture, m_version);
            break;
        }

        this->m_smaa_view.ResetDirty();
    }

    return true;
}

bool megamol::compositing::AntiAliasing::getMetaDataCallback(core::Call& caller) { return true; }
