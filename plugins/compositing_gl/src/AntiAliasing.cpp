/*
 * AntiAliasing.cpp
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

// TOOD: consistent naming (e.g. tex or tx2D, camelCasing or underscore)

#define TIMER_ENABLED 0

#include "stdafx.h"
#include "AntiAliasing.h"

#include <array>
#include <random>

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"

#include "vislib/graphics/gl/ShaderSource.h"
#include "mmcore/PerformanceQueryManager.h"

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
    , m_smaa_predication("Predication", "Used to counter ghosting, which gets introduced in SMAAT2x")
    , m_smaa_view("Output", "Sets the texture to view: final output, edges or weights. Edges or weights should "
                            "only be used when directly connected to the screen for debug purposes")
    , m_output_tex_slot("OutputTexture", "Gives access to the resulting output texture")
    , m_input_tex_slot("InputTexture", "Connects the input texture")
    , m_camera_slot("Camera", "Connects the camera")
    , m_depth_tex_slot("DepthTexture", "Connects the depth texture")
    , m_settings_have_changed(false)
        , m_cnt(0), m_totalTimeSpent(0.0) {
    this->m_mode << new megamol::core::param::EnumParam(0);
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "SMAA");
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "FXAA");
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "None");
    this->m_mode.SetUpdateCallback(&megamol::compositing::AntiAliasing::visibilityCallback);
    this->MakeSlotAvailable(&this->m_mode);

    this->m_smaa_mode << new megamol::core::param::EnumParam(1);
    this->m_smaa_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "SMAA 1x");
    this->m_smaa_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "SMAA T2x");
    // S2x and 4x requires multisampling at the connected renderer
    // and is therefore currently omitted
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

    this->m_smaa_predication << new megamol::core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->m_smaa_predication);

    this->m_smaa_view << new megamol::core::param::EnumParam(0);
    this->m_smaa_view.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Result");
    this->m_smaa_view.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "Edges");
    this->m_smaa_view.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "Weights");
    this->m_smaa_view.Param<megamol::core::param::EnumParam>()->SetTypePair(3, "Velocity");
    this->m_smaa_view.Param<megamol::core::param::EnumParam>()->SetTypePair(4, "Temporal");
    this->m_smaa_view.Param<megamol::core::param::EnumParam>()->SetTypePair(5, "PrevInput");
    this->MakeSlotAvailable(&this->m_smaa_view);

    this->m_output_tex_slot.SetCallback(CallTexture2D::ClassName(), "GetData", &AntiAliasing::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &AntiAliasing::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_input_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_slot);

    this->m_camera_slot.SetCompatibleCall<CallCameraDescription>();
    this->MakeSlotAvailable(&this->m_camera_slot);

    this->m_depth_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_depth_tex_slot);
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
        m_smaa_temporal_resolving_prgm = std::make_unique<GLSLComputeShader>();
        m_copy_prgm = std::make_unique<GLSLComputeShader>();

        vislib::graphics::gl::ShaderSource compute_fxaa_src;
        vislib::graphics::gl::ShaderSource compute_smaa_velocity_src;
        vislib::graphics::gl::ShaderSource compute_smaa_edge_detection_src;
        vislib::graphics::gl::ShaderSource compute_smaa_blending_weights_src;
        vislib::graphics::gl::ShaderSource compute_smaa_neighborhood_blending_src;
        vislib::graphics::gl::ShaderSource compute_smaa_temporal_resolving_src;
        vislib::graphics::gl::ShaderSource compute_copy_src;

        // TODO: look into and use new shaderfactory (see e.g. simplstsphererenderer)
        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::copy", compute_copy_src))
            return false;
        if (!m_copy_prgm->Compile(compute_copy_src.Code(), compute_copy_src.Count()))
            return false;
        if (!m_copy_prgm->Link())
            return false;

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

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::smaa::temporalResolvingCS", compute_smaa_temporal_resolving_src))
            return false;
        if (!m_smaa_temporal_resolving_prgm->Compile(compute_smaa_temporal_resolving_src.Code(), compute_smaa_temporal_resolving_src.Count()))
            return false;
        if (!m_smaa_temporal_resolving_prgm->Link())
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
    m_prev_input_tx2D = std::make_shared<glowl::Texture2D>("prev_frame_input_texture", tx_layout, nullptr);
    m_temporal_tex = std::make_shared<glowl::Texture2D>("intermediate_temporal_texture", tx_layout, nullptr);

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
    glowl::TextureLayout velocity_layout(GL_RG16F, 1, 1, 1, GL_RG, GL_HALF_FLOAT, 1);
    glowl::TextureLayout depth_layout(GL_DEPTH_COMPONENT24_ARB, 1, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, 1);


    // need to flip image around horizontal axis
    //m_area.resize(AREATEX_SIZE);
    //for (size_t y = 0; y < AREATEX_HEIGHT; ++y) {
    //    for (size_t x = 0; x < AREATEX_WIDTH; ++x) {
    //        size_t id = x + y * AREATEX_WIDTH;

    //        size_t flip_id = x + (AREATEX_HEIGHT - 1 - y) * AREATEX_WIDTH;

    //        m_area[2 * id + 0] = areaTexBytes[2 * flip_id + 0]; // R
    //        m_area[2 * id + 1] = areaTexBytes[2 * flip_id + 1]; // G
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
    m_prev_depth_tx2D = std::make_shared<glowl::Texture2D>("smaa_prev_depth", depth_layout, nullptr);

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
    // smaa enabled
    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0) {
        m_smaa_quality.Param<core::param::EnumParam>()->SetGUIVisible(true);
        m_smaa_detection_technique.Param<core::param::EnumParam>()->SetGUIVisible(true);
        m_smaa_view.Param<core::param::EnumParam>()->SetGUIVisible(true);
        m_smaa_mode.Param<core::param::EnumParam>()->SetGUIVisible(true);
    }
    // smaa disabled
    else {
        m_smaa_quality.Param<core::param::EnumParam>()->SetGUIVisible(false);
        m_smaa_detection_technique.Param<core::param::EnumParam>()->SetGUIVisible(false);
        m_smaa_view.Param<core::param::EnumParam>()->SetGUIVisible(false);
        m_smaa_mode.Param<core::param::EnumParam>()->SetGUIVisible(false);
    }

    // smaat2x enabled
    if (this->m_smaa_mode.Param<core::param::EnumParam>()->Value() == 1) {
        m_smaa_predication.Param<core::param::BoolParam>()->SetGUIVisible(true);
    }
    // smaat2x disabled
    else {
        m_smaa_predication.Param<core::param::BoolParam>()->SetGUIVisible(false);
    }

    m_settings_have_changed = true;

    return true;
}

void megamol::compositing::AntiAliasing::dispatchComputeShader(
    const std::unique_ptr<GLSLComputeShader>& prgm,
    const std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, const char*>>& inputs,
    std::shared_ptr<glowl::Texture2D> output,
    const std::vector<std::pair<const char*, int>>& uniforms,
    bool calc_weights_pass,
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

    if (calc_weights_pass) {
        // smaat2x enabled
        if (this->m_smaa_mode.Param<core::param::EnumParam>()->Value() == 1) {
            glUniform4fv(prgm->ParameterLocation("g_subsampleIndices"), 1, glm::value_ptr(m_subsampleIndices[m_version % 2]));
        }
        // smaat2x disabled
        else {
            glUniform4fv(
                prgm->ParameterLocation("g_subsampleIndices"), 1, glm::value_ptr(glm::vec4(0.0)));
        }
    }

    if (ssbo != nullptr) {
        ssbo->bind(0);
    }

    output->bindImage(0, GL_WRITE_ONLY);

    prgm->Dispatch(static_cast<int>(std::ceil(output->getWidth() / 8.0f)),
        static_cast<int>(std::ceil(output->getHeight() / 8.0f)), 1);

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    prgm->Disable();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void megamol::compositing::AntiAliasing::copyTextureViaShader(
    const std::shared_ptr<glowl::Texture2D>& src, const std::shared_ptr<glowl::Texture2D>& tgt) {
    m_copy_prgm->Enable();

    glActiveTexture(GL_TEXTURE0);
    src->bindTexture();
    glUniform1i(m_copy_prgm->ParameterLocation("src_tx2D"), 0);

    tgt->bindImage(0, GL_WRITE_ONLY);

    m_copy_prgm->Dispatch(
        static_cast<int>(std::ceil(tgt->getWidth() / 8.0f)), static_cast<int>(std::ceil(tgt->getHeight() / 8.0f)), 1);

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    m_copy_prgm->Disable();

    glBindTexture(GL_TEXTURE_2D, 0);
}

bool megamol::compositing::AntiAliasing::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    auto rhs_call_input = m_input_tex_slot.CallAs<CallTexture2D>();
    auto rhs_call_camera = m_camera_slot.CallAs<CallCamera>();
    auto rhs_call_depth = m_depth_tex_slot.CallAs<CallTexture2D>();

    if (lhs_tc == NULL) return false;
    
    if(rhs_call_input != NULL) { if (!(*rhs_call_input)(0)) return false; }

    bool temporal = this->m_smaa_mode.Param<core::param::EnumParam>()->Value() == 1;
    if (temporal) {
        if (rhs_call_camera != NULL) {
            if (!(*rhs_call_camera)(0)) {
                return false;
            }
        }
        else {
            core::utility::log::Log::DefaultLog.WriteError("Temporal SMAA requires the camera to be plugged in.");
            return false;
        }

        if (rhs_call_depth != NULL) {
            if (!(*rhs_call_depth)(0)) {
                return false;
            }
        }
        else {
            core::utility::log::Log::DefaultLog.WriteError("Temporal SMAA requires the depth texture of the"
                " rendered mesh. Check the depth texture slot.");
            return false;
        }
    }

    GLint technique = m_smaa_detection_technique.Param<core::param::EnumParam>()->Value();
    // depth based edge detection requires the depth texture
    if (technique == 2) {
        if (rhs_call_depth != NULL) {
            if (!(*rhs_call_depth)(0)) {
                return false;
            }
        }
        else {
            core::utility::log::Log::DefaultLog.WriteError("Depth based edge detection for SMAA requires the depth texture of the"
                " rendered mesh. Check the depth texture slot.");
            return false;
        }
    }

    bool something_has_changed =
        (rhs_call_input != NULL ? rhs_call_input->hasUpdate() : false)
        || m_settings_have_changed
        || this->m_smaa_detection_technique.IsDirty()
        || (temporal ? rhs_call_camera->hasUpdate() : false)
        || (temporal ? rhs_call_depth->hasUpdate()  : false)
        || (technique == 2 ? rhs_call_depth->hasUpdate() : false);

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

        auto input_tx2D = rhs_call_input->getData();
        setupOutputTexture(input_tx2D, m_output_texture);
        if(temporal) setupOutputTexture(input_tx2D, m_prev_input_tx2D);

        int mode = this->m_mode.Param<core::param::EnumParam>()->Value();

        // smaa
        if (mode == 0) {
            int input_width = input_tx2D->getWidth();
            int input_height = input_tx2D->getHeight();

            if (temporal || technique == 2) {
                m_depth_tx2D = rhs_call_depth->getData();
                if ((m_prev_depth_tx2D->getWidth() != input_width)
                    || (m_prev_depth_tx2D->getHeight() != input_height)) {
                    glowl::TextureLayout ly = m_prev_depth_tx2D->getTextureLayout();
                    ly.width = input_width;
                    ly.height = input_height;
                    m_prev_depth_tx2D->reload(ly, nullptr);
                }

                if ((m_velocity_tex->getWidth() != input_width)
                    || (m_velocity_tex->getHeight() != input_height)) {
                    glowl::TextureLayout ly = m_velocity_tex->getTextureLayout();
                    ly.width = input_width;
                    ly.height = input_height;
                    m_velocity_tex->reload(ly, nullptr);
                }

                if ((m_temporal_tex->getWidth() != input_width)
                    || (m_temporal_tex->getHeight() != input_height)) {
                    glowl::TextureLayout ly = m_temporal_tex->getTextureLayout();
                    ly.width = input_width;
                    ly.height = input_height;
                    m_temporal_tex->reload(ly, nullptr);
                }
            }


            // init textures
            if ((input_width != m_smaa_layout.width) || (input_height != m_smaa_layout.height)) {
                m_smaa_layout.width = input_width;
                m_smaa_layout.height = input_height;
                m_edges_tex->reload(m_smaa_layout, nullptr);
                m_blend_tex->reload(m_smaa_layout, nullptr); 
            }

            // always clear them
            GLubyte col[4] = { 0, 0, 0, 0 };
            m_edges_tex->clearTexImage(col);
            m_blend_tex->clearTexImage(col);

            m_smaa_constants.Rt_metrics = glm::vec4(
                1.f / (float) input_width, 1.f / (float) input_height, (float) input_width, (float) input_height);
            m_ssbo_constants->rebuffer(&m_smaa_constants, sizeof(m_smaa_constants));

            
            GLint preset = m_smaa_quality.Param<core::param::EnumParam>()->Value();


            // TODO: one program for all? one mega shader with barriers?

#if TIMER_ENABLED
            GLuint64 startTime, stopTime;
            unsigned int queryID[2];
            glGenQueries(2, queryID);
            glQueryCounter(queryID[0], GL_TIMESTAMP);
#endif

            if (temporal) {
                m_cam = rhs_call_camera->getData();
                glm::mat4 view_mx = m_cam.getViewMatrix();
                glm::mat4 proj_mx = m_cam.getProjectionMatrix();
                glm::mat4 curr_view_proj_mx = proj_mx * view_mx;

                // calculate velocity tex for temporal smaa (SMAA T2x)
                m_smaa_velocity_prgm->Enable();

                glActiveTexture(GL_TEXTURE0);
                m_depth_tx2D->bindTexture();
                glUniform1i(m_smaa_velocity_prgm->ParameterLocation("g_currDepthTex"), 0);
                glActiveTexture(GL_TEXTURE1);
                m_prev_depth_tx2D->bindTexture();
                glUniform1i(m_smaa_velocity_prgm->ParameterLocation("g_prevDepthTex"), 1);
                glUniformMatrix4fv(m_smaa_velocity_prgm->ParameterLocation("currProjMx"),
                    1, false, glm::value_ptr(proj_mx));
                glUniformMatrix4fv(m_smaa_velocity_prgm->ParameterLocation("currViewMx"), 1, false,
                    glm::value_ptr(view_mx));
                glUniformMatrix4fv(m_smaa_velocity_prgm->ParameterLocation("prevProjMx"),
                    1, false, glm::value_ptr(m_prev_proj_mx));
                glUniformMatrix4fv(
                    m_smaa_velocity_prgm->ParameterLocation("prevViewMx"), 1, false, glm::value_ptr(m_prev_view_mx));
                glUniform2fv(m_smaa_velocity_prgm->ParameterLocation("jitter"), 1, glm::value_ptr(m_jitter[m_version % 2]));

                m_velocity_tex->bindImage(0, GL_WRITE_ONLY);

                m_smaa_velocity_prgm->Dispatch(static_cast<int>(std::ceil(input_width / 8.0f)),
                    static_cast<int>(std::ceil(input_height / 8.0f)), 1);

                m_smaa_velocity_prgm->Disable();

                glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

                m_prev_proj_mx = proj_mx;
                m_prev_view_mx = view_mx;
                glowl::Texture2D::copy(m_depth_tx2D.get(), m_prev_depth_tx2D.get());


                // only used for temporal reprojection
                m_smaa_temporal_resolving_prgm->Enable();

                glActiveTexture(GL_TEXTURE0);
                input_tx2D->bindTexture();
                glUniform1i(m_smaa_temporal_resolving_prgm->ParameterLocation("g_currColorTex"), 0);
                glActiveTexture(GL_TEXTURE1);
                m_prev_input_tx2D->bindTexture();
                glUniform1i(m_smaa_temporal_resolving_prgm->ParameterLocation("g_prevColorTex"), 1);
                glActiveTexture(GL_TEXTURE2);
                m_velocity_tex->bindTexture();
                glUniform1i(m_smaa_temporal_resolving_prgm->ParameterLocation("g_velocityTex"), 2);

                glUniform1i(m_smaa_temporal_resolving_prgm->ParameterLocation("SMAA_REPROJECTION"),
                    m_smaa_predication.Param<core::param::BoolParam>()->Value());

                m_ssbo_constants->bind(0);

                m_temporal_tex->bindImage(0, GL_WRITE_ONLY);

                m_smaa_temporal_resolving_prgm->Dispatch(static_cast<int>(std::ceil(input_width / 8.0f)),
                    static_cast<int>(std::ceil(input_height / 8.0f)), 1);

                m_smaa_temporal_resolving_prgm->Disable();

                glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, 0);

                copyTextureViaShader(input_tx2D, m_prev_input_tx2D);
            }

            // edge detection
            std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, const char*>> inputs =
                {{temporal ? m_temporal_tex : input_tx2D, "g_colorTex"}};
            if (technique == 2) inputs.push_back({ m_depth_tx2D, "g_depthTex" });
            std::vector<std::pair<const char*, int>> uniforms = {{"technique", technique}};

            dispatchComputeShader(m_smaa_edge_detection_prgm, inputs, m_edges_tex, uniforms, false, m_ssbo_constants);

            // blending weights calculation
            inputs.clear();
            inputs = {{m_edges_tex, "g_edgesTex"}, {m_area_tex, "g_areaTex"}, {m_search_tex, "g_searchTex"}};

            dispatchComputeShader(m_smaa_blending_weight_calculation_prgm, inputs, m_blend_tex, {}, true, m_ssbo_constants);

            // final step: neighborhood blending
            inputs.clear();
            inputs = {{input_tx2D, "g_colorTex"}, {m_blend_tex, "g_blendingWeightsTex"}};

            dispatchComputeShader(
                m_smaa_neighborhood_blending_prgm, inputs, m_output_texture, {}, false, m_ssbo_constants);

#if TIMER_ENABLED
            glQueryCounter(queryID[1], GL_TIMESTAMP);

            GLint stopTimerAvailable = 0;
            while (!stopTimerAvailable) {
                glGetQueryObjectiv(queryID[1], GL_QUERY_RESULT_AVAILABLE, &stopTimerAvailable);
            }

            glGetQueryObjectui64v(queryID[0], GL_QUERY_RESULT, &startTime);
            glGetQueryObjectui64v(queryID[1], GL_QUERY_RESULT, &stopTime);

            ++m_cnt;
            m_totalTimeSpent += (double) (stopTime - startTime) / 1000000.0;

            int numRuns = 1000;

            if (m_cnt == numRuns) {
                double avg = m_totalTimeSpent / (double) numRuns;
                megamol::core::utility::log::Log::DefaultLog.WriteInfo("Average time spent over the last %i runs: %fms", numRuns, avg);
                m_cnt = 0;
                m_totalTimeSpent = 0.0;
            }
#endif
        }
        // fxaa
        else if (mode == 1) {
            dispatchComputeShader(m_fxaa_prgm, {{input_tx2D, "src_tx2D"}}, m_output_texture, {}, false);
        }
        // no aa
        else if (mode == 2) {
            copyTextureViaShader(input_tx2D, m_output_texture);
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
            lhs_tc->setData(m_velocity_tex, m_version);
            break;
        case 4:
            lhs_tc->setData(m_temporal_tex, m_version);
            break;
        case 5:
            lhs_tc->setData(m_prev_input_tx2D, m_version);
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
