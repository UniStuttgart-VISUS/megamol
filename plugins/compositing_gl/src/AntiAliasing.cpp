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
#include <chrono>

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore_gl/utility/ShaderSourceFactory.h"
#include "compositing_gl/CompositingCalls.h"

#ifdef PROFILING
#include "PerformanceManager.h"
#endif

#include "SMAAAreaTex.h"
#include "SMAASearchTex.h"


megamol::compositing::AntiAliasing::AntiAliasing() : core::Module()
    , m_version(0)
    , m_output_tx2D(nullptr)
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
    , m_depth_tex_slot("DepthTexture", "Connects the depth texture")
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
    this->MakeSlotAvailable(&this->m_smaa_view);

    this->m_output_tex_slot.SetCallback(CallTexture2D::ClassName(), "GetData", &AntiAliasing::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &AntiAliasing::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_input_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_slot);

    this->m_depth_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_depth_tex_slot);
}

megamol::compositing::AntiAliasing::~AntiAliasing() { this->Release(); }

bool megamol::compositing::AntiAliasing::create() {
    try {
// profiling
#ifdef PROFILING
        m_perf_manager = const_cast<frontend_resources::PerformanceManager*>(
            &frontend_resources.get<frontend_resources::PerformanceManager>());

        frontend_resources::PerformanceManager::basic_timer_config render_timer;
        render_timer.name = "render";
        render_timer.api = frontend_resources::PerformanceManager::query_api::OPENGL;
        m_timers = m_perf_manager->add_timers(this, {render_timer});
#endif

        // create shader program
        m_fxaa_prgm = std::make_unique<GLSLComputeShader>();
        m_smaa_edge_detection_prgm = std::make_unique<GLSLComputeShader>();
        m_smaa_blending_weight_calculation_prgm = std::make_unique<GLSLComputeShader>();
        m_smaa_neighborhood_blending_prgm = std::make_unique<GLSLComputeShader>();
        m_copy_prgm = std::make_unique<GLSLComputeShader>();

        vislib_gl::graphics::gl::ShaderSource fxaa_src_c;
        vislib_gl::graphics::gl::ShaderSource smaa_edge_detection_src_c;
        vislib_gl::graphics::gl::ShaderSource smaa_blending_weights_src_c;
        vislib_gl::graphics::gl::ShaderSource smaa_neighborhood_blending_src_c;
        vislib_gl::graphics::gl::ShaderSource copy_src_c;

        auto ssf = std::make_shared<megamol::core_gl::utility::ShaderSourceFactory>(
            instance()->Configuration().ShaderDirectories());
        if (!ssf->MakeShaderSource("Compositing::copy", copy_src_c))
            return false;
        if (!m_copy_prgm->Compile(copy_src_c.Code(), copy_src_c.Count()))
            return false;
        if (!m_copy_prgm->Link())
            return false;

        if (!ssf->MakeShaderSource("Compositing::fxaa", fxaa_src_c))
            return false;
        if (!m_fxaa_prgm->Compile(fxaa_src_c.Code(), fxaa_src_c.Count()))
            return false;
        if (!m_fxaa_prgm->Link())
            return false;

        if (!ssf->MakeShaderSource("Compositing::smaa::edgeDetectionCS", smaa_edge_detection_src_c))
            return false;
        if (!m_smaa_edge_detection_prgm->Compile(
                smaa_edge_detection_src_c.Code(), smaa_edge_detection_src_c.Count()))
            return false;
        if (!m_smaa_edge_detection_prgm->Link())
            return false;

        if (!ssf->MakeShaderSource(
                "Compositing::smaa::blendingWeightsCalculationCS", smaa_blending_weights_src_c))
            return false;
        if (!m_smaa_blending_weight_calculation_prgm->Compile(
                smaa_blending_weights_src_c.Code(), smaa_blending_weights_src_c.Count()))
            return false;
        if (!m_smaa_blending_weight_calculation_prgm->Link())
            return false;

        if (!ssf->MakeShaderSource("Compositing::smaa::neighborhoodBlendingCS", smaa_neighborhood_blending_src_c))
            return false;
        if (!m_smaa_neighborhood_blending_prgm->Compile(
                smaa_neighborhood_blending_src_c.Code(), smaa_neighborhood_blending_src_c.Count()))
            return false;
        if (!m_smaa_neighborhood_blending_prgm->Link())
            return false;
        
    } catch (vislib_gl::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile shader (@%s): %s\n",
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

    // init all textures
    glowl::TextureLayout tx_layout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_HALF_FLOAT, 1);
    m_output_tx2D = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    
    // textures for smaa
    std::vector<std::pair<GLenum, GLint>> int_params = {
        {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST},
        {GL_TEXTURE_MAG_FILTER, GL_LINEAR} };
    m_smaa_layout = glowl::TextureLayout(GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1, int_params, {});
    glowl::TextureLayout area_layout(
        GL_RG8, AREATEX_WIDTH, AREATEX_HEIGHT, 1, GL_RG, GL_UNSIGNED_BYTE, 1, int_params, {});
    glowl::TextureLayout search_layout(
        GL_R8, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, 1, GL_RED, GL_UNSIGNED_BYTE, 1, int_params, {});

    m_edges_tx2D = std::make_shared<glowl::Texture2D>("smaa_edges_tx2D", m_smaa_layout, nullptr);
    m_blending_weights_tx2D = std::make_shared<glowl::Texture2D>("smaa_blend_tx2D", m_smaa_layout, nullptr);


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
    m_area_tx2D = std::make_shared<glowl::Texture2D>("smaa_area_tx2D", area_layout, areaTexBytes);
    m_search_tx2D = std::make_shared<glowl::Texture2D>("smaa_search_tx2D", search_layout, searchTexBytes);

    m_ssbo_constants = std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    return true;
}

void megamol::compositing::AntiAliasing::release() {
#ifdef PROFILING
    m_perf_manager->remove_timers(m_timers);
#endif
}

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

    m_settings_have_changed = true;

    return true;
}

void megamol::compositing::AntiAliasing::edgeDetection(
    const std::shared_ptr<glowl::Texture2D>& input,
    const std::shared_ptr<glowl::Texture2D>& depth,
    const std::shared_ptr<glowl::Texture2D>& edges,
    GLint detection_technique) {
    m_smaa_edge_detection_prgm->Enable();

    glActiveTexture(GL_TEXTURE0);
    input->bindTexture();
    glUniform1i(m_smaa_edge_detection_prgm->ParameterLocation("g_colorTex"), 0);

    // find edges based on the depth
    if (detection_technique == 2) {
        if (depth == nullptr) {
            core::utility::log::Log::DefaultLog.WriteError(
                "AntiAliasing::edgeDetection: depth texture is nullptr");
        } else {
            glActiveTexture(GL_TEXTURE1);
            depth->bindTexture();
            glUniform1i(m_smaa_edge_detection_prgm->ParameterLocation("g_depthTex"), 0);
        }
    }

    glUniform1i(m_smaa_edge_detection_prgm->ParameterLocation("technqiue"), detection_technique);

    m_ssbo_constants->bind(0);

    edges->bindImage(0, GL_WRITE_ONLY);

    m_smaa_edge_detection_prgm->Dispatch(static_cast<int>(std::ceil(edges->getWidth() / 8.0f)),
        static_cast<int>(std::ceil(edges->getHeight() / 8.0f)), 1);

    m_smaa_edge_detection_prgm->Disable();

    // reset gl states
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(m_ssbo_constants->getTarget(), 0);
}

void megamol::compositing::AntiAliasing::blendingWeightCalculation(
    const std::shared_ptr<glowl::Texture2D>& edges,
    const std::shared_ptr<glowl::Texture2D>& area,
    const std::shared_ptr<glowl::Texture2D>& search,
    const std::shared_ptr<glowl::Texture2D>& weights) {
    m_smaa_blending_weight_calculation_prgm->Enable();

    glActiveTexture(GL_TEXTURE0);
    edges->bindTexture();
    glUniform1i(m_smaa_blending_weight_calculation_prgm->ParameterLocation("g_edgesTex"), 0);
    glActiveTexture(GL_TEXTURE1);
    area->bindTexture();
    glUniform1i(m_smaa_blending_weight_calculation_prgm->ParameterLocation("g_areaTex"), 1);
    glActiveTexture(GL_TEXTURE2);
    search->bindTexture();
    glUniform1i(m_smaa_blending_weight_calculation_prgm->ParameterLocation("g_searchTex"), 2);

    m_ssbo_constants->bind(0);

    weights->bindImage(0, GL_WRITE_ONLY);

    m_smaa_blending_weight_calculation_prgm->Dispatch(static_cast<int>(std::ceil(weights->getWidth() / 8.0f)),
        static_cast<int>(std::ceil(weights->getHeight() / 8.0f)), 1);

    m_smaa_blending_weight_calculation_prgm->Disable();

    // reset gl states
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(m_ssbo_constants->getTarget(), 0);
}

void megamol::compositing::AntiAliasing::neighborhoodBlending(
    const std::shared_ptr<glowl::Texture2D>& input,
    const std::shared_ptr<glowl::Texture2D>& weights,
    const std::shared_ptr<glowl::Texture2D>& result) {
    m_smaa_neighborhood_blending_prgm->Enable();

    glActiveTexture(GL_TEXTURE0);
    input->bindTexture();
    glUniform1i(m_smaa_neighborhood_blending_prgm->ParameterLocation("g_colorTex"), 0);
    glActiveTexture(GL_TEXTURE1);
    weights->bindTexture();
    glUniform1i(m_smaa_neighborhood_blending_prgm->ParameterLocation("g_blendingWeightsTex"), 1);

    m_ssbo_constants->bind(0);

    result->bindImage(0, GL_WRITE_ONLY);

    m_smaa_neighborhood_blending_prgm->Dispatch(static_cast<int>(std::ceil(result->getWidth() / 8.0f)),
        static_cast<int>(std::ceil(result->getHeight() / 8.0f)), 1);

    m_smaa_neighborhood_blending_prgm->Disable();

    // reset gl states
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(m_ssbo_constants->getTarget(), 0);
}

void megamol::compositing::AntiAliasing::fxaa(
    const std::shared_ptr<glowl::Texture2D>& input,
    const std::shared_ptr<glowl::Texture2D>& output) {
    m_fxaa_prgm->Enable();

    glActiveTexture(GL_TEXTURE0);
    input->bindTexture();
    glUniform1i(m_smaa_neighborhood_blending_prgm->ParameterLocation("src_tx2D"), 0);

    output->bindImage(0, GL_WRITE_ONLY);

    m_fxaa_prgm->Dispatch(static_cast<int>(std::ceil(output->getWidth() / 8.0f)),
        static_cast<int>(std::ceil(output->getHeight() / 8.0f)), 1);

    m_fxaa_prgm->Disable();

    // reset gl states
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
    auto rhs_call_depth = m_depth_tex_slot.CallAs<CallTexture2D>();

    if (lhs_tc == NULL) return false;
    
    if(rhs_call_input != NULL) { if (!(*rhs_call_input)(0)) return false; }

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
        || (technique == 2 ? rhs_call_depth->hasUpdate() : false);

    if (something_has_changed) {
        // get input
        auto input_tx2D = rhs_call_input->getData();
        int input_width = input_tx2D->getWidth();
        int input_height = input_tx2D->getHeight();

        // init output texture if necessary
        if (m_output_tx2D->getWidth() != input_width || m_output_tx2D->getHeight() != input_height) {
            glowl::TextureLayout tx_layout(GL_RGBA16F, input_width, input_height, 1, GL_RGBA, GL_HALF_FLOAT, 1);

            m_output_tx2D->reload(tx_layout, nullptr);
        }

        // get aliasing mode: smaa, fxaa, or none
        int mode = this->m_mode.Param<core::param::EnumParam>()->Value();

        // smaa
        if (mode == 0) {
            // TODO: does this work as intended?
            // textures the smaa passes need
            std::shared_ptr<glowl::Texture2D> depth_tx2D;

            if (technique == 2) {
                depth_tx2D = rhs_call_depth->getData();
            }

            // init edge and blending weight textures
            if ((input_width != m_smaa_layout.width) || (input_height != m_smaa_layout.height)) {
                m_smaa_layout.width = input_width;
                m_smaa_layout.height = input_height;
                m_edges_tx2D->reload(m_smaa_layout, nullptr);
                m_blending_weights_tx2D->reload(m_smaa_layout, nullptr); 
            }

            // always clear them to guarantee correct textures
            GLubyte col[4] = { 0, 0, 0, 0 };
            m_edges_tx2D->clearTexImage(col);
            m_blending_weights_tx2D->clearTexImage(col);

            if (m_smaa_constants.Rt_metrics[2] != input_width
                || m_smaa_constants.Rt_metrics[3] != input_height
                || m_settings_have_changed) {
                m_smaa_constants.Rt_metrics = glm::vec4(
                    1.f / (float)input_width, 1.f / (float)input_height, (float)input_width, (float)input_height);
                m_ssbo_constants->rebuffer(&m_smaa_constants, sizeof(m_smaa_constants));
            }

#ifdef PROFILING
            m_perf_manager->start_timer(m_timers[0], this->GetCoreInstance()->GetFrameID());
#endif
            
            // perform smaa!
            edgeDetection(input_tx2D, depth_tx2D, m_edges_tx2D, technique);
            blendingWeightCalculation(m_edges_tx2D, m_area_tx2D, m_search_tx2D, m_blending_weights_tx2D);
            neighborhoodBlending(input_tx2D, m_blending_weights_tx2D, m_output_tx2D);

#ifdef PROFILING
            m_perf_manager->stop_timer(m_timers[0]);
#endif
        }
        // fxaa
        else if (mode == 1) {
            fxaa(input_tx2D, m_output_tx2D);
        }
        // no aa
        else if (mode == 2) {
            copyTextureViaShader(input_tx2D, m_output_tx2D);
        }

        ++m_version;
        m_settings_have_changed = false;
        this->m_smaa_detection_technique.ResetDirty();
    }

    
    if (lhs_tc->version() < m_version || this->m_smaa_view.IsDirty()) {
        int view = this->m_smaa_view.Param<core::param::EnumParam>()->Value();

        switch (view) {
        case 0:
            lhs_tc->setData(m_output_tx2D, m_version);
            break;
        case 1:
            lhs_tc->setData(m_edges_tx2D, m_version);
            break;
        case 2:
            lhs_tc->setData(m_blending_weights_tx2D, m_version);
            break;
        default:
            lhs_tc->setData(m_output_tx2D, m_version);
            break;
        }

        this->m_smaa_view.ResetDirty();
    }

    return true;
}

bool megamol::compositing::AntiAliasing::getMetaDataCallback(core::Call& caller) { return true; }
