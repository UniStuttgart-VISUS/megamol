/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "AntiAliasing.h"

#include <array>
#include <chrono>
#include <random>

#include "SMAA/SMAAAreaTex.h"
#include "SMAA/SMAASearchTex.h"

#include "compositing_gl/CompositingCalls.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#ifdef MEGAMOL_USE_PROFILING
#include "PerformanceManager.h"
#endif


/*
 * @megamol::compositing_gl::AntiAliasing::AntiAliasing
 */
megamol::compositing_gl::AntiAliasing::AntiAliasing()
        : mmstd_gl::ModuleGL()
        , version_(0)
        , tex_inspector_({"Edges", "BlendingWeights", "Output"})
        , output_tx2D_(nullptr)
        , mode_("Mode", "Sets antialiasing technqiue: SMAA, FXAA, no AA")
        , smaa_mode_("SMAA Mode", "Sets the SMAA mode: SMAA 1x or SMAA T2x")
        , smaa_quality_("QualityLevel", "Sets smaa quality level")
        , smaa_threshold_("Threshold", "Sets smaa threshold")
        , smaa_max_search_steps_("MaxSearchSteps", "Sets smaa max search steps")
        , smaa_max_search_steps_diag_("MaxDiagSearchSteps", "Sets smaa max diagonal search steps")
        , smaa_disable_diag_detection_("DisableDiagDetection",
              "Enables/Disables diagonal detection. If set to false, diagonal detection is enabled")
        , smaa_disable_corner_detection_("DisableCornerDetection",
              "Enables/Disables corner detection. If set to false, corner detection is enabled")
        , smaa_corner_rounding_("CornerRounding", "Sets smaa corner rounding parameter")
        , smaa_detection_technique_("EdgeDetection",
              "Sets smaa edge detection base: luma, color, or depth. Use depth only when a depth "
              "texture can be provided as it is mandatory to have one")
        , out_texture_format_slot_("OutTexFormat", "texture format of output texture")
        , output_tex_slot_("OutputTexture", "Gives access to the resulting output texture")
        , input_tex_slot_("InputTexture", "Connects the input texture")
        , depth_tex_slot_("DepthTexture", "Connects the depth texture")
        , settings_have_changed_(false) {
    auto aa_modes = new core::param::EnumParam(1);
    aa_modes->SetTypePair(0, "SMAA");
    aa_modes->SetTypePair(1, "FXAA");
    aa_modes->SetTypePair(2, "None");
    this->mode_.SetParameter(aa_modes);
    this->mode_.SetUpdateCallback(&megamol::compositing_gl::AntiAliasing::visibilityCallback);
    this->MakeSlotAvailable(&this->mode_);

    auto smaa_modes = new core::param::EnumParam(0);
    smaa_modes->SetTypePair(0, "SMAA 1x");
    this->smaa_mode_.SetParameter(smaa_modes);
    this->smaa_mode_.SetUpdateCallback(&megamol::compositing_gl::AntiAliasing::visibilityCallback);
    this->MakeSlotAvailable(&this->smaa_mode_);

    auto qualities = new core::param::EnumParam(2);
    qualities->SetTypePair(0, "Low");
    qualities->SetTypePair(1, "Medium");
    qualities->SetTypePair(2, "High");
    qualities->SetTypePair(3, "Ultra");
    qualities->SetTypePair(4, "Custom");
    this->smaa_quality_.SetParameter(qualities);
    this->smaa_quality_.SetUpdateCallback(&megamol::compositing_gl::AntiAliasing::setSettingsCallback);
    this->MakeSlotAvailable(&this->smaa_quality_);

    auto threshold = new megamol::core::param::FloatParam(0.1f, 0.f, 0.5f);
    threshold->SetGUIVisible(false);
    threshold->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->smaa_threshold_.SetParameter(threshold);
    this->smaa_threshold_.SetUpdateCallback(&megamol::compositing_gl::AntiAliasing::setCustomSettingsCallback);
    this->MakeSlotAvailable(&this->smaa_threshold_);

    auto max_search_step = new megamol::core::param::IntParam(16, 0, 112);
    max_search_step->SetGUIVisible(false);
    max_search_step->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->smaa_max_search_steps_.SetParameter(max_search_step);
    this->smaa_max_search_steps_.SetUpdateCallback(&megamol::compositing_gl::AntiAliasing::setCustomSettingsCallback);
    this->MakeSlotAvailable(&this->smaa_max_search_steps_);

    auto max_search_diag = new megamol::core::param::IntParam(8, 0, 20);
    max_search_diag->SetGUIVisible(false);
    max_search_diag->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->smaa_max_search_steps_diag_.SetParameter(max_search_diag);
    this->smaa_max_search_steps_diag_.SetUpdateCallback(
        &megamol::compositing_gl::AntiAliasing::setCustomSettingsCallback);
    this->MakeSlotAvailable(&this->smaa_max_search_steps_diag_);

    auto disable_diag = new megamol::core::param::BoolParam(false);
    disable_diag->SetGUIVisible(false);
    this->smaa_disable_diag_detection_.SetParameter(disable_diag);
    this->smaa_disable_diag_detection_.SetUpdateCallback(
        &megamol::compositing_gl::AntiAliasing::setCustomSettingsCallback);
    this->MakeSlotAvailable(&this->smaa_disable_diag_detection_);

    auto disable_corner = new megamol::core::param::BoolParam(false);
    disable_corner->SetGUIVisible(false);
    this->smaa_disable_corner_detection_.SetParameter(disable_corner);
    this->smaa_disable_corner_detection_.SetUpdateCallback(
        &megamol::compositing_gl::AntiAliasing::setCustomSettingsCallback);
    this->MakeSlotAvailable(&this->smaa_disable_corner_detection_);

    auto corner_rounding = new megamol::core::param::IntParam(25, 0, 100);
    corner_rounding->SetGUIVisible(false);
    corner_rounding->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Drag);
    this->smaa_corner_rounding_.SetParameter(corner_rounding);
    this->smaa_corner_rounding_.SetUpdateCallback(&megamol::compositing_gl::AntiAliasing::setCustomSettingsCallback);
    this->MakeSlotAvailable(&this->smaa_corner_rounding_);

    auto detection_technique = new megamol::core::param::EnumParam(0);
    detection_technique->SetTypePair(0, "Luma");
    detection_technique->SetTypePair(1, "Color");
    detection_technique->SetTypePair(2, "Depth");
    this->smaa_detection_technique_.SetParameter(detection_technique);
    this->MakeSlotAvailable(&this->smaa_detection_technique_);

    auto tex_inspector_slots = this->tex_inspector_.GetParameterSlots();
    for (auto& tex_slot : tex_inspector_slots) {
        this->MakeSlotAvailable(tex_slot);
    }

    auto out_tex_formats = new megamol::core::param::EnumParam(0);
    out_tex_formats->SetTypePair(0, "RGBA_32F");
    out_tex_formats->SetTypePair(1, "RGBA_16F");
    out_tex_formats->SetTypePair(2, "RGBA_8UI");

    this->out_texture_format_slot_.SetParameter(out_tex_formats);
    this->out_texture_format_slot_.SetUpdateCallback(&megamol::compositing_gl::AntiAliasing::setTextureFormatCallback);
    this->MakeSlotAvailable(&this->out_texture_format_slot_);

    this->output_tex_slot_.SetCallback(
        compositing_gl::CallTexture2D::ClassName(), "GetData", &AntiAliasing::getDataCallback);
    this->output_tex_slot_.SetCallback(
        compositing_gl::CallTexture2D::ClassName(), "GetMetaData", &AntiAliasing::getMetaDataCallback);
    this->MakeSlotAvailable(&this->output_tex_slot_);

    this->input_tex_slot_.SetCompatibleCall<compositing_gl::CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->input_tex_slot_);

    this->depth_tex_slot_.SetCompatibleCall<compositing_gl::CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->depth_tex_slot_);
}


/*
 * @megamol::compositing_gl::AntiAliasing::~AntiAliasing
 */
megamol::compositing_gl::AntiAliasing::~AntiAliasing() {
    this->Release();
}

/*
 * @megamol::compositing_gl::AntiAliasing::create
 */
bool megamol::compositing_gl::AntiAliasing::create() {
// profiling
#ifdef MEGAMOL_USE_PROFILING
    perf_manager_ = const_cast<frontend_resources::PerformanceManager*>(
        &frontend_resources.get<frontend_resources::PerformanceManager>());

    frontend_resources::PerformanceManager::basic_timer_config render_timer;
    render_timer.name = "render";
    render_timer.api = frontend_resources::PerformanceManager::query_api::OPENGL;
    timers_ = perf_manager_->add_timers(this, {render_timer});
#endif

    // create shader programs
    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());
    auto shader_options_flags = std::make_unique<msf::ShaderFactoryOptionsOpenGL>(shader_options);
    shader_options_flags->addDefinition("OUT32F");
    try {
        copy_prgm_ = core::utility::make_glowl_shader("copy_texture", shader_options, "compositing_gl/copy.comp.glsl");

        fxaa_prgm_ = core::utility::make_glowl_shader(
            "fxaa", *shader_options_flags, "compositing_gl/AntiAliasing/fxaa.comp.glsl");

        smaa_edge_detection_prgm_ = core::utility::make_glowl_shader(
            "smaa_edge_detection", shader_options, "compositing_gl/AntiAliasing/smaa_edge_detection.comp.glsl");

        smaa_blending_weight_calculation_prgm_ = core::utility::make_glowl_shader("smaa_blending_weight_calculation",
            shader_options, "compositing_gl/AntiAliasing/smaa_blending_weights_calculation.comp.glsl");

        smaa_neighborhood_blending_prgm_ = core::utility::make_glowl_shader("smaa_neighborhood_blending",
            *shader_options_flags, "compositing_gl/AntiAliasing/smaa_neighborhood_blending.comp.glsl");
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(("AntiAliasing: " + std::string(e.what())).c_str());
    }

    // init all textures
    glowl::TextureLayout tx_layout(out_tex_internal_format_, 1, 1, 1, out_tex_format_, out_tex_type_, 1);
    output_tx2D_ = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);

    // textures for smaa
    std::vector<std::pair<GLenum, GLint>> int_params = {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST},
        {GL_TEXTURE_MAG_FILTER, GL_LINEAR}};
    smaa_layout_ = glowl::TextureLayout(GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1, int_params, {});
    glowl::TextureLayout area_layout(
        GL_RG8, AREATEX_WIDTH, AREATEX_HEIGHT, 1, GL_RG, GL_UNSIGNED_BYTE, 1, int_params, {});
    glowl::TextureLayout search_layout(
        GL_R8, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, 1, GL_RED, GL_UNSIGNED_BYTE, 1, int_params, {});

    edges_tx2D_ = std::make_shared<glowl::Texture2D>("smaa_edges_tx2D", smaa_layout_, nullptr);
    blending_weights_tx2D_ = std::make_shared<glowl::Texture2D>("smaa_blend_tx2D", smaa_layout_, nullptr);


    // need to flip image around horizontal axis
    //area_.resize(AREATEX_SIZE);
    //for (size_t y = 0; y < AREATEX_HEIGHT; ++y) {
    //    for (size_t x = 0; x < AREATEX_WIDTH; ++x) {
    //        size_t id = x + y * AREATEX_WIDTH;

    //        size_t flip_id = x + (AREATEX_HEIGHT - 1 - y) * AREATEX_WIDTH;

    //        area_[2 * id + 0] = areaTexBytes[2 * flip_id + 0]; // R
    //        area_[2 * id + 1] = areaTexBytes[2 * flip_id + 1]; // G
    //    }
    //}

    //search_.resize(SEARCHTEX_SIZE);
    //for (size_t y = 0; y < SEARCHTEX_HEIGHT; ++y) {
    //    for (size_t x = 0; x < SEARCHTEX_WIDTH; ++x) {
    //        size_t id = x + y * SEARCHTEX_WIDTH;

    //        size_t flip_id = x + (SEARCHTEX_HEIGHT - 1 - y) * SEARCHTEX_WIDTH;

    //        search_[id + 0] = searchTexBytes[flip_id + 0]; // R
    //    }
    //}

    // TODO: flip y coordinate in texture accesses in shadercode and also flip textures here?
    area_tx2D_ = std::make_shared<glowl::Texture2D>("smaa_area_tx2D", area_layout, areaTexBytes);
    search_tx2D_ = std::make_shared<glowl::Texture2D>("smaa_search_tx2D", search_layout, searchTexBytes);

    ssbo_constants_ = std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    return true;
}


/*
 * @megamol::compositing_gl::AntiAliasing::release
 */
void megamol::compositing_gl::AntiAliasing::release() {
#ifdef MEGAMOL_USE_PROFILING
    perf_manager_->remove_timers(timers_);
#endif
}


/*
 * @megamol::compositing_gl::AntiAliasing::setSettingsCallback
 */
bool megamol::compositing_gl::AntiAliasing::setSettingsCallback(core::param::ParamSlot& slot) {
    // low
    if (slot.Param<core::param::EnumParam>()->Value() == 0) {
        smaa_constants_.Smaa_threshold = 0.15f;
        smaa_constants_.Smaa_depth_threshold = 0.1f * smaa_constants_.Smaa_threshold;
        smaa_constants_.Max_search_steps = 4;
        smaa_constants_.Max_search_steps_diag = 8;
        smaa_constants_.Disable_diag_detection = true;
        smaa_constants_.Disable_corner_detection = true;
        smaa_constants_.Corner_rounding = 25;
        smaa_constants_.Corner_rounding_norm = smaa_constants_.Corner_rounding / 100.f;
    }
    // medium
    else if (slot.Param<core::param::EnumParam>()->Value() == 1) {
        smaa_constants_.Smaa_threshold = 0.1f;
        smaa_constants_.Smaa_depth_threshold = 0.1f * smaa_constants_.Smaa_threshold;
        smaa_constants_.Max_search_steps = 8;
        smaa_constants_.Max_search_steps_diag = 8;
        smaa_constants_.Disable_diag_detection = true;
        smaa_constants_.Disable_corner_detection = true;
        smaa_constants_.Corner_rounding = 25;
        smaa_constants_.Corner_rounding_norm = smaa_constants_.Corner_rounding / 100.f;
    }
    // high
    else if (slot.Param<core::param::EnumParam>()->Value() == 2) {
        smaa_constants_.Smaa_threshold = 0.1f;
        smaa_constants_.Smaa_depth_threshold = 0.1f * smaa_constants_.Smaa_threshold;
        smaa_constants_.Max_search_steps = 16;
        smaa_constants_.Max_search_steps_diag = 8;
        smaa_constants_.Disable_diag_detection = false;
        smaa_constants_.Disable_corner_detection = false;
        smaa_constants_.Corner_rounding = 25;
        smaa_constants_.Corner_rounding_norm = smaa_constants_.Corner_rounding / 100.f;
    }
    // ultra
    else if (slot.Param<core::param::EnumParam>()->Value() == 3) {
        smaa_constants_.Smaa_threshold = 0.05f;
        smaa_constants_.Smaa_depth_threshold = 0.1f * smaa_constants_.Smaa_threshold;
        smaa_constants_.Max_search_steps = 32;
        smaa_constants_.Max_search_steps_diag = 16;
        smaa_constants_.Disable_diag_detection = false;
        smaa_constants_.Disable_corner_detection = false;
        smaa_constants_.Corner_rounding = 25;
        smaa_constants_.Corner_rounding_norm = smaa_constants_.Corner_rounding / 100.f;
    }


    // custom
    if (slot.Param<core::param::EnumParam>()->Value() == 4) {
        this->smaa_threshold_.Param<core::param::FloatParam>()->SetValue(constants_.Smaa_threshold);
        this->smaa_max_search_steps_.Param<core::param::IntParam>()->SetValue(constants_.Max_search_steps);
        this->smaa_max_search_steps_diag_.Param<core::param::IntParam>()->SetValue(constants_.Max_search_steps_diag);
        this->smaa_disable_diag_detection_.Param<core::param::BoolParam>()->SetValue(constants_.Disable_diag_detection);
        this->smaa_disable_corner_detection_.Param<core::param::BoolParam>()->SetValue(
            constants_.Disable_corner_detection);
        this->smaa_corner_rounding_.Param<core::param::IntParam>()->SetValue(constants_.Corner_rounding);

        smaa_constants_ = constants_;

        this->smaa_threshold_.Param<core::param::FloatParam>()->SetGUIVisible(true);
        this->smaa_max_search_steps_.Param<core::param::IntParam>()->SetGUIVisible(true);
        this->smaa_max_search_steps_diag_.Param<core::param::IntParam>()->SetGUIVisible(true);
        this->smaa_disable_diag_detection_.Param<core::param::BoolParam>()->SetGUIVisible(true);
        this->smaa_disable_corner_detection_.Param<core::param::BoolParam>()->SetGUIVisible(true);
        this->smaa_corner_rounding_.Param<core::param::IntParam>()->SetGUIVisible(true);
    } else {
        this->smaa_threshold_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        this->smaa_max_search_steps_.Param<core::param::IntParam>()->SetGUIVisible(false);
        this->smaa_max_search_steps_diag_.Param<core::param::IntParam>()->SetGUIVisible(false);
        this->smaa_disable_diag_detection_.Param<core::param::BoolParam>()->SetGUIVisible(false);
        this->smaa_disable_corner_detection_.Param<core::param::BoolParam>()->SetGUIVisible(false);
        this->smaa_corner_rounding_.Param<core::param::IntParam>()->SetGUIVisible(false);
    }

    settings_have_changed_ = true;

    return true;
}


/*
 * @megamol::compositing_gl::AntiAliasing::setCustomSettingsCallback
 */
bool megamol::compositing_gl::AntiAliasing::setCustomSettingsCallback(core::param::ParamSlot& slot) {
    smaa_constants_.Smaa_threshold = this->smaa_threshold_.Param<core::param::FloatParam>()->Value();
    smaa_constants_.Smaa_depth_threshold = 0.1f * smaa_constants_.Smaa_threshold;
    smaa_constants_.Max_search_steps = this->smaa_max_search_steps_.Param<core::param::IntParam>()->Value();
    smaa_constants_.Max_search_steps_diag = this->smaa_max_search_steps_diag_.Param<core::param::IntParam>()->Value();
    smaa_constants_.Disable_diag_detection =
        this->smaa_disable_diag_detection_.Param<core::param::BoolParam>()->Value();
    smaa_constants_.Disable_corner_detection =
        this->smaa_disable_corner_detection_.Param<core::param::BoolParam>()->Value();
    smaa_constants_.Corner_rounding = this->smaa_corner_rounding_.Param<core::param::IntParam>()->Value();
    smaa_constants_.Corner_rounding_norm = smaa_constants_.Corner_rounding / 100.f;

    // keep a backup from the custom settings, so if custom is selected again
    // the previous values are loaded
    constants_ = smaa_constants_;

    settings_have_changed_ = true;

    return true;
}


/*
 * @megamol::compositing_gl::AntiAliasing::visibilityCallback
 */
bool megamol::compositing_gl::AntiAliasing::visibilityCallback(core::param::ParamSlot& slot) {
    // smaa enabled
    if (this->mode_.Param<core::param::EnumParam>()->Value() == 0) {
        smaa_quality_.Param<core::param::EnumParam>()->SetGUIVisible(true);
        smaa_detection_technique_.Param<core::param::EnumParam>()->SetGUIVisible(true);
        smaa_mode_.Param<core::param::EnumParam>()->SetGUIVisible(true);
    }
    // smaa disabled
    else {
        smaa_quality_.Param<core::param::EnumParam>()->SetGUIVisible(false);
        smaa_detection_technique_.Param<core::param::EnumParam>()->SetGUIVisible(false);
        smaa_mode_.Param<core::param::EnumParam>()->SetGUIVisible(false);
    }

    settings_have_changed_ = true;

    return true;
}


/*
 * @megamol::compositing_gl::AntiAliasing::edgeDetection
 */
void megamol::compositing_gl::AntiAliasing::edgeDetection(const std::shared_ptr<glowl::Texture2D>& input,
    const std::shared_ptr<glowl::Texture2D>& depth, const std::shared_ptr<glowl::Texture2D>& edges,
    GLint detection_technique) {
    smaa_edge_detection_prgm_->use();

    glActiveTexture(GL_TEXTURE0);
    input->bindTexture();
    glUniform1i(smaa_edge_detection_prgm_->getUniformLocation("g_colorTex"), 0);

    // find edges based on the depth
    if (detection_technique == 2) {
        if (depth == nullptr) {
            core::utility::log::Log::DefaultLog.WriteError("AntiAliasing::edgeDetection: depth texture is nullptr");
        } else {
            glActiveTexture(GL_TEXTURE1);
            depth->bindTexture();
            glUniform1i(smaa_edge_detection_prgm_->getUniformLocation("g_depthTex"), 0);
        }
    }

    glUniform1i(smaa_edge_detection_prgm_->getUniformLocation("technique"), detection_technique);

    ssbo_constants_->bind(0);

    edges->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(static_cast<int>(std::ceil(edges->getWidth() / 8.0f)),
        static_cast<int>(std::ceil(edges->getHeight() / 8.0f)), 1);
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    resetGLStates();
}


/*
 * @megamol::compositing_gl::AntiAliasing::blendingWeightCalculation
 */
void megamol::compositing_gl::AntiAliasing::blendingWeightCalculation(const std::shared_ptr<glowl::Texture2D>& edges,
    const std::shared_ptr<glowl::Texture2D>& area, const std::shared_ptr<glowl::Texture2D>& search,
    const std::shared_ptr<glowl::Texture2D>& weights) {
    smaa_blending_weight_calculation_prgm_->use();

    glActiveTexture(GL_TEXTURE0);
    edges->bindTexture();
    glUniform1i(smaa_blending_weight_calculation_prgm_->getUniformLocation("g_edgesTex"), 0);
    glActiveTexture(GL_TEXTURE1);
    area->bindTexture();
    glUniform1i(smaa_blending_weight_calculation_prgm_->getUniformLocation("g_areaTex"), 1);
    glActiveTexture(GL_TEXTURE2);
    search->bindTexture();
    glUniform1i(smaa_blending_weight_calculation_prgm_->getUniformLocation("g_searchTex"), 2);

    ssbo_constants_->bind(0);

    weights->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(static_cast<int>(std::ceil(weights->getWidth() / 8.0f)),
        static_cast<int>(std::ceil(weights->getHeight() / 8.0f)), 1);
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    resetGLStates();
}


/*
 * @megamol::compositing_gl::AntiAliasing::neighborhoodBlending
 */
void megamol::compositing_gl::AntiAliasing::neighborhoodBlending(const std::shared_ptr<glowl::Texture2D>& input,
    const std::shared_ptr<glowl::Texture2D>& weights, const std::shared_ptr<glowl::Texture2D>& result) {
    smaa_neighborhood_blending_prgm_->use();

    glActiveTexture(GL_TEXTURE0);
    input->bindTexture();
    glUniform1i(smaa_neighborhood_blending_prgm_->getUniformLocation("g_colorTex"), 0);
    glActiveTexture(GL_TEXTURE1);
    weights->bindTexture();
    glUniform1i(smaa_neighborhood_blending_prgm_->getUniformLocation("g_blendingWeightsTex"), 1);

    ssbo_constants_->bind(0);

    result->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(static_cast<int>(std::ceil(result->getWidth() / 8.0f)),
        static_cast<int>(std::ceil(result->getHeight() / 8.0f)), 1);
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    resetGLStates();
}


/*
 * @megamol::compositing_gl::AntiAliasing::fxaa
 */
void megamol::compositing_gl::AntiAliasing::fxaa(
    const std::shared_ptr<glowl::Texture2D>& input, const std::shared_ptr<glowl::Texture2D>& output) {
    fxaa_prgm_->use();

    glActiveTexture(GL_TEXTURE0);
    input->bindTexture();
    glUniform1i(smaa_neighborhood_blending_prgm_->getUniformLocation("src_tx2D"), 0);

    output->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(static_cast<int>(std::ceil(output->getWidth() / 8.0f)),
        static_cast<int>(std::ceil(output->getHeight() / 8.0f)), 1);
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // reset gl states
    glUseProgram(0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
}


/*
 * @megamol::compositing_gl::AntiAliasing::copyTextureViaShader
 */
void megamol::compositing_gl::AntiAliasing::copyTextureViaShader(
    const std::shared_ptr<glowl::Texture2D>& src, const std::shared_ptr<glowl::Texture2D>& tgt) {
    copy_prgm_->use();

    glActiveTexture(GL_TEXTURE0);
    src->bindTexture();
    glUniform1i(copy_prgm_->getUniformLocation("src_tx2D"), 0);

    tgt->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(tgt->getWidth() / 8.0f)), static_cast<int>(std::ceil(tgt->getHeight() / 8.0f)), 1);
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}


/*
 * @megamol::compositing_gl::AntiAliasing::getDataCallback
 */
bool megamol::compositing_gl::AntiAliasing::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<compositing_gl::CallTexture2D*>(&caller);
    auto rhs_call_input = input_tex_slot_.CallAs<compositing_gl::CallTexture2D>();
    auto rhs_call_depth = depth_tex_slot_.CallAs<compositing_gl::CallTexture2D>();

    if (lhs_tc == NULL)
        return false;

    if (rhs_call_input != NULL) {
        if (!(*rhs_call_input)(0))
            return false;
    }

    GLint technique = smaa_detection_technique_.Param<core::param::EnumParam>()->Value();
    // depth based edge detection requires the depth texture
    if (technique == 2) {
        if (rhs_call_depth != NULL) {
            if (!(*rhs_call_depth)(0)) {
                return false;
            }
        } else {
            core::utility::log::Log::DefaultLog.WriteError(
                "Depth based edge detection for SMAA requires the depth texture of the"
                " rendered mesh. Check the depth texture slot.");
            return false;
        }
    }

    bool something_has_changed = (rhs_call_input != NULL ? rhs_call_input->hasUpdate() : false) ||
                                 settings_have_changed_ || this->smaa_detection_technique_.IsDirty() ||
                                 (technique == 2 ? rhs_call_depth->hasUpdate() : false);

    if (something_has_changed) {
#ifdef MEGAMOL_USE_PROFILING
        perf_manager_->start_timer(timers_[0]);
#endif

        // get input
        auto input_tx2D = rhs_call_input->getData();
        int input_width = input_tx2D->getWidth();
        int input_height = input_tx2D->getHeight();

        // init output texture if necessary
        if (output_tx2D_->getWidth() != input_width || output_tx2D_->getHeight() != input_height) {
            glowl::TextureLayout tx_layout(
                out_tex_internal_format_, input_width, input_height, 1, out_tex_format_, out_tex_type_, 1);

            output_tx2D_->reload(tx_layout, nullptr);
        }

        // get aliasing mode: smaa, fxaa, or none
        int mode = this->mode_.Param<core::param::EnumParam>()->Value();

        // smaa
        if (mode == 0) {
            // textures the smaa passes need
            std::shared_ptr<glowl::Texture2D> depth_tx2D;

            if (technique == 2) {
                depth_tx2D = rhs_call_depth->getData();
            }

            // init edge and blending weight textures
            if ((input_width != smaa_layout_.width) || (input_height != smaa_layout_.height)) {
                smaa_layout_.width = input_width;
                smaa_layout_.height = input_height;
                edges_tx2D_->reload(smaa_layout_, nullptr);
                blending_weights_tx2D_->reload(smaa_layout_, nullptr);
            }

            // always clear them to guarantee correct textures
            GLubyte col[4] = {0, 0, 0, 0};
            edges_tx2D_->clearTexImage(col);
            blending_weights_tx2D_->clearTexImage(col);

            if (smaa_constants_.Rt_metrics[2] != input_width || smaa_constants_.Rt_metrics[3] != input_height ||
                settings_have_changed_) {
                smaa_constants_.Rt_metrics = glm::vec4(
                    1.f / (float)input_width, 1.f / (float)input_height, (float)input_width, (float)input_height);
                ssbo_constants_->rebuffer(&smaa_constants_, sizeof(smaa_constants_));
            }


            // perform smaa!
            edgeDetection(input_tx2D, depth_tx2D, edges_tx2D_, technique);
            blendingWeightCalculation(edges_tx2D_, area_tx2D_, search_tx2D_, blending_weights_tx2D_);
            neighborhoodBlending(input_tx2D, blending_weights_tx2D_, output_tx2D_);

        }
        // fxaa
        else if (mode == 1) {
            fxaa(input_tx2D, output_tx2D_);
        }
        // no aa
        else if (mode == 2) {
            copyTextureViaShader(input_tx2D, output_tx2D_);
        }

#ifdef MEGAMOL_USE_PROFILING
        perf_manager_->stop_timer(timers_[0]);
#endif

        ++version_;
        settings_have_changed_ = false;
        this->smaa_detection_technique_.ResetDirty();
    }

    if (tex_inspector_.GetShowInspectorSlotValue()) {
        glm::vec2 tex_dim = glm::vec2(smaa_layout_.width, smaa_layout_.height);

        GLuint tex_to_show = 0;
        switch (tex_inspector_.GetSelectTextureSlotValue()) {
        case 0:
            tex_to_show = edges_tx2D_->getName();
            break;
        case 1:
            tex_to_show = blending_weights_tx2D_->getName();
            break;
        case 2:
            tex_to_show = output_tx2D_->getName();
            break;
        default:
            tex_to_show = output_tx2D_->getName();
            break;
        }

        tex_inspector_.SetTexture((void*)(intptr_t)tex_to_show, tex_dim.x, tex_dim.y);
        tex_inspector_.ShowWindow();
    }


    if (lhs_tc->version() < version_) {
        lhs_tc->setData(output_tx2D_, version_);
    }

    return true;
}


bool megamol::compositing_gl::AntiAliasing::setTextureFormatCallback(core::param::ParamSlot& slot) {
    switch (this->out_texture_format_slot_.Param<core::param::EnumParam>()->Value()) {
    case 0: //RGBA32F
        out_tex_internal_format_ = GL_RGBA32F;
        out_tex_format_ = GL_RGB;
        out_tex_type_ = GL_FLOAT;
        break;
    case 1: //RGBA16F
        out_tex_internal_format_ = GL_RGBA16F;
        out_tex_format_ = GL_RGBA;
        out_tex_type_ = GL_HALF_FLOAT;
        break;
    case 2: //RGBA8UI
        out_tex_internal_format_ = GL_RGBA8_SNORM;
        out_tex_format_ = GL_RGBA;
        out_tex_type_ = GL_UNSIGNED_BYTE;
        break;
    }
    // reinit all textures
    glowl::TextureLayout tx_layout(out_tex_internal_format_, 1, 1, 1, out_tex_format_, out_tex_type_, 1);
    output_tx2D_ = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    checkFormatsAndRecompile();
    return true;
}

/*
 * @megamol::compositing_gl::AntiAliasing::getMetaDataCallback
 */
bool megamol::compositing_gl::AntiAliasing::getMetaDataCallback(core::Call& caller) {
    return true;
}

void megamol::compositing_gl::AntiAliasing::checkFormatsAndRecompile() {
    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());
    auto shader_options_flags = std::make_unique<msf::ShaderFactoryOptionsOpenGL>(shader_options);
    if (this->out_texture_format_slot_.Param<core::param::EnumParam>()->Value() == 0) {
        shader_options_flags->addDefinition("OUT32F");
    } else if (this->out_texture_format_slot_.Param<core::param::EnumParam>()->Value() == 1) {
        shader_options_flags->addDefinition("OUT16HF");
    } else if (this->out_texture_format_slot_.Param<core::param::EnumParam>()->Value() == 2) {
        shader_options_flags->addDefinition("OUT8NB");
    }
    
    try {

        fxaa_prgm_ = core::utility::make_glowl_shader(
            "fxaa", *shader_options_flags, "compositing_gl/AntiAliasing/fxaa.comp.glsl");
        smaa_neighborhood_blending_prgm_ = core::utility::make_glowl_shader("smaa_neighborhood_blending",
            shader_options, "compositing_gl/AntiAliasing/smaa_neighborhood_blending.comp.glsl");
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(("AntiAliasing: " + std::string(e.what())).c_str());
    }
}
