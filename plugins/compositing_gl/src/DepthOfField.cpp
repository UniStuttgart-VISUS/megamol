/**
 * MegaMol
 * Copyright (c) 2024, MegaMol Dev Team
 * All rights reserved.
 */

#include "DepthOfField.h"

#include <array>
#include <chrono>
#include <random>

#include "compositing_gl/CompositingCalls.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#ifdef MEGAMOL_USE_PROFILING
#include "PerformanceManager.h"
#endif


/*
 * @megamol::compositing_gl::DepthOfField::DepthOfField
 */
megamol::compositing_gl::DepthOfField::DepthOfField()
        : mmstd_gl::ModuleGL()
        , version_(0)
        , res_(0)
        , ps_strength_("Strength", "Used to determine the kernel scale and blend values.")
        , ps_focal_distance_("FocalDistance", "Used to determine ne, nb, fb, fe values for coc generation.")
        , ps_focal_range_("FocalRange",
              "Range between in-focus and out-focus. Used to determine ne, nb, fb, fe values for coc generation.")
        , tex_inspector_({"Color/Input", "Depth", "CoC", "Color_4", "Color_mul_coc_far_4", "CoC_4", "CoC_near_blurred_4",
              "Near_field_4", "Far_field_4", "Near_field_filled_4", "Far_field_filled_4", "Output"})
        , output_tex_slot_("OutputTexture", "Gives access to the resulting output texture")
        , input_tex_slot_("InputTexture", "Connects the input texture")
        , depth_tex_slot_("DepthTexture", "Connects the depth texture")
        , camera_slot_("Camera", "Connects the camera")
        , settings_have_changed_(false)
        , out_format_handler_("OUTFORMAT",
              {
                  GL_RGBA8_SNORM,
                  GL_RGBA16F,
                  GL_RGBA32F,
              },
              std::function<bool()>(std::bind(&DepthOfField::textureFormatUpdate, this))) {

    ps_strength_ << new core::param::FloatParam(1.f);
    ps_strength_.SetUpdateCallback(&megamol::compositing_gl::DepthOfField::setSettingsCallback);
    MakeSlotAvailable(&ps_strength_);

    ps_focal_distance_ << new core::param::FloatParam(40.f);
    ps_focal_distance_.SetUpdateCallback(&megamol::compositing_gl::DepthOfField::setSettingsCallback);
    MakeSlotAvailable(&ps_focal_distance_);

    ps_focal_range_ << new core::param::FloatParam(20.f);
    ps_focal_range_.SetUpdateCallback(&megamol::compositing_gl::DepthOfField::setSettingsCallback);
    MakeSlotAvailable(&ps_focal_range_);

    auto tex_inspector_slots = tex_inspector_.GetParameterSlots();
    for (auto& tex_slot : tex_inspector_slots) {
        MakeSlotAvailable(tex_slot);
    }

    MakeSlotAvailable(out_format_handler_.getFormatSelectorSlot());

    output_tex_slot_.SetCallback(
        compositing_gl::CallTexture2D::ClassName(), "GetData", &DepthOfField::getDataCallback);
    output_tex_slot_.SetCallback(
        compositing_gl::CallTexture2D::ClassName(), "GetMetaData", &DepthOfField::getMetaDataCallback);
    MakeSlotAvailable(&output_tex_slot_);

    input_tex_slot_.SetCompatibleCall<compositing_gl::CallTexture2DDescription>();
    MakeSlotAvailable(&input_tex_slot_);

    depth_tex_slot_.SetCompatibleCall<compositing_gl::CallTexture2DDescription>();
    MakeSlotAvailable(&depth_tex_slot_);

    camera_slot_.SetCompatibleCall<CallCameraDescription>();
    MakeSlotAvailable(&camera_slot_);
}


/*
 * @megamol::compositing_gl::DepthOfField::~DepthOfField
 */
megamol::compositing_gl::DepthOfField::~DepthOfField() {
    Release();
}

/*
 * @megamol::compositing_gl::DepthOfField::create
 */
bool megamol::compositing_gl::DepthOfField::create() {
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
    auto shader_options_flags = out_format_handler_.addDefinitions(shader_options);

    try {
        coc_generation_prgm_ = core::utility::make_glowl_shader("coc_generation/*",
            *shader_options_flags, "compositing_gl/DepthOfField/coc_generation.comp.glsl");

        downsample_prgm_ = core::utility::make_glowl_shader(
            "downsample/*", *shader_options_flags, "compositing_gl/DepthOfField/downsample.comp.glsl");

        // TODO: removeDefinition() for shader_options
        auto shader_options_max_filter_horizontal = shader_options;
        shader_options_max_filter_horizontal.addDefinition("MAX_FILTER_HORIZONTAL");
        coc_near_blur_prgm_[0] = core::utility::make_glowl_shader("coc_near_blur[0]/*",
            shader_options_max_filter_horizontal, "compositing_gl/DepthOfField/near_coc_blur.comp.glsl");

        auto shader_options_max_filter_vertical = shader_options;
        shader_options_max_filter_vertical.addDefinition("MAX_FILTER_VERTICAL");
        coc_near_blur_prgm_[1] = core::utility::make_glowl_shader("coc_near_blur[1]/*",
            shader_options_max_filter_vertical, "compositing_gl/DepthOfField/near_coc_blur.comp.glsl");

        auto shader_options_blur_filter_horizontal = shader_options;
        shader_options_blur_filter_horizontal.addDefinition("BLUR_FILTER_HORIZONTAL");
        coc_near_blur_prgm_[2] = core::utility::make_glowl_shader("coc_near_blur[2]/*",
            shader_options_blur_filter_horizontal, "compositing_gl/DepthOfField/near_coc_blur.comp.glsl");

        auto shader_options_blur_filter_vertical = shader_options;
        shader_options_blur_filter_vertical.addDefinition("BLUR_FILTER_VERTICAL");
        coc_near_blur_prgm_[3] = core::utility::make_glowl_shader("coc_near_blur[3]/*",
            shader_options_blur_filter_vertical, "compositing_gl/DepthOfField/near_coc_blur.comp.glsl");

        computation_prgm_ = core::utility::make_glowl_shader(
            "computation/*", *shader_options_flags, "compositing_gl/DepthOfField/computation.comp.glsl");

        fill_prgm_ = core::utility::make_glowl_shader(
            "fill/*", *shader_options_flags, "compositing_gl/DepthOfField/fill.comp.glsl");

        composite_prgm_ = core::utility::make_glowl_shader(
            "composite/*", *shader_options_flags, "compositing_gl/DepthOfField/composite.comp.glsl");
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(("DepthOfField: " + std::string(e.what())).c_str());
    }

    // layouts
    // TODO: use out_format_handler_ for output
    glowl::TextureLayout tx_layout_base = glowl::TextureLayout(GL_RGBA32F, 1, 1, 1, GL_RGBA, GL_FLOAT, 1);
    glowl::TextureLayout tx_layout_r11f_g11f_b10f = glowl::TextureLayout(GL_R11F_G11F_B10F, 1, 1, 1, GL_RGB, GL_FLOAT, 1);
    glowl::TextureLayout tx_layout_r8_g8_unorm = glowl::TextureLayout(GL_RG8, 1, 1, 1, GL_RG, GL_FLOAT, 1);
    glowl::TextureLayout tx_layout_r8_unorm = glowl::TextureLayout(GL_R8, 1, 1, 1, GL_RED, GL_FLOAT, 1);

    // textures
    //color_tx2D_               = std::make_shared<glowl::Texture2D>("color", tx_layout_base, nullptr);
    //depth_tx2D_               = std::make_shared<glowl::Texture2D>("depth", depth_layout_lol, nullptr);
    coc_tx2D_                   = std::make_shared<glowl::Texture2D>("coc", tx_layout_r8_g8_unorm, nullptr);
    color_4_tx2D_               = std::make_shared<glowl::Texture2D>("color_4", tx_layout_r11f_g11f_b10f, nullptr);
    color_mul_coc_far_4_tx2D_   = std::make_shared<glowl::Texture2D>("color_mul_coc_far_4", tx_layout_r11f_g11f_b10f, nullptr);
    coc_4_tx2D_                 = std::make_shared<glowl::Texture2D>("coc_4", tx_layout_r8_g8_unorm, nullptr);
    coc_near_blurred_4_tx2D_[0] = std::make_shared<glowl::Texture2D>("coc_near_blurred_max_x_4", tx_layout_r8_unorm, nullptr);
    coc_near_blurred_4_tx2D_[1] = std::make_shared<glowl::Texture2D>("coc_near_blurred_max_4", tx_layout_r8_unorm, nullptr);
    coc_near_blurred_4_tx2D_[2] = std::make_shared<glowl::Texture2D>("coc_near_blurred_blur_x_4", tx_layout_r8_unorm, nullptr);
    coc_near_blurred_4_tx2D_[3] = std::make_shared<glowl::Texture2D>("coc_near_blurred_4", tx_layout_r8_unorm, nullptr);
    near_field_4_tx2D_          = std::make_shared<glowl::Texture2D>("near_field_4", tx_layout_r11f_g11f_b10f, nullptr);
    far_field_4_tx2D_           = std::make_shared<glowl::Texture2D>("far_field_4", tx_layout_r11f_g11f_b10f, nullptr);
    near_field_filled_4_tx2D_   = std::make_shared<glowl::Texture2D>("near_field_filled_4", tx_layout_r11f_g11f_b10f, nullptr);
    far_field_filled_4_tx2D_    = std::make_shared<glowl::Texture2D>("far_field_filled_4", tx_layout_r11f_g11f_b10f, nullptr);
    output_tx2D_                = std::make_shared<glowl::Texture2D>("dof_output", tx_layout_base, nullptr);

    // point sampler
    std::vector<std::pair<GLenum, GLint>> int_params = {
        {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_MIN_FILTER, GL_NEAREST},
        {GL_TEXTURE_MAG_FILTER, GL_NEAREST}
    };
    point_sampler_ = std::make_shared<glowl::Sampler>("point_sampler", glowl::SamplerLayout(int_params));

    // linear sampler
    int_params.clear();
    int_params = {
        {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR},
        {GL_TEXTURE_MAG_FILTER, GL_LINEAR}
    };
    bilinear_sampler_ = std::make_shared<glowl::Sampler>("bilinear_sampler", glowl::SamplerLayout(int_params));

    return true;
}


/*
 * @megamol::compositing_gl::DepthOfField::release
 */
void megamol::compositing_gl::DepthOfField::release() {
#ifdef MEGAMOL_USE_PROFILING
    perf_manager_->remove_timers(timers_);
#endif
}


/*
 * @megamol::compositing_gl::AntiAliasing::setSettingsCallback
 */
bool megamol::compositing_gl::DepthOfField::setSettingsCallback(core::param::ParamSlot& slot) {
    settings_have_changed_ = true;
    return true;
}


// TODO: make output texture also as function parameter?
/*
 * @megamol::compositing_gl::DepthOfField::cocGeneration
 */
void megamol::compositing_gl::DepthOfField::cocGeneration(
    const std::shared_ptr<glowl::Texture2D>& depth,
    const glm::vec2& proj_params,
    const glm::vec4& fields
) {
    coc_generation_prgm_->use();

    glActiveTexture(GL_TEXTURE0);
    bindTextureWithPointSampler(depth, 0);
    glUniform1i(coc_generation_prgm_->getUniformLocation("depth_point_tx2D"), 0);

    glUniform2fv(coc_generation_prgm_->getUniformLocation("proj_params"), 1, glm::value_ptr(proj_params));
    glUniform4fv(coc_generation_prgm_->getUniformLocation("fields"), 1, glm::value_ptr(fields));

    coc_tx2D_->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(getWidth() / 8.0f)),
        static_cast<int>(std::ceil(getHeight() / 8.0f)),
        1
    );
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    resetGLStates(1);
}


/*
 * @megamol::compositing_gl::DepthOfField::downsample
 */
void megamol::compositing_gl::DepthOfField::downsample(
    const std::shared_ptr<glowl::Texture2D>& color,
    const std::shared_ptr<glowl::Texture2D>& coc
) {
    downsample_prgm_->use();

    glActiveTexture(GL_TEXTURE0);
    bindTextureWithPointSampler(color, 0);
    glUniform1i(downsample_prgm_->getUniformLocation("color_point_tx2D"), 0);

    glActiveTexture(GL_TEXTURE1);
    bindTextureWithBilinearSampler(color, 1);
    glUniform1i(downsample_prgm_->getUniformLocation("color_linear_tx2D"), 1);

    glActiveTexture(GL_TEXTURE2);
    bindTextureWithPointSampler(coc, 2);
    glUniform1i(downsample_prgm_->getUniformLocation("coc_point_tx2D"), 2);

    color_4_tx2D_->bindImage(0, GL_WRITE_ONLY);
    color_mul_coc_far_4_tx2D_->bindImage(1, GL_WRITE_ONLY);
    coc_4_tx2D_->bindImage(2, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(getHalfWidth() / 8.0f)),
        static_cast<int>(std::ceil(getHalfHeight() / 8.0f)),
        1
    );
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    resetGLStates(3);
}


/*
 * @megamol::compositing_gl::DepthOfField::nearCoCBlur
 */
void megamol::compositing_gl::DepthOfField::nearCoCBlur(
    const std::shared_ptr<glowl::Texture2D>& coc_4
) {
    // FIRST PASS - HORIZONTAL MAX
    coc_near_blur_prgm_[0]->use();

    glActiveTexture(GL_TEXTURE0);
    bindTextureWithPointSampler(coc_4, 0);
    glUniform1i(coc_near_blur_prgm_[0]->getUniformLocation("coc_4_point_tx2D"), 0);

    coc_near_blurred_4_tx2D_[0]->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(getHalfWidth() / 8.0f)),
        static_cast<int>(std::ceil(getHalfHeight() / 8.0f)),
        1
    );
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);


    // SECOND PASS - VERTICAL MAX
    coc_near_blur_prgm_[1]->use();

    glActiveTexture(GL_TEXTURE0);
    bindTextureWithPointSampler(coc_near_blurred_4_tx2D_[0], 0);
    glUniform1i(coc_near_blur_prgm_[1]->getUniformLocation("coc_4_point_tx2D"), 0);

    coc_near_blurred_4_tx2D_[1]->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(getHalfWidth() / 8.0f)),
        static_cast<int>(std::ceil(getHalfHeight() / 8.0f)),
        1
    );
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);


    // THIRD PASS - HORIZONTAL BLUR
    coc_near_blur_prgm_[2]->use();

    glActiveTexture(GL_TEXTURE0);
    bindTextureWithPointSampler(coc_near_blurred_4_tx2D_[1], 0);
    glUniform1i(coc_near_blur_prgm_[2]->getUniformLocation("coc_4_point_tx2D"), 0);

    coc_near_blurred_4_tx2D_[2]->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(getHalfWidth() / 8.0f)),
        static_cast<int>(std::ceil(getHalfHeight() / 8.0f)),
        1
    );
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);


    // FINAL PASS - VERTICAL BLUR
    coc_near_blur_prgm_[3]->use();

    glActiveTexture(GL_TEXTURE0);
    bindTextureWithPointSampler(coc_near_blurred_4_tx2D_[2], 0);
    glUniform1i(coc_near_blur_prgm_[3]->getUniformLocation("coc_4_point_tx2D"), 0);

    coc_near_blurred_4_tx2D_[3]->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(getHalfWidth() / 8.0f)),
        static_cast<int>(std::ceil(getHalfHeight() / 8.0f)),
        1
    );
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    resetGLStates(1);
}


/*
 * @megamol::compositing_gl::DepthOfField::computation
 */
void megamol::compositing_gl::DepthOfField::computation(
    const std::shared_ptr<glowl::Texture2D>& color_4,
    const std::shared_ptr<glowl::Texture2D>& color_mul_coc_far_4,
    const std::shared_ptr<glowl::Texture2D>& coc_4,
    const std::shared_ptr<glowl::Texture2D>& coc_near_blurred_4,
    float kernel_scale
) {
    computation_prgm_->use();

    glActiveTexture(GL_TEXTURE0);
    bindTextureWithPointSampler(color_4, 0);
    glUniform1i(computation_prgm_->getUniformLocation("color_4_point_tx2D"), 0);
    glActiveTexture(GL_TEXTURE1);
    bindTextureWithBilinearSampler(color_4, 1);
    glUniform1i(computation_prgm_->getUniformLocation("color_4_linear_tx2D"), 1);

    glActiveTexture(GL_TEXTURE2);
    bindTextureWithPointSampler(color_mul_coc_far_4, 2);
    glUniform1i(computation_prgm_->getUniformLocation("color_mul_coc_far_4_point_tx2D"), 2);
    glActiveTexture(GL_TEXTURE3);
    bindTextureWithBilinearSampler(color_mul_coc_far_4, 3);
    glUniform1i(computation_prgm_->getUniformLocation("color_mul_coc_far_4_linear_tx2D"), 3);

    glActiveTexture(GL_TEXTURE4);
    bindTextureWithPointSampler(coc_4, 4);
    glUniform1i(computation_prgm_->getUniformLocation("coc_4_point_tx2D"), 4);
    glActiveTexture(GL_TEXTURE5);
    bindTextureWithBilinearSampler(coc_4, 5);
    glUniform1i(computation_prgm_->getUniformLocation("coc_4_linear_tx2D"), 5);

    glActiveTexture(GL_TEXTURE6);
    bindTextureWithPointSampler(coc_near_blurred_4, 6);
    glUniform1i(computation_prgm_->getUniformLocation("coc_near_blurred_4_point_tx2D"), 6);

    glUniform1f(computation_prgm_->getUniformLocation("kernel_scale"), kernel_scale);

    near_field_4_tx2D_->bindImage(0, GL_WRITE_ONLY);
    far_field_4_tx2D_->bindImage(1, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(getHalfWidth() / 8.0f)),
        static_cast<int>(std::ceil(getHalfHeight() / 8.0f)),
        1
    );
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    resetGLStates(7);
}


/*
 * @megamol::compositing_gl::DepthOfField::fill
 */
void megamol::compositing_gl::DepthOfField::fill(
    const std::shared_ptr<glowl::Texture2D>& coc_4,
    const std::shared_ptr<glowl::Texture2D>& coc_near_blurred_4,
    const std::shared_ptr<glowl::Texture2D>& near_4,
    const std::shared_ptr<glowl::Texture2D>& far_4
) {
    fill_prgm_->use();

    glActiveTexture(GL_TEXTURE0);
    bindTextureWithPointSampler(coc_4, 0);
    glUniform1i(fill_prgm_->getUniformLocation("coc_4_point_tx2D"), 0);

    glActiveTexture(GL_TEXTURE1);
    bindTextureWithPointSampler(coc_near_blurred_4, 1);
    glUniform1i(fill_prgm_->getUniformLocation("coc_near_blurred_4_point_tx2D"), 1);

    glActiveTexture(GL_TEXTURE2);
    bindTextureWithPointSampler(near_4, 2);
    glUniform1i(fill_prgm_->getUniformLocation("near_field_4_point_tx2D"), 2);

    glActiveTexture(GL_TEXTURE3);
    bindTextureWithPointSampler(far_4, 3);
    glUniform1i(fill_prgm_->getUniformLocation("far_field_4_point_tx2D"), 3);

    near_field_filled_4_tx2D_->bindImage(0, GL_WRITE_ONLY);
    far_field_filled_4_tx2D_->bindImage(1, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(getHalfWidth() / 8.0f)),
        static_cast<int>(std::ceil(getHalfHeight() / 8.0f)),
        1
    );
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    resetGLStates(4);
}


/*
 * @megamol::compositing_gl::DepthOfField::composite
 */
void megamol::compositing_gl::DepthOfField::composite(
    const std::shared_ptr<glowl::Texture2D>& color,
    const std::shared_ptr<glowl::Texture2D>& coc,
    const std::shared_ptr<glowl::Texture2D>& coc_4,
    const std::shared_ptr<glowl::Texture2D>& coc_near_blurred_4,
    const std::shared_ptr<glowl::Texture2D>& near_fill_4,
    const std::shared_ptr<glowl::Texture2D>& far_fill_4,
    float blend
) {
    composite_prgm_->use();

    glActiveTexture(GL_TEXTURE0);
    bindTextureWithPointSampler(color, 0);
    glUniform1i(composite_prgm_->getUniformLocation("color_point_tx2D"), 0);

    glActiveTexture(GL_TEXTURE1);
    bindTextureWithPointSampler(coc, 1);
    glUniform1i(composite_prgm_->getUniformLocation("coc_point_tx2D"), 1);

    glActiveTexture(GL_TEXTURE2);
    bindTextureWithPointSampler(coc_4, 2);
    glUniform1i(composite_prgm_->getUniformLocation("coc_4_point_tx2D"), 2);

    glActiveTexture(GL_TEXTURE3);
    bindTextureWithBilinearSampler(coc_near_blurred_4, 3);
    glUniform1i(composite_prgm_->getUniformLocation("coc_near_blurred_4_linear_tx2D"), 3);

    glActiveTexture(GL_TEXTURE4);
    bindTextureWithBilinearSampler(near_fill_4, 4);
    glUniform1i(composite_prgm_->getUniformLocation("near_field_filled_4_linear_tx2D"), 4);

    glActiveTexture(GL_TEXTURE5);
    bindTextureWithPointSampler(far_fill_4, 5);
    glUniform1i(composite_prgm_->getUniformLocation("far_field_filled_4_point_tx2D"), 5);

    glUniform1f(composite_prgm_->getUniformLocation("blend"), blend);

    output_tx2D_->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(getWidth() / 8.0f)),
        static_cast<int>(std::ceil(getHeight() / 8.0f)),
        1
    );
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    resetGLStates(6);
}


/*
 * @megamol::compositing_gl::DepthOfField::clearAllTextures
 */
void megamol::compositing_gl::DepthOfField::clearAllTextures() {
    GLubyte col[4] = {0, 0, 0, 0};
    coc_tx2D_->clearTexImage(col);
    color_4_tx2D_->clearTexImage(col);
    color_mul_coc_far_4_tx2D_->clearTexImage(col);
    coc_4_tx2D_->clearTexImage(col);
    coc_near_blurred_4_tx2D_[0]->clearTexImage(col);
    coc_near_blurred_4_tx2D_[1]->clearTexImage(col);
    coc_near_blurred_4_tx2D_[2]->clearTexImage(col);
    coc_near_blurred_4_tx2D_[3]->clearTexImage(col);
    near_field_4_tx2D_->clearTexImage(col);
    far_field_4_tx2D_->clearTexImage(col);
    near_field_filled_4_tx2D_->clearTexImage(col);
    far_field_filled_4_tx2D_->clearTexImage(col);
    output_tx2D_->clearTexImage(col);
}


/*
 * @megamol::compositing_gl::DepthOfField::resizeTexture
 */
void megamol::compositing_gl::DepthOfField::resizeTexture(
    const std::shared_ptr<glowl::Texture2D>& tex, int width, int height) {
    auto tl = tex->getTextureLayout();
    tl.width = width;
    tl.height = height;
    tex->reload(tl, nullptr);
}


/*
 * @megamol::compositing_gl::DepthOfField::reloadAllTextures
 */
void megamol::compositing_gl::DepthOfField::reloadAllTextures() {
    resizeTexture(coc_tx2D_,                   res_[0], res_[1]);
    resizeTexture(color_4_tx2D_,               res_[2], res_[3]);
    resizeTexture(color_mul_coc_far_4_tx2D_,   res_[2], res_[3]);
    resizeTexture(coc_4_tx2D_,                 res_[2], res_[3]);
    resizeTexture(coc_near_blurred_4_tx2D_[0], res_[2], res_[3]);
    resizeTexture(coc_near_blurred_4_tx2D_[1], res_[2], res_[3]);
    resizeTexture(coc_near_blurred_4_tx2D_[2], res_[2], res_[3]);
    resizeTexture(coc_near_blurred_4_tx2D_[3], res_[2], res_[3]);
    resizeTexture(near_field_4_tx2D_,          res_[2], res_[3]);
    resizeTexture(far_field_4_tx2D_,           res_[2], res_[3]);
    resizeTexture(near_field_filled_4_tx2D_,   res_[2], res_[3]);
    resizeTexture(far_field_filled_4_tx2D_,    res_[2], res_[3]);
    resizeTexture(output_tx2D_,                res_[0], res_[1]);
}


/*
 * @megamol::compositing_gl::DepthOfField::getDataCallback
 */
bool megamol::compositing_gl::DepthOfField::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<compositing_gl::CallTexture2D*>(&caller);
    auto rhs_call_input =  input_tex_slot_.CallAs<compositing_gl::CallTexture2D>();
    auto rhs_call_depth =  depth_tex_slot_.CallAs<compositing_gl::CallTexture2D>();
    auto rhs_call_camera = camera_slot_.CallAs<compositing_gl::CallCamera>();

    if (lhs_tc == NULL)
        return false;

    if (rhs_call_input != NULL) {
        // TODO: what does (0) call again?
        if (!(*rhs_call_input)(0))
            return false;
    }

    if (rhs_call_depth != NULL) {
        // TODO: what does (0) call again?
        if (!(*rhs_call_depth)(0))
            return false;
    }

    if (rhs_call_camera != NULL) {
        if (!(*rhs_call_camera)(0))
            return false; 
    }

    bool something_has_changed = (rhs_call_input != NULL ? rhs_call_input->hasUpdate() : false) ||
                                 (rhs_call_input != NULL ? rhs_call_camera->hasUpdate() : false) ||
                                 settings_have_changed_;

    // get input
    auto input_tx2D = rhs_call_input->getData();
    auto depth_tx2D = rhs_call_depth->getData();

    if (something_has_changed) {
#ifdef MEGAMOL_USE_PROFILING
        perf_manager_->start_timer(timers_[0]);
#endif

        int input_width = input_tx2D->getWidth();
        int input_height = input_tx2D->getHeight();
        
        // resize all textures if necessary
        if (input_width != res_[0] || input_height != res_[1]) {
            res_ = glm::ivec4(input_width, input_height, input_width / 2, input_height / 2);

            reloadAllTextures();
        }

        // always clear them to guarantee correct textures
        // TODO: always? or just in case of resize? probably always
        clearAllTextures();

        // set kernel_scale and blend based on param strength from ps_strength_
        float strength = ps_strength_.Param<core::param::FloatParam>()->Value();
        float kernel_scale = 1.f;
        float blend = 1.f;
        // TODO: strength might need to be restricted to some range (possibly [0.25, 1.0]?)
        // otherwise it could blow the texture coordinates out of the park
        if (strength >= 0.25f) {
            kernel_scale = strength;
            //blend = 1.f;
        } else {
            kernel_scale = 0.25f;
            blend = 4.f * strength;
        }

        // calculate ne, nb, fb, fe based on focal distance and focal range
        float focal_distance = ps_focal_distance_.Param<core::param::FloatParam>()->Value();
        float focal_range = ps_focal_range_.Param<core::param::FloatParam>()->Value();

        float ne = std::max(0.f, focal_distance - focal_range);
        float nb = focal_distance;
        float fb = focal_distance;
        float fe = focal_distance + focal_range;

        glm::vec4 fields(ne, nb, fb, fe);

        // get camera to retrieve projection transformation matrix
        // needed to calc viewspace depth from ndc depth in coc generation pass
        core::view::Camera cam = rhs_call_camera->getData();
        glm::mat4 proj_mx = cam.getProjectionMatrix();
        // TODO: actually correct? might be wrong row-major order
        // TODO: possibly try proj_mx[2][3] for the second parameter
        glm::vec2 proj_params(proj_mx[2][2], proj_mx[3][2]);


        // ACTUAL DEPTH OF FIELD CALCULATION
        // ---------------------------------
        cocGeneration(depth_tx2D, proj_params, fields);
        downsample(input_tx2D, coc_tx2D_);
        nearCoCBlur(coc_4_tx2D_);
        computation(color_4_tx2D_, color_mul_coc_far_4_tx2D_, coc_4_tx2D_, coc_near_blurred_4_tx2D_[3], kernel_scale);
        fill(coc_4_tx2D_, coc_near_blurred_4_tx2D_[3], near_field_4_tx2D_, far_field_4_tx2D_);
        composite(input_tx2D, coc_tx2D_, near_field_4_tx2D_, far_field_4_tx2D_, near_field_filled_4_tx2D_, far_field_filled_4_tx2D_, blend);
        // ---------------------------------


#ifdef MEGAMOL_USE_PROFILING
        perf_manager_->stop_timer(timers_[0]);
#endif

        ++version_;
        settings_have_changed_ = false;
    }

    if (tex_inspector_.GetShowInspectorSlotValue()) {
        glm::vec2 tex_dim = glm::vec2(0.0);

        GLuint tex_to_show = 0;
        switch (tex_inspector_.GetSelectTextureSlotValue()) {
        case 0:
            tex_to_show = input_tx2D->getName();
            tex_dim = glm::vec2(input_tx2D->getWidth(), input_tx2D->getHeight());
            break;
        case 1:
            tex_to_show = depth_tx2D->getName();
            tex_dim = glm::vec2(depth_tx2D->getWidth(), depth_tx2D->getHeight());
            break;
        case 2:
            tex_to_show = coc_tx2D_->getName();
            tex_dim = glm::vec2(coc_tx2D_->getWidth(), coc_tx2D_->getHeight());
            break;
        case 3:
            tex_to_show = color_4_tx2D_->getName();
            tex_dim = glm::vec2(color_4_tx2D_->getWidth(), color_4_tx2D_->getHeight());
            break;
        case 4:
            tex_to_show = color_mul_coc_far_4_tx2D_->getName();
            tex_dim = glm::vec2(color_mul_coc_far_4_tx2D_->getWidth(), color_mul_coc_far_4_tx2D_->getHeight());
            break;
        case 5:
            tex_to_show = coc_4_tx2D_->getName();
            tex_dim = glm::vec2(coc_4_tx2D_->getWidth(), coc_4_tx2D_->getHeight());
            break;
        case 6:
            tex_to_show = coc_near_blurred_4_tx2D_[3]->getName();
            tex_dim = glm::vec2(coc_near_blurred_4_tx2D_[3]->getWidth(), coc_near_blurred_4_tx2D_[3]->getHeight());
            break;
        case 7:
            tex_to_show = near_field_4_tx2D_->getName();
            tex_dim = glm::vec2(near_field_4_tx2D_->getWidth(), near_field_4_tx2D_->getHeight());
            break;
        case 8:
            tex_to_show = far_field_4_tx2D_->getName();
            tex_dim = glm::vec2(far_field_4_tx2D_->getWidth(), far_field_4_tx2D_->getHeight());
            break;
        case 9:
            tex_to_show = near_field_filled_4_tx2D_->getName();
            tex_dim = glm::vec2(near_field_filled_4_tx2D_->getWidth(), near_field_filled_4_tx2D_->getHeight());
            break;
        case 10:
            tex_to_show = far_field_filled_4_tx2D_->getName();
            tex_dim = glm::vec2(far_field_filled_4_tx2D_->getWidth(), far_field_filled_4_tx2D_->getHeight());
            break;
        case 11:
            tex_to_show = output_tx2D_->getName();
            tex_dim = glm::vec2(output_tx2D_->getWidth(), output_tx2D_->getHeight());
            break;
        default:
            tex_to_show = output_tx2D_->getName();
            break;
        }

        tex_inspector_.SetTexture((void*) (intptr_t) tex_to_show, tex_dim.x, tex_dim.y);
        tex_inspector_.ShowWindow();
    }


    if (lhs_tc->version() < version_) {
        lhs_tc->setData(output_tx2D_, version_);
    }

    return true;
}


bool megamol::compositing_gl::DepthOfField::textureFormatUpdate() {
    /*glowl::TextureLayout tx_layout(out_format_handler_.getInternalFormat(), 1, 1, 1, out_format_handler_.getFormat(),
        out_format_handler_.getType(), 1);
    output_tx2D_ = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());
    auto shader_options_flags = out_format_handler_.addDefinitions(shader_options);

    try {
        fxaa_prgm_ = core::utility::make_glowl_shader(
            "fxaa", *shader_options_flags, "compositing_gl/DepthOfField/fxaa.comp.glsl");
        smaa_neighborhood_blending_prgm_ = core::utility::make_glowl_shader("smaa_neighborhood_blending",
            *shader_options_flags, "compositing_gl/DepthOfField/smaa_neighborhood_blending.comp.glsl");
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(("DepthOfField: " + std::string(e.what())).c_str());
        return false;
    }*/
    return true;
}

/*
 * @megamol::compositing_gl::DepthOfField::getMetaDataCallback
 */
bool megamol::compositing_gl::DepthOfField::getMetaDataCallback(core::Call& caller) {
    return true;
}
