/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
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
        , tex_inspector_({"Color/Input", "Depth", "CoC", "Color_4", "Color_mul_coc_far_4", "CoC_4", "CoC_near_blurred_4",
              "Near_4", "Far_4", "Near_fill_4", "Far_fill_4", "Output"})
        , output_tex_slot_("OutputTexture", "Gives access to the resulting output texture")
        , input_tex_slot_("InputTexture", "Connects the input texture")
        , depth_tex_slot_("DepthTexture", "Connects the depth texture")
        , settings_have_changed_(false)
        , out_format_handler_("OUTFORMAT",
              {
                  GL_RGBA8_SNORM,
                  GL_RGBA16F,
                  GL_RGBA32F,
              },
              std::function<bool()>(std::bind(&DepthOfField::textureFormatUpdate, this))) {

    /*auto aa_modes = new core::param::EnumParam(1);
    aa_modes->SetTypePair(0, "SMAA");
    aa_modes->SetTypePair(1, "FXAA");
    aa_modes->SetTypePair(2, "None");
    this->mode_.SetParameter(aa_modes);
    this->mode_.SetUpdateCallback(&megamol::compositing_gl::DepthOfField::visibilityCallback);
    this->MakeSlotAvailable(&this->mode_);*/

    auto tex_inspector_slots = this->tex_inspector_.GetParameterSlots();
    for (auto& tex_slot : tex_inspector_slots) {
        this->MakeSlotAvailable(tex_slot);
    }

    this->MakeSlotAvailable(out_format_handler_.getFormatSelectorSlot());

    this->output_tex_slot_.SetCallback(
        compositing_gl::CallTexture2D::ClassName(), "GetData", &DepthOfField::getDataCallback);
    this->output_tex_slot_.SetCallback(
        compositing_gl::CallTexture2D::ClassName(), "GetMetaData", &DepthOfField::getMetaDataCallback);
    this->MakeSlotAvailable(&this->output_tex_slot_);

    this->input_tex_slot_.SetCompatibleCall<compositing_gl::CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->input_tex_slot_);

    this->depth_tex_slot_.SetCompatibleCall<compositing_gl::CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->depth_tex_slot_);
}


/*
 * @megamol::compositing_gl::DepthOfField::~DepthOfField
 */
megamol::compositing_gl::DepthOfField::~DepthOfField() {
    this->Release();
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
    // TODO: need to use out_format_handler_?
    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());
    auto shader_options_flags = out_format_handler_.addDefinitions(shader_options);

    try {
        coc_generation_prgm_ = core::utility::make_glowl_shader("coc_generation/*",
            *shader_options_flags, "compositing_gl/DepthOfField/coc_generation.comp.glsl");

        downsample_prgm_ = core::utility::make_glowl_shader(
            "downsample/*", *shader_options_flags, "compositing_gl/DepthOfField/downsample.comp.glsl");

        near_coc_blur_prgm_ = core::utility::make_glowl_shader(
            "near_coc_blur/*", *shader_options_flags, "compositing_gl/DepthOfField/near_coc_blur.comp.glsl");

        computation_prgm_ = core::utility::make_glowl_shader(
            "computation/*", *shader_options_flags, "compositing_gl/DepthOfField/computation.comp.glsl");

        fill_prgm_ = core::utility::make_glowl_shader(
            "fill/*", *shader_options_flags, "compositing_gl/DepthOfField/fill.comp.glsl");

        composite_prgm_ = core::utility::make_glowl_shader(
            "composite/*", *shader_options_flags, "compositing_gl/DepthOfField/composite.comp.glsl");
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(("DepthOfField: " + std::string(e.what())).c_str());
    }

    // init all textures
    glowl::TextureLayout tx_layout_base = glowl::TextureLayout(GL_RGBA32F, 1, 1, 1, GL_RGBA, GL_FLOAT, 1);
    glowl::TextureLayout tx_layout_r10f_g11f_b10f = glowl::TextureLayout(GL_R11F_G11F_B10F, 1, 1, 1, GL_RGB, GL_FLOAT, 1);
    glowl::TextureLayout tx_layout_r8_g8_unorm = glowl::TextureLayout(GL_RG8, 1, 1, 1, GL_RG, GL_FLOAT, 1);
    glowl::TextureLayout tx_layout_r8_unorm = glowl::TextureLayout(GL_R8, 1, 1, 1, GL_RED, GL_FLOAT, 1);

    // textures for dof
    //color_tx2D_               = std::make_shared<glowl::Texture2D>("color", tx_layout_base, nullptr);
    //depth_tx2D_               = std::make_shared<glowl::Texture2D>("depth", depth_layout_lol, nullptr);
    coc_tx2D_                   = std::make_shared<glowl::Texture2D>("coc", tx_layout_r8_g8_unorm, nullptr);
    color_4_tx2D_               = std::make_shared<glowl::Texture2D>("color_4", tx_layout_r10f_g11f_b10f, nullptr);
    color_mul_coc_far_4_tx2D_   = std::make_shared<glowl::Texture2D>("color_mul_coc_far_4", tx_layout_r10f_g11f_b10f, nullptr);
    coc_4_tx2D_                 = std::make_shared<glowl::Texture2D>("coc_4", tx_layout_r8_g8_unorm, nullptr);
    coc_near_blurred_4_tx2D_    = std::make_shared<glowl::Texture2D>("coc_near_blurred_4", tx_layout_r8_unorm, nullptr);
    near_4_tx2D_                = std::make_shared<glowl::Texture2D>("near_4", tx_layout_r10f_g11f_b10f, nullptr);
    far_4_tx2D_                 = std::make_shared<glowl::Texture2D>("far_4", tx_layout_r10f_g11f_b10f, nullptr);
    near_fill_4_tx2D_           = std::make_shared<glowl::Texture2D>("near_fill_4", tx_layout_r10f_g11f_b10f, nullptr);
    far_fill_4_tx2D_            = std::make_shared<glowl::Texture2D>("far_fill_4", tx_layout_r10f_g11f_b10f, nullptr);
    output_tx2D_                = std::make_shared<glowl::Texture2D>("dof_output", tx_layout_base, nullptr);

    // TODO: create 2 samplers for linear and point sampling
    std::vector<std::pair<GLenum, GLint>> int_params = {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST},
        {GL_TEXTURE_MAG_FILTER, GL_LINEAR}};
    //smaa_layout_ = glowl::TextureLayout(GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1, int_params, {});

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

// TODO: make output texture also as function parameter?
/*
 * @megamol::compositing_gl::DepthOfField::cocGeneration
 */
void megamol::compositing_gl::DepthOfField::cocGeneration(
    const std::shared_ptr<glowl::Texture2D>& depth
) {
    coc_generation_prgm_->use();

    glActiveTexture(GL_TEXTURE0);
    depth->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("depth_tx2D"), 0);

    coc_tx2D_->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(getWidth() / 8.0f)),
        static_cast<int>(std::ceil(getHeight() / 8.0f)),
        1
    );
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    resetGLStates();
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
    color->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("color_tx2D"), 0);
    glActiveTexture(GL_TEXTURE1);
    coc->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("coc_tx2D"), 1);

    color_4_tx2D_->bindImage(0, GL_WRITE_ONLY);
    color_mul_coc_far_4_tx2D_->bindImage(1, GL_WRITE_ONLY);
    coc_4_tx2D_->bindImage(2, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(getHalfWidth() / 8.0f)),
        static_cast<int>(std::ceil(getHalfHeight() / 8.0f)),
        1
    );
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    resetGLStates();
}


/*
 * @megamol::compositing_gl::DepthOfField::nearCoCBlur
 */
void megamol::compositing_gl::DepthOfField::nearCoCBlur(
    const std::shared_ptr<glowl::Texture2D>& coc_4
) {
    near_coc_blur_prgm_->use();

    glActiveTexture(GL_TEXTURE0);
    coc_4->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("coc_4_tx2D"), 0);

    coc_near_blurred_4_tx2D_->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(getHalfWidth() / 8.0f)),
        static_cast<int>(std::ceil(getHalfHeight() / 8.0f)),
        1
    );
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    resetGLStates();
}


/*
 * @megamol::compositing_gl::DepthOfField::computation
 */
void megamol::compositing_gl::DepthOfField::computation(
    const std::shared_ptr<glowl::Texture2D>& color_4,
    const std::shared_ptr<glowl::Texture2D>& color_mul_coc_far_4,
    const std::shared_ptr<glowl::Texture2D>& coc_4,
    const std::shared_ptr<glowl::Texture2D>& coc_near_blurred_4
) {
    computation_prgm_->use();

    glActiveTexture(GL_TEXTURE0);
    color_4->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("color_4_tx2D"), 0);
    glActiveTexture(GL_TEXTURE1);
    color_mul_coc_far_4->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("color_mul_coc_far_4_tx2D"), 1);
    glActiveTexture(GL_TEXTURE2);
    coc_4->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("coc_4_tx2D"), 2);
    glActiveTexture(GL_TEXTURE3);
    coc_near_blurred_4->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("coc_near_blurred_4_tx2D"), 3);

    near_4_tx2D_->bindImage(0, GL_WRITE_ONLY);
    far_4_tx2D_->bindImage(1, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(getHalfWidth() / 8.0f)),
        static_cast<int>(std::ceil(getHalfHeight() / 8.0f)),
        1
    );
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    resetGLStates();
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
    coc_4->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("coc_4_tx2D"), 0);
    glActiveTexture(GL_TEXTURE1);
    coc_near_blurred_4->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("coc_near_blurred_4_tx2D"), 1);
    glActiveTexture(GL_TEXTURE2);
    near_4->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("near_4_tx2D"), 2);
    glActiveTexture(GL_TEXTURE3);
    far_4->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("far_4_tx2D"), 3);

    near_fill_4_tx2D_->bindImage(0, GL_WRITE_ONLY);
    far_fill_4_tx2D_->bindImage(1, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(getHalfWidth() / 8.0f)),
        static_cast<int>(std::ceil(getHalfHeight() / 8.0f)),
        1
    );
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    resetGLStates();
}


/*
 * @megamol::compositing_gl::DepthOfField::composite
 */
void megamol::compositing_gl::DepthOfField::composite(
    const std::shared_ptr<glowl::Texture2D>& color,
    const std::shared_ptr<glowl::Texture2D>& coc,
    const std::shared_ptr<glowl::Texture2D>& near_4,
    const std::shared_ptr<glowl::Texture2D>& far_4,
    const std::shared_ptr<glowl::Texture2D>& near_fill_4,
    const std::shared_ptr<glowl::Texture2D>& far_fill_4
) {
    composite_prgm_->use();

    glActiveTexture(GL_TEXTURE0);
    color->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("coc_4_tx2D"), 0);
    glActiveTexture(GL_TEXTURE1);
    coc->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("coc_near_blurred_4_tx2D"), 1);
    glActiveTexture(GL_TEXTURE2);
    near_4->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("near_4_tx2D"), 2);
    glActiveTexture(GL_TEXTURE3);
    far_4->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("far_4_tx2D"), 3);
    glActiveTexture(GL_TEXTURE4);
    near_fill_4->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("near_fill_4_tx2D"), 4);
    glActiveTexture(GL_TEXTURE5);
    far_fill_4->bindTexture();
    glUniform1i(coc_generation_prgm_->getUniformLocation("far_fill_4_tx2D"), 5);

    output_tx2D_->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(getWidth() / 8.0f)),
        static_cast<int>(std::ceil(getHeight() / 8.0f)),
        1
    );
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    resetGLStates();
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
    coc_near_blurred_4_tx2D_->clearTexImage(col);
    near_4_tx2D_->clearTexImage(col);
    far_4_tx2D_->clearTexImage(col);
    near_fill_4_tx2D_->clearTexImage(col);
    far_fill_4_tx2D_->clearTexImage(col);
    output_tx2D_->clearTexImage(col);
}


/*
 * @megamol::compositing_gl::DepthOfField::reloadAllTextures
 */
void megamol::compositing_gl::DepthOfField::reloadAllTextures() {
    resizeTexture(coc_tx2D_,                 res_[0], res_[1]);
    resizeTexture(color_4_tx2D_,             res_[2], res_[3]);
    resizeTexture(color_mul_coc_far_4_tx2D_, res_[2], res_[3]);
    resizeTexture(coc_4_tx2D_,               res_[2], res_[3]);
    resizeTexture(coc_near_blurred_4_tx2D_,  res_[2], res_[3]);
    resizeTexture(near_4_tx2D_,              res_[2], res_[3]);
    resizeTexture(far_4_tx2D_,               res_[2], res_[3]);
    resizeTexture(near_fill_4_tx2D_,         res_[2], res_[3]);
    resizeTexture(far_fill_4_tx2D_,          res_[2], res_[3]);
    resizeTexture(output_tx2D_,              res_[0], res_[1]);
}


/*
* @megamol::compositing_gl::DepthOfField::resizeTexture
*/
void megamol::compositing_gl::DepthOfField::resizeTexture(
    const std::shared_ptr<glowl::Texture2D>& tex,
    int width,
    int height)
{
    auto tl = tex->getTextureLayout();
    tl.width = width;
    tl.height = height;
    tex->reload(tl, nullptr);
}

/*
 * @megamol::compositing_gl::DepthOfField::getDataCallback
 */
bool megamol::compositing_gl::DepthOfField::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<compositing_gl::CallTexture2D*>(&caller);
    auto rhs_call_input = input_tex_slot_.CallAs<compositing_gl::CallTexture2D>();
    auto rhs_call_depth = depth_tex_slot_.CallAs<compositing_gl::CallTexture2D>();

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

    bool something_has_changed = (rhs_call_input != NULL ? rhs_call_input->hasUpdate() : false) ||
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


        // actual dof
        cocGeneration(depth_tx2D);
        downsample(input_tx2D, coc_tx2D_);
        nearCoCBlur(coc_4_tx2D_);
        computation(color_4_tx2D_, color_mul_coc_far_4_tx2D_, coc_4_tx2D_, coc_near_blurred_4_tx2D_);
        fill(coc_4_tx2D_, coc_near_blurred_4_tx2D_, near_4_tx2D_, far_4_tx2D_);
        composite(input_tx2D, coc_tx2D_, near_4_tx2D_, far_4_tx2D_, near_fill_4_tx2D_, far_fill_4_tx2D_);


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
            tex_to_show = coc_near_blurred_4_tx2D_->getName();
            tex_dim = glm::vec2(coc_near_blurred_4_tx2D_->getWidth(), coc_near_blurred_4_tx2D_->getHeight());
            break;
        case 7:
            tex_to_show = near_4_tx2D_->getName();
            tex_dim = glm::vec2(near_4_tx2D_->getWidth(), near_4_tx2D_->getHeight());
            break;
        case 8:
            tex_to_show = far_4_tx2D_->getName();
            tex_dim = glm::vec2(far_4_tx2D_->getWidth(), far_4_tx2D_->getHeight());
            break;
        case 9:
            tex_to_show = near_fill_4_tx2D_->getName();
            tex_dim = glm::vec2(near_fill_4_tx2D_->getWidth(), near_fill_4_tx2D_->getHeight());
            break;
        case 10:
            tex_to_show = far_fill_4_tx2D_->getName();
            tex_dim = glm::vec2(far_fill_4_tx2D_->getWidth(), far_fill_4_tx2D_->getHeight());
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
