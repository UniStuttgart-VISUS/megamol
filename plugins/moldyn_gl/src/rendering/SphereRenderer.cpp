/*/*
 * SphereRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 *
 */

#include "SphereRenderer.h"
#include "stdafx.h"

#include "mmcore/view/light/DistantLight.h"
#include "mmcore_gl/flags/FlagCallsGL.h"

#include "vislib_gl/graphics/gl/GLSLGeometryShader.h" // only for RequiredExtensions

#include "OpenGL_Context.h"


using namespace megamol::core;
using namespace megamol::geocalls;
using namespace megamol::moldyn_gl::rendering;
using namespace vislib_gl::graphics::gl;


//#define CHRONOTIMING

#define SSBO_GENERATED_SHADER_INSTANCE "gl_VertexID" // or "gl_InstanceID"
#define SSBO_GENERATED_SHADER_ALIGNMENT "std430"     // "std430"
#define AO_DIR_UBO_BINDING_POINT 0


// Beware of changing the binding points
// Need to be changed in shaders accordingly
const GLuint ssbo_flags_binding_point = 2;
const GLuint ssbo_vertex_binding_point = 3;
const GLuint ssbo_color_binding_point = 4;

SphereRenderer::SphereRenderer(void)
        : core_gl::view::Renderer3DModuleGL()
        , get_data_slot_("getdata", "Connects to the data source")
        , get_tf_slot_("gettransferfunction", "The slot for the transfer function module")
        , get_clip_plane_slot_("getclipplane", "The slot for the clipping plane module")
        , read_flags_slot_("readFlags", "The slot for reading the selection flags")
        , get_lights_slot_("lights", "Lights are retrieved over this slot.")
        , cur_view_attrib_()
        , cur_clip_dat_()
        , old_clip_dat_()
        , cur_clip_col_()
        , cur_light_dir_()
        , cur_vp_width_(-1)
        , cur_vp_height_(-1)
        , last_vp_width_(0)
        , last_vp_height_(0)
        , cur_mv_inv_()
        , cur_mv_transp_()
        , cur_mvp_()
        , cur_mvp_inv_()
        , cur_mvp_transp_()
        , shader_options_flags_(nullptr)
        , init_resources_(true)
        , render_mode_(RenderMode::SIMPLE)
        , grey_tf_(0)
        , range_()
        , flags_enabled_(false)
        , flags_available_(false)
        , sphere_prgm_()
        , sphere_geometry_prgm_()
        , lighting_prgm_()
        , ao_dir_ubo_(nullptr)
        , vert_array_()
        , col_type_(SimpleSphericalParticles::ColourDataType::COLDATA_NONE)
        , vert_type_(SimpleSphericalParticles::VertexDataType::VERTDATA_NONE)
        , new_shader_(nullptr)
        , the_shaders_()
        , the_single_buffer_()
        , curr_buf_(0)
        , buf_size_(32 * 1024 * 1024)
        , num_buffers_(3)
        , the_single_mapped_mem_(nullptr)
        , gpu_data_()
        , g_buffer_()
        , old_hash_(-1)
        , old_frame_id_(0)
        , state_invalid_(0)
        , amb_cone_constants_()
        , vol_gen_(nullptr)
        , trigger_rebuild_g_buffer_(false)
// , timer()
#if defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)
        /// This variant should not need the fence (?)
        // ,single_buffer_creation_bits_(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_COHERENT_BIT);
        // ,single_buffer_mapping_bits_(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_COHERENT_BIT);
        , single_buffer_creation_bits_(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT)
        , single_buffer_mapping_bits_(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT)
        , fences_()
#endif // defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)
#ifdef SPHERE_MIN_OGL_SSBO_STREAM
        , streamer_()
        , col_streamer_()
        , buf_array_()
        , col_buf_array_()
#endif // SPHERE_MIN_OGL_SSBO_STREAM
        , render_mode_param_("renderMode", "The sphere render mode.")
        , radius_scaling_param_("scaling", "Scaling factor for particle radii.")
        , force_time_slot_(
              "forceTime", "Flag to force the time code to the specified value. Set to true when rendering a video.")
        , use_local_bbox_param_("useLocalBbox", "Enforce usage of local bbox for camera setup")
        , select_color_param_("flag storage::selectedColor", "Color for selected spheres in flag storage.")
        , soft_select_color_param_("flag storage::softSelectedColor", "Color for soft selected spheres in flag storage.")
        , alpha_scaling_param_("splat::alphaScaling", "Splat: Scaling factor for particle alpha.")
        , attenuate_subpixel_param_(
              "splat::attenuateSubpixel", "Splat: Attenuate alpha of points that should have subpixel size.")
        , use_static_data_param_(
              "ssbo::staticData", "SSBO: Upload data only once per hash change and keep data static on GPU")
        , enable_lighting_slot_("ambient occlusion::enableLighting", "Ambient Occlusion: Enable Lighting")
        , enable_geometry_shader_("ambient occlusion::useGsProxies",
              "Ambient Occlusion: Enables rendering using triangle strips from the geometry shader")
        , ao_vol_size_slot_("ambient occlusion::volumeSize", "Ambient Occlusion: Longest volume edge")
        , ao_cone_apex_slot_("ambient occlusion::apex", "Ambient Occlusion: Cone Apex Angle")
        , ao_offset_slot_("ambient occlusion::offset", "Ambient Occlusion: Offset from Surface")
        , ao_strength_slot_("ambient occlusion::strength", "Ambient Occlusion: Strength")
        , ao_cone_length_slot_("ambient occlusion::coneLength", "Ambient Occlusion: Cone length")
        , use_hp_textures_slot_("ambient occlusion::highPrecisionTexture", "Ambient Occlusion: Use high precision textures")
        , outline_width_slot_("outline::width", "Width of the outline in pixels") {

    this->get_data_slot_.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->get_data_slot_.SetNecessity(AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->get_data_slot_);

    this->get_tf_slot_.SetCompatibleCall<core_gl::view::CallGetTransferFunctionGLDescription>();
    this->get_tf_slot_.SetNecessity(AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->get_tf_slot_);

    this->get_lights_slot_.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->get_lights_slot_.SetNecessity(AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->get_lights_slot_);

    this->get_clip_plane_slot_.SetCompatibleCall<view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->get_clip_plane_slot_);

    this->read_flags_slot_.SetCompatibleCall<core_gl::FlagCallRead_GLDescription>();
    this->MakeSlotAvailable(&this->read_flags_slot_);

    // Initialising enum param with all possible modes (needed for configurator)
    // (Removing not available render modes later in create function)
    param::EnumParam* rmp = new param::EnumParam(this->render_mode_);
    rmp->SetTypePair(RenderMode::SIMPLE, this->getRenderModeString(RenderMode::SIMPLE).c_str());
    rmp->SetTypePair(RenderMode::SIMPLE_CLUSTERED, this->getRenderModeString(RenderMode::SIMPLE_CLUSTERED).c_str());
    rmp->SetTypePair(RenderMode::GEOMETRY_SHADER, this->getRenderModeString(RenderMode::GEOMETRY_SHADER).c_str());
    rmp->SetTypePair(RenderMode::SSBO_STREAM, this->getRenderModeString(RenderMode::SSBO_STREAM).c_str());
    rmp->SetTypePair(RenderMode::BUFFER_ARRAY, this->getRenderModeString(RenderMode::BUFFER_ARRAY).c_str());
    rmp->SetTypePair(RenderMode::SPLAT, this->getRenderModeString(RenderMode::SPLAT).c_str());
    rmp->SetTypePair(RenderMode::AMBIENT_OCCLUSION, this->getRenderModeString(RenderMode::AMBIENT_OCCLUSION).c_str());
    rmp->SetTypePair(RenderMode::OUTLINE, this->getRenderModeString(RenderMode::OUTLINE).c_str());
    this->render_mode_param_ << rmp;
    this->MakeSlotAvailable(&this->render_mode_param_);
    rmp = nullptr;

    this->radius_scaling_param_ << new param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->radius_scaling_param_);

    this->force_time_slot_ << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->force_time_slot_);

    this->use_local_bbox_param_ << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->use_local_bbox_param_);

    this->select_color_param_ << new param::ColorParam(1.0f, 0.0f, 0.0f, 1.0f);
    this->MakeSlotAvailable(&this->select_color_param_);

    this->soft_select_color_param_ << new param::ColorParam(1.0f, 0.5f, 0.5f, 1.0f);
    this->MakeSlotAvailable(&this->soft_select_color_param_);

    this->alpha_scaling_param_ << new param::FloatParam(5.0f);
    this->MakeSlotAvailable(&this->alpha_scaling_param_);

    this->attenuate_subpixel_param_ << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->attenuate_subpixel_param_);

    this->use_static_data_param_ << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->use_static_data_param_);

    this->enable_lighting_slot_ << (new param::BoolParam(false));
    this->MakeSlotAvailable(&this->enable_lighting_slot_);

    this->enable_geometry_shader_ << (new param::BoolParam(false));
    this->MakeSlotAvailable(&this->enable_geometry_shader_);

    this->ao_vol_size_slot_ << (new param::IntParam(128, 1, 1024));
    this->MakeSlotAvailable(&this->ao_vol_size_slot_);

    this->ao_cone_apex_slot_ << (new param::FloatParam(50.0f, 1.0f, 90.0f));
    this->MakeSlotAvailable(&this->ao_cone_apex_slot_);

    this->ao_offset_slot_ << (new param::FloatParam(0.01f, 0.0f, 0.2f));
    this->MakeSlotAvailable(&this->ao_offset_slot_);

    this->ao_strength_slot_ << (new param::FloatParam(1.0f, 0.1f, 20.0f));
    this->MakeSlotAvailable(&this->ao_strength_slot_);

    this->ao_cone_length_slot_ << (new param::FloatParam(0.8f, 0.01f, 1.0f));
    this->MakeSlotAvailable(&this->ao_cone_length_slot_);

    this->use_hp_textures_slot_ << (new param::BoolParam(false));
    this->MakeSlotAvailable(&this->use_hp_textures_slot_);

    this->outline_width_slot_ << (new core::param::FloatParam(2.0f, 0.0f));
    this->MakeSlotAvailable(&this->outline_width_slot_);
}


SphereRenderer::~SphereRenderer(void) {
    this->Release();
}


bool SphereRenderer::GetExtents(core_gl::view::CallRender3DGL& call) {

    auto cr = &call;
    if (cr == nullptr)
        return false;

    MultiParticleDataCall* c2 = this->get_data_slot_.CallAs<MultiParticleDataCall>();
    if ((c2 != nullptr)) {
        c2->SetFrameID(static_cast<unsigned int>(cr->Time()), this->force_time_slot_.Param<param::BoolParam>()->Value());
        if (!(*c2)(1))
            return false;
        cr->SetTimeFramesCount(c2->FrameCount());
        auto const plcount = c2->GetParticleListCount();
        if (this->use_local_bbox_param_.Param<param::BoolParam>()->Value() && plcount > 0) {
            auto bbox = c2->AccessParticles(0).GetBBox();
            auto cbbox = bbox;
            cbbox.Grow(c2->AccessParticles(0).GetGlobalRadius());
            for (unsigned pidx = 1; pidx < plcount; ++pidx) {
                auto temp = c2->AccessParticles(pidx).GetBBox();
                bbox.Union(temp);
                temp.Grow(c2->AccessParticles(pidx).GetGlobalRadius());
                cbbox.Union(temp);
            }
            cr->AccessBoundingBoxes().SetBoundingBox(bbox);
            cr->AccessBoundingBoxes().SetClipBox(cbbox);
        } else {
            cr->AccessBoundingBoxes() = c2->AccessBoundingBoxes();
        }

    } else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
    }
    this->cur_clip_box_ = cr->AccessBoundingBoxes().ClipBox();

    return true;
}


bool SphereRenderer::create(void) {

    ASSERT(IsAvailable());

    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    if (ogl_ctx.isVersionGEQ(1, 4) == 0) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "[SphereRenderer] No render mode available. OpenGL version 1.4 or greater is required.");
        return false;
    }
    if (!ogl_ctx.isExtAvailable("GL_ARB_explicit_attrib_location")) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_WARN,
            "[SphereRenderer] No render mode is available. Extension "
            "GL_ARB_explicit_attrib_location is not available.");
        return false;
    }
    if (!ogl_ctx.isExtAvailable("GL_ARB_conservative_depth")) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_WARN,
            "[SphereRenderer] No render mode is available. Extension GL_ARB_conservative_depth is not available.");
        return false;
    }
    if (!ogl_ctx.areExtAvailable(GLSLShader::RequiredExtensions())) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "[SphereRenderer] No render mode is available. Shader extensions are not available.");
        return false;
    }

    // At least the simple render mode must be available
    ASSERT(this->isRenderModeAvailable(RenderMode::SIMPLE));

    // Reduce to available render modes
    this->SetSlotUnavailable(&this->render_mode_param_);
    this->render_mode_param_.Param<param::EnumParam>()->ClearTypePairs();
    this->render_mode_param_.Param<param::EnumParam>()->SetTypePair(
        RenderMode::SIMPLE, this->getRenderModeString(RenderMode::SIMPLE).c_str());
    if (this->isRenderModeAvailable(RenderMode::SIMPLE_CLUSTERED)) {
        this->render_mode_param_.Param<param::EnumParam>()->SetTypePair(
            RenderMode::SIMPLE_CLUSTERED, this->getRenderModeString(RenderMode::SIMPLE_CLUSTERED).c_str());
    }
    if (this->isRenderModeAvailable(RenderMode::GEOMETRY_SHADER)) {
        this->render_mode_param_.Param<param::EnumParam>()->SetTypePair(
            RenderMode::GEOMETRY_SHADER, this->getRenderModeString(RenderMode::GEOMETRY_SHADER).c_str());
    }
    if (this->isRenderModeAvailable(RenderMode::SSBO_STREAM)) {
        this->render_mode_param_.Param<param::EnumParam>()->SetTypePair(
            RenderMode::SSBO_STREAM, this->getRenderModeString(RenderMode::SSBO_STREAM).c_str());
    }
    if (this->isRenderModeAvailable(RenderMode::SPLAT)) {
        this->render_mode_param_.Param<param::EnumParam>()->SetTypePair(
            RenderMode::SPLAT, this->getRenderModeString(RenderMode::SPLAT).c_str());
    }
    if (this->isRenderModeAvailable(RenderMode::BUFFER_ARRAY)) {
        this->render_mode_param_.Param<param::EnumParam>()->SetTypePair(
            RenderMode::BUFFER_ARRAY, this->getRenderModeString(RenderMode::BUFFER_ARRAY).c_str());
    }
    if (this->isRenderModeAvailable(RenderMode::AMBIENT_OCCLUSION)) {
        this->render_mode_param_.Param<param::EnumParam>()->SetTypePair(
            RenderMode::AMBIENT_OCCLUSION, this->getRenderModeString(RenderMode::AMBIENT_OCCLUSION).c_str());
    }
    if (this->isRenderModeAvailable(RenderMode::OUTLINE)) {
        this->render_mode_param_.Param<param::EnumParam>()->SetTypePair(
            RenderMode::OUTLINE, this->getRenderModeString(RenderMode::OUTLINE).c_str());
    }
    this->MakeSlotAvailable(&this->render_mode_param_);

    // Check initial render mode
    if (!this->isRenderModeAvailable(this->render_mode_)) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_WARN,
            "[SphereRenderer] Render mode '%s' is not available - falling back to render mode '%s'.",
            (this->getRenderModeString(this->render_mode_)).c_str(),
            (this->getRenderModeString(RenderMode::SIMPLE)).c_str());
        // Always available fallback render mode
        this->render_mode_ = RenderMode::SIMPLE;
    }

    // timer_.SetNumRegions(4);
    // const char *regions[4] = {"Upload1", "Upload2", "Upload3", "Rendering"};megamol::
    // timer_.SetRegionNames(4, regions);
    // timer_.SetStatisticsFileName("fullstats.csv");
    // timer_.SetSummaryFileName("summary.csv");
    // timer_.SetMaximumFrames(20, 100);

#ifdef PROFILING
    perf_manager = const_cast<frontend_resources::PerformanceManager*>(
        &frontend_resources.get<frontend_resources::PerformanceManager>());
    frontend_resources::PerformanceManager::basic_timer_config upload_timer, render_timer;
    upload_timer.name = "upload";
    upload_timer.api = frontend_resources::PerformanceManager::query_api::OPENGL;
    render_timer.name = "render";
    render_timer.api = frontend_resources::PerformanceManager::query_api::OPENGL;
    timers_ = perf_manager_->add_timers(this, {upload_timer, render_timer});
#endif

    std::vector<float> dummy = {0};
    ao_dir_ubo_ = std::make_unique<glowl::BufferObject>(GL_UNIFORM_BUFFER, dummy);

    return true;
}


void SphereRenderer::release(void) {
#ifdef PROFILING
    perf_manager->remove_timers(timers);
#endif
    this->resetResources();
}


bool SphereRenderer::resetResources(void) {

    this->select_color_param_.Param<param::ColorParam>()->SetGUIVisible(false);
    this->soft_select_color_param_.Param<param::ColorParam>()->SetGUIVisible(false);

    // Set all render mode dependent parameter to GUI invisible
    // SPLAT
    this->alpha_scaling_param_.Param<param::FloatParam>()->SetGUIVisible(false);
    this->attenuate_subpixel_param_.Param<param::BoolParam>()->SetGUIVisible(false);
    // SSBO
    this->use_static_data_param_.Param<param::BoolParam>()->SetGUIVisible(false);
    // Ambient Occlusion
    this->enable_lighting_slot_.Param<param::BoolParam>()->SetGUIVisible(false);
    this->enable_geometry_shader_.Param<param::BoolParam>()->SetGUIVisible(false);
    this->ao_vol_size_slot_.Param<param::IntParam>()->SetGUIVisible(false);
    this->ao_cone_apex_slot_.Param<param::FloatParam>()->SetGUIVisible(false);
    this->ao_offset_slot_.Param<param::FloatParam>()->SetGUIVisible(false);
    this->ao_strength_slot_.Param<param::FloatParam>()->SetGUIVisible(false);
    this->ao_cone_length_slot_.Param<param::FloatParam>()->SetGUIVisible(false);
    this->use_hp_textures_slot_.Param<param::BoolParam>()->SetGUIVisible(false);
    // Outlining
    this->outline_width_slot_.Param<param::FloatParam>()->SetGUIVisible(false);

    this->flags_enabled_ = false;
    this->flags_available_ = false;

    if (this->grey_tf_ != 0) {
        glDeleteTextures(1, &this->grey_tf_);
    }
    this->grey_tf_ = 0;

    this->the_single_mapped_mem_ = nullptr;

    this->the_shaders_.clear();

    if (this->vol_gen_ != nullptr) {
        delete this->vol_gen_;
        this->vol_gen_ = nullptr;
    }

    this->curr_buf_ = 0;
    this->buf_size_ = (32 * 1024 * 1024);
    this->num_buffers_ = 3;

    this->col_type_ = SimpleSphericalParticles::ColourDataType::COLDATA_NONE;
    this->vert_type_ = SimpleSphericalParticles::VertexDataType::VERTDATA_NONE;

    // AMBIENT OCCLUSION
    if (this->isRenderModeAvailable(RenderMode::AMBIENT_OCCLUSION, true)) {
        for (unsigned int i = 0; i < this->gpu_data_.size(); i++) {
            glDeleteVertexArrays(3, reinterpret_cast<GLuint*>(&(this->gpu_data_[i])));
        }
        this->gpu_data_.clear();

        if (this->g_buffer_.color != 0) {
            glDeleteTextures(1, &this->g_buffer_.color);
        }
        this->g_buffer_.color = 0;
        if (this->g_buffer_.depth != 0) {
            glDeleteTextures(1, &this->g_buffer_.depth);
        }
        this->g_buffer_.depth = 0;
        if (this->g_buffer_.normals != 0) {
            glDeleteTextures(1, &this->g_buffer_.normals);
        }
        this->g_buffer_.normals = 0;

        glDeleteFramebuffers(1, &this->g_buffer_.fbo);
    }

    // SPLAT or BUFFER_ARRAY
    if (this->isRenderModeAvailable(RenderMode::SPLAT, true) ||
        this->isRenderModeAvailable(RenderMode::BUFFER_ARRAY, true)) {

        for (auto& x : fences_) {
            if (x) {
                glDeleteSync(x);
            }
        }
        this->fences_.clear();
        this->fences_.resize(num_buffers_);

        this->single_buffer_creation_bits_ = (GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT);
        this->single_buffer_mapping_bits_ = (GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);

        // Named buffer object is automatically unmapped, too
        glDeleteBuffers(1, &(this->the_single_buffer_));
    }

    // SSBO or SPLAT or BUFFER_ARRAY
    if (this->isRenderModeAvailable(RenderMode::SSBO_STREAM) || this->isRenderModeAvailable(RenderMode::SPLAT) ||
        this->isRenderModeAvailable(RenderMode::BUFFER_ARRAY)) {
        glDeleteVertexArrays(1, &(this->vert_array_));
    }

    return true;
}


bool SphereRenderer::createResources() {

    this->resetResources();

    this->state_invalid_ = true;

    if (!this->isRenderModeAvailable(this->render_mode_)) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_WARN,
            "[SphereRenderer] Render mode: '%s' is not available - falling back to render mode '%s'.",
            (this->getRenderModeString(this->render_mode_)).c_str(),
            (this->getRenderModeString(RenderMode::SIMPLE)).c_str());
        this->render_mode_ = RenderMode::SIMPLE; // Fallback render mode ...
        return false;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
            "[SphereRenderer] Using render mode '%s'.", (this->getRenderModeString(this->render_mode_)).c_str());
    }

    // Fallback transfer function texture
    glGenTextures(1, &this->grey_tf_);
    unsigned char tex[6] = {0, 0, 0, 255, 255, 255};
    glBindTexture(GL_TEXTURE_1D, this->grey_tf_);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glBindTexture(GL_TEXTURE_1D, 0);

    // Check for flag storage availability and get specific shader snippet
    // TODO: test flags!
    // create shader programs
    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
    shader_options_flags_ = std::make_unique<msf::ShaderFactoryOptionsOpenGL>(shader_options);
    
    std::string flags_shader_snippet;
    if (this->flags_available_) {
        shader_options_flags_->addDefinition("flags_available");
    }

    try {
        switch (this->render_mode_) {

        case (RenderMode::SIMPLE):
        case (RenderMode::SIMPLE_CLUSTERED):
        {
            sphere_prgm_.reset();
            sphere_prgm_ = core::utility::make_glowl_shader("sphere_simple", *shader_options_flags_,
                "sphere_renderer/sphere_simple.vert.glsl", "sphere_renderer/sphere_simple.frag.glsl");

            glBindAttribLocation(this->sphere_prgm_->getHandle(), 0, "inPosition");
            glBindAttribLocation(this->sphere_prgm_->getHandle(), 1, "inColor");
            glBindAttribLocation(this->sphere_prgm_->getHandle(), 2, "inColIdx");
        }
        break;

        case (RenderMode::GEOMETRY_SHADER):
        {
            sphere_geometry_prgm_.reset();
            sphere_geometry_prgm_ = core::utility::make_glowl_shader("sphere_geometry", *shader_options_flags_,
                "sphere_renderer/sphere_geometry.vert.glsl",
                    "sphere_renderer/sphere_geometry.geom.glsl", "sphere_renderer/sphere_geometry.frag.glsl");

            glBindAttribLocation(this->sphere_geometry_prgm_->getHandle(), 0, "inPosition");
            glBindAttribLocation(this->sphere_geometry_prgm_->getHandle(), 1, "inColor");
            glBindAttribLocation(this->sphere_geometry_prgm_->getHandle(), 2, "inColIdx");
        }
        break;

        case (RenderMode::SSBO_STREAM):
        {
            this->use_static_data_param_.Param<param::BoolParam>()->SetGUIVisible(true);

            glGenVertexArrays(1, &this->vert_array_);
            glBindVertexArray(this->vert_array_);
            glBindVertexArray(0);
        }
        break;

        case (RenderMode::SPLAT):
        {
            this->alpha_scaling_param_.Param<param::FloatParam>()->SetGUIVisible(true);
            this->attenuate_subpixel_param_.Param<param::BoolParam>()->SetGUIVisible(true);

            glGenVertexArrays(1, &this->vert_array_);
            glBindVertexArray(this->vert_array_);
            glGenBuffers(1, &this->the_single_buffer_);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->the_single_buffer_);
            glBufferStorage(
                GL_SHADER_STORAGE_BUFFER, this->buf_size_ * this->num_buffers_, nullptr, single_buffer_creation_bits_);
            this->the_single_mapped_mem_ = glMapNamedBufferRange(
                this->the_single_buffer_, 0, this->buf_size_ * this->num_buffers_, single_buffer_mapping_bits_);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            glBindVertexArray(0);
        }
        break;

        case (RenderMode::BUFFER_ARRAY):
        {
            sphere_prgm_.reset();
            sphere_prgm_ = core::utility::make_glowl_shader("sphere_bufferarray", *shader_options_flags_,
                "sphere_renderer/sphere_bufferarray.vert.glsl", "sphere_renderer/sphere_bufferarray.frag.glsl");

            glBindAttribLocation(this->sphere_prgm_->getHandle(), 0, "inPosition");
            glBindAttribLocation(this->sphere_prgm_->getHandle(), 1, "inColor");
            glBindAttribLocation(this->sphere_prgm_->getHandle(), 2, "inColIdx");
            
            glGenVertexArrays(1, &this->vert_array_);
            glBindVertexArray(this->vert_array_);
            glGenBuffers(1, &this->the_single_buffer_);
            glBindBuffer(GL_ARRAY_BUFFER, this->the_single_buffer_);
            glBufferStorage(GL_ARRAY_BUFFER, this->buf_size_ * this->num_buffers_, nullptr, single_buffer_creation_bits_);
            this->the_single_mapped_mem_ = glMapNamedBufferRange(
                this->the_single_buffer_, 0, this->buf_size_ * this->num_buffers_, single_buffer_mapping_bits_);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindVertexArray(0);
        }
        break;
        
        case (RenderMode::AMBIENT_OCCLUSION):
        {
            this->enable_lighting_slot_.Param<param::BoolParam>()->SetGUIVisible(true);
            this->enable_geometry_shader_.Param<param::BoolParam>()->SetGUIVisible(true);
            this->ao_vol_size_slot_.Param<param::IntParam>()->SetGUIVisible(true);
            this->ao_cone_apex_slot_.Param<param::FloatParam>()->SetGUIVisible(true);
            this->ao_offset_slot_.Param<param::FloatParam>()->SetGUIVisible(true);
            this->ao_strength_slot_.Param<param::FloatParam>()->SetGUIVisible(true);
            this->ao_cone_length_slot_.Param<param::FloatParam>()->SetGUIVisible(true);
            this->use_hp_textures_slot_.Param<param::BoolParam>()->SetGUIVisible(true);

            this->ao_cone_apex_slot_.ResetDirty();
            this->enable_lighting_slot_.ResetDirty();

            // Generate texture and frame buffer handles
            glGenTextures(1, &this->g_buffer_.color);
            glGenTextures(1, &this->g_buffer_.normals);
            glGenTextures(1, &this->g_buffer_.depth);
            glGenFramebuffers(1, &(this->g_buffer_.fbo));

            // Create the sphere shader
            sphere_prgm_.reset();
            sphere_prgm_ = core::utility::make_glowl_shader("sphere_mdao", *shader_options_flags_,
                "sphere_renderer/sphere_mdao.vert.glsl", "sphere_renderer/sphere_mdao.frag.glsl");

            glBindAttribLocation(this->sphere_prgm_->getHandle(), 0, "inPosition");
            glBindAttribLocation(this->sphere_prgm_->getHandle(), 1, "inColor");
            glBindAttribLocation(this->sphere_prgm_->getHandle(), 2, "inColIdx");

            // Create the geometry shader
            sphere_geometry_prgm_.reset();
            sphere_geometry_prgm_ = core::utility::make_glowl_shader("sphere_mdao_geometry", *shader_options_flags_,
                "sphere_renderer/sphere_mdao_geometry.vert.glsl", "sphere_renderer/sphere_mdao_geometry.geom.glsl",
                "sphere_renderer/sphere_mdao_geometry.frag.glsl");

            glBindAttribLocation(this->sphere_geometry_prgm_->getHandle(), 0, "position");

            // Create the deferred shader
            auto lighting_so = shader_options;

            bool enable_lighting = this->enable_lighting_slot_.Param<param::BoolParam>()->Value();
            if (enable_lighting) {
                lighting_so.addDefinition("ENABLE_LIGHTING");
            }

            float apex = this->ao_cone_apex_slot_.Param<param::FloatParam>()->Value();
            std::vector<glm::vec4> directions;
            this->generate3ConeDirections(directions, apex * static_cast<float>(M_PI) / 180.0f);
            lighting_so.addDefinition("NUM_CONEDIRS", std::to_string(directions.size()));

            ao_dir_ubo_->rebuffer(directions);

            lighting_prgm_.reset();
            lighting_prgm_ = core::utility::make_glowl_shader("sphere_mdao_deferred", lighting_so,
                "sphere_renderer/sphere_mdao_deferred.vert.glsl", "sphere_renderer/sphere_mdao_deferred.frag.glsl");

            // TODO glowl implementation of GLSLprogram misses this functionality
            auto ubo_idx = glGetUniformBlockIndex(lighting_prgm_->getHandle(), "cone_buffer");
            glUniformBlockBinding(lighting_prgm_->getHandle(), ubo_idx, (GLuint)AO_DIR_UBO_BINDING_POINT);

            // Init volume generator
            this->vol_gen_ = new misc::MDAOVolumeGenerator();
            this->vol_gen_->SetShaderSourceFactory(&lighting_so);
            if (!this->vol_gen_->Init(frontend_resources.get<frontend_resources::OpenGL_Context>())) {
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
                    "Error initializing volume generator. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                return false;
            }

            this->trigger_rebuild_g_buffer_ = true;
        }
        break;

        case RenderMode::OUTLINE:
        {
            this->outline_width_slot_.Param<param::FloatParam>()->SetGUIVisible(true);

            // Create the sphere shader
            sphere_prgm_ = core::utility::make_glowl_shader("sphere_outline", *shader_options_flags_,
                "sphere_renderer/sphere_outline.vert.glsl", "sphere_renderer/sphere_outline.frag.glsl");

            glBindAttribLocation(this->sphere_prgm_->getHandle(), 0, "inPosition");
            glBindAttribLocation(this->sphere_prgm_->getHandle(), 1, "inColor");
            glBindAttribLocation(this->sphere_prgm_->getHandle(), 2, "inColIdx");
        }
        break;

        default:
            return false;
        }
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, ("SphereRenderer: " + std::string(e.what())).c_str());
    }

    return true;
}


MultiParticleDataCall* SphereRenderer::getData(unsigned int t, float& out_scaling) {

    MultiParticleDataCall* c2 = this->get_data_slot_.CallAs<MultiParticleDataCall>();
    out_scaling = 1.0f;
    if (c2 != nullptr) {
        c2->SetFrameID(t, this->force_time_slot_.Param<param::BoolParam>()->Value());
        if (!(*c2)(1))
            return nullptr;

        // calculate scaling
        auto const plcount = c2->GetParticleListCount();
        if (this->use_local_bbox_param_.Param<param::BoolParam>()->Value() && plcount > 0) {
            out_scaling = c2->AccessParticles(0).GetBBox().LongestEdge();
            for (unsigned pidx = 0; pidx < plcount; ++pidx) {
                auto const temp = c2->AccessParticles(pidx).GetBBox().LongestEdge();
                if (out_scaling < temp) {
                    out_scaling = temp;
                }
            }
        } else {
            out_scaling = c2->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        }
        if (out_scaling > 0.0000001) {
            out_scaling = 10.0f / out_scaling;
        } else {
            out_scaling = 1.0f;
        }

        c2->SetFrameID(t, this->force_time_slot_.Param<param::BoolParam>()->Value());
        if (!(*c2)(0))
            return nullptr;

        return c2;
    } else {
        return nullptr;
    }
}


void SphereRenderer::getClipData(glm::vec4& out_clip_dat, glm::vec4& out_clip_col) {

    view::CallClipPlane* ccp = this->get_clip_plane_slot_.CallAs<view::CallClipPlane>();
    if ((ccp != nullptr) && (*ccp)()) {
        out_clip_dat[0] = ccp->GetPlane().Normal().X();
        out_clip_dat[1] = ccp->GetPlane().Normal().Y();
        out_clip_dat[2] = ccp->GetPlane().Normal().Z();

        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
        out_clip_dat[3] = grr.Dot(ccp->GetPlane().Normal());

        out_clip_col[0] = static_cast<float>(ccp->GetColour()[0]) / 255.0f;
        out_clip_col[1] = static_cast<float>(ccp->GetColour()[1]) / 255.0f;
        out_clip_col[2] = static_cast<float>(ccp->GetColour()[2]) / 255.0f;
        out_clip_col[3] = static_cast<float>(ccp->GetColour()[3]) / 255.0f;
    } else {
        out_clip_dat[0] = out_clip_dat[1] = out_clip_dat[2] = out_clip_dat[3] = 0.0f;

        out_clip_col[0] = out_clip_col[1] = out_clip_col[2] = 0.75f;
        out_clip_col[3] = 1.0f;
    }
}


bool SphereRenderer::isRenderModeAvailable(RenderMode rm, bool silent) {

    std::string warnstr;
    std::string warnmode =
        "[SphereRenderer] Render Mode '" + SphereRenderer::getRenderModeString(rm) + "' is not available. ";

    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    // Check additonal requirements for each render mode separatly
    switch (rm) {
    case (RenderMode::SIMPLE):
        if (ogl_ctx.isVersionGEQ(1, 4) == 0) {
            warnstr += warnmode + "OpenGL version 1.4 or greater is required.\n";
        }
        break;
    case (RenderMode::SIMPLE_CLUSTERED):
        if (ogl_ctx.isVersionGEQ(1, 4) == 0) {
            warnstr += warnmode + "OpenGL version 1.4 or greater is required.\n";
        }
        break;
    case (RenderMode::GEOMETRY_SHADER):
        if (ogl_ctx.isVersionGEQ(3, 2) == 0) {
            warnstr += warnmode + "OpenGL version 3.2 or greater is required.\n";
        }
        if (!ogl_ctx.areExtAvailable(vislib_gl::graphics::gl::GLSLGeometryShader::RequiredExtensions())) {
            warnstr += warnmode + "Geometry shader extensions are required. \n";
        }
        if (!ogl_ctx.isExtAvailable("GL_EXT_geometry_shader4")) {
            warnstr += warnmode + "Extension GL_EXT_geometry_shader4 is required. \n";
        }
        if (!ogl_ctx.isExtAvailable("GL_EXT_gpu_shader4")) {
            warnstr += warnmode + "Extension GL_EXT_gpu_shader4 is required. \n";
        }
        if (!ogl_ctx.isExtAvailable("GL_EXT_bindable_uniform")) {
            warnstr += warnmode + "Extension GL_EXT_bindable_uniform is required. \n";
        }
        if (!ogl_ctx.isExtAvailable("GL_ARB_shader_objects")) {
            warnstr += warnmode + "Extension GL_ARB_shader_objects is required. \n";
        }
        break;
    case (RenderMode::SSBO_STREAM):
        if (ogl_ctx.isVersionGEQ(4, 2) == 0) {
            warnstr += warnmode + "OpenGL version 4.2 or greater is required. \n";
        }
        if (!ogl_ctx.isExtAvailable("GL_ARB_shader_storage_buffer_object")) {
            warnstr += warnmode + "Extension GL_ARB_shader_storage_buffer_object is required. \n";
        }
        if (!ogl_ctx.isExtAvailable("GL_ARB_gpu_shader5")) {
            warnstr += warnmode + "Extension GL_ARB_gpu_shader5 is required. \n";
        }
        if (!ogl_ctx.isExtAvailable("GL_ARB_gpu_shader_fp64")) {
            warnstr += warnmode + "Extension GL_ARB_gpu_shader_fp64 is required. \n";
        }
        break;
    case (RenderMode::SPLAT):
        if (ogl_ctx.isVersionGEQ(4, 5) == 0) {
            warnstr += warnmode + "OpenGL version 4.5 or greater is required. \n";
        }
        if (!ogl_ctx.isExtAvailable("GL_ARB_shader_storage_buffer_object")) {
            warnstr += warnmode + "Extension GL_ARB_shader_storage_buffer_object is required. \n";
        }
        if (!ogl_ctx.isExtAvailable("GL_EXT_gpu_shader4")) {
            warnstr += warnmode + "Extension GL_EXT_gpu_shader4 is required. \n";
        }
        break;
    case (RenderMode::BUFFER_ARRAY):
        if (ogl_ctx.isVersionGEQ(4, 5) == 0) {
            warnstr += warnmode + "OpenGL version 4.5 or greater is required. \n";
        }
        break;
    case (RenderMode::AMBIENT_OCCLUSION):
        if (ogl_ctx.isVersionGEQ(4, 2) == 0) {
            warnstr += warnmode + "OpenGL version 4.2 or greater is required. \n";
        }
        if (!ogl_ctx.areExtAvailable(vislib_gl::graphics::gl::GLSLGeometryShader::RequiredExtensions())) {
            warnstr += warnmode + "Geometry shader extensions are required. \n";
        }
        if (!ogl_ctx.isExtAvailable("GL_EXT_geometry_shader4")) {
            warnstr += warnmode + "Extension GL_EXT_geometry_shader4 is required. \n";
        }
        if (!ogl_ctx.isExtAvailable("GL_ARB_gpu_shader_fp64")) {
            warnstr += warnmode + "Extension GL_ARB_gpu_shader_fp64 is required. \n";
        }
        if (!ogl_ctx.isExtAvailable("GL_ARB_compute_shader")) {
            warnstr += warnmode + "Extension GL_ARB_compute_shader is required. \n";
        }
        break;
    case (RenderMode::OUTLINE):
        if (ogl_ctx.isVersionGEQ(1, 4) == 0) {
            warnstr += warnmode + "Minimum OpenGL version is 1.4 \n";
        }
        break;
    default:
        warnstr += "[SphereRenderer] Unknown render mode.\n";
        break;
    }

    if (!silent && !warnstr.empty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_WARN, warnstr.c_str());
    }

    return (warnstr.empty());
}


bool SphereRenderer::isFlagStorageAvailable() {
    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();

    auto flagc = this->read_flags_slot_.CallAs<core_gl::FlagCallRead_GL>();

    // Update parameter visibility
    this->select_color_param_.Param<param::ColorParam>()->SetGUIVisible((bool)(flagc != nullptr));
    this->soft_select_color_param_.Param<param::ColorParam>()->SetGUIVisible((bool)(flagc != nullptr));

    if (flagc == nullptr) {
        this->flags_available_ = false;

        return false;
    }

    // Check availbility of flag storage
    this->flags_available_ = true;
    std::string warnstr;
    int major = -1;
    int minor = -1;
    this->getGLSLVersion(major, minor);
    if (!((major == 4) && (minor >= 3) || (major > 4))) {
        warnstr +=
            "[SphereRenderer] Flag Storage is not available. GLSL version greater or equal to 4.3 is required. \n";
        this->flags_available_ = false;
    }

    if (!ogl_ctx.isExtAvailable("GL_ARB_gpu_shader_fp64")) {
        warnstr += "[SphereRenderer] Flag Storage is not available. Extension "
                   "GL_ARB_gpu_shader_fp64 is required. \n";
        this->flags_available_ = false;
    }

    if (!ogl_ctx.isExtAvailable("GL_ARB_shader_storage_buffer_object")) {
        warnstr += "[SphereRenderer] Flag Storage is not available. Extension "
                   "GL_ARB_shader_storage_buffer_object is required. \n";
        this->flags_available_ = false;
    }

    if (!warnstr.empty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_WARN, warnstr.c_str());
        return false;
    }

    return true;
}


std::string SphereRenderer::getRenderModeString(RenderMode rm) {

    std::string mode;

    switch (rm) {
    case (RenderMode::SIMPLE):
        mode = "Simple";
        break;
    case (RenderMode::SIMPLE_CLUSTERED):
        mode = "Simple_Clustered";
        break;
    case (RenderMode::GEOMETRY_SHADER):
        mode = "Geometry_Shader";
        break;
    case (RenderMode::SSBO_STREAM):
        mode = "SSBO_Stream";
        break;
    case (RenderMode::SPLAT):
        mode = "Splat";
        break;
    case (RenderMode::BUFFER_ARRAY):
        mode = "Buffer_Array";
        break;
    case (RenderMode::AMBIENT_OCCLUSION):
        mode = "Ambient_Occlusion";
        break;
    case (RenderMode::OUTLINE):
        mode = "Outline";
        break;
    default:
        mode = "unknown";
        break;
    }

    return mode;
}


bool SphereRenderer::Render(core_gl::view::CallRender3DGL& call) {
    // timer.BeginFrame();

    auto cgtf = this->get_tf_slot_.CallAs<core_gl::view::CallGetTransferFunctionGL>();

    // Get data
    float scaling = 1.0f;
    MultiParticleDataCall* mpdc = this->getData(static_cast<unsigned int>(call.Time()), scaling);
    if (mpdc == nullptr)
        return false;
    // Check if we got a new data set
    const SIZE_T hash = mpdc->DataHash();
    const unsigned int frame_id = mpdc->FrameID();
    this->state_invalid_ = ((hash != this->old_hash_) || (frame_id != this->old_frame_id_));

    // Checking for changed render mode
    auto current_render_mode = static_cast<RenderMode>(this->render_mode_param_.Param<param::EnumParam>()->Value());
    if (this->init_resources_ || (current_render_mode != this->render_mode_)) {
        this->render_mode_ = current_render_mode;
        init_resources_ = false;
        if (!this->createResources()) {
            return false;
        }
    }

    // Update current state variables -----------------------------------------

    // Update data set range_ (only if new data set was loaded, not on frame loading)
    if (hash != this->old_hash_) {                            // or (this->state_invalid_) {
        this->range_[0] = std::numeric_limits<float>::max(); // min
        this->range_[1] = std::numeric_limits<float>::min(); // max
        for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
            MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);
            this->range_[0] = std::min(parts.GetMinColourIndexValue(), this->range_[0]);
            this->range_[1] = std::max(parts.GetMaxColourIndexValue(), this->range_[1]);
        }
        if (cgtf != nullptr) {
            cgtf->SetRange(this->range_);
        }
    }

    this->old_hash_ = hash;
    this->old_frame_id_ = frame_id;

    // Clipping
    this->getClipData(this->cur_clip_dat_, this->cur_clip_col_);

    // Camera
    core::view::Camera cam = call.GetCamera();
    auto view = cam.getViewMatrix();
    auto proj = cam.getProjectionMatrix();
    auto cam_pose = cam.get<core::view::Camera::Pose>();
    auto fbo = call.GetFramebuffer();

    this->cur_cam_pos_ = glm::vec4(cam_pose.position, 1.0);
    this->cur_cam_view_ = glm::vec4(cam_pose.direction, 1.0);
    this->cur_cam_up_ = glm::vec4(cam_pose.up, 1.0);
    this->cur_cam_right_ = glm::vec4(glm::cross(cam_pose.direction, cam_pose.up), 1.0);

    this->cur_mv_inv_ = glm::inverse(view);
    this->cur_mv_transp_ = glm::transpose(view);
    this->cur_mvp_ = proj * view;
    this->cur_mvp_inv_ = glm::inverse(this->cur_mvp_);
    this->cur_mvp_transp_ = glm::transpose(this->cur_mvp_);

    // Lights
    this->cur_light_dir_ = {0.0f, 0.0f, 0.0f, 1.0f};

    auto call_light = get_lights_slot_.CallAs<core::view::light::CallLight>();
    if (call_light != nullptr) {
        if (!(*call_light)(0)) {
            return false;
        }

        auto lights = call_light->getData();
        auto distant_lights = lights.get<core::view::light::DistantLightType>();

        if (distant_lights.size() > 1) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[SphereRenderer] Only one single 'Distant Light' source is supported by this renderer");
        } else if (distant_lights.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("[SphereRenderer] No 'Distant Light' found");
        }

        for (auto const& light : distant_lights) {
            auto use_eyedir = light.eye_direction;
            if (use_eyedir) {
                cur_light_dir_ = cur_cam_view_;
            } else {
                auto light_dir = light.direction;
                if (light_dir.size() == 3) {
                    cur_light_dir_[0] = light_dir[0];
                    cur_light_dir_[1] = light_dir[1];
                    cur_light_dir_[2] = light_dir[2];
                }
                if (light_dir.size() == 4) {
                    cur_light_dir_[3] = light_dir[3];
                }
                /// View Space Lighting. Comment line to change to Object Space Lighting.
                // this->cur_light_dir_ = this->cur_mv_transp_ * this->cur_light_dir_;
            }
            /// TODO Implement missing distant light parameters:
            // light.second.dl_angularDiameter;
            // light.second.lightColor;
            // light.second.lightIntensity;
        }
    }

    // Viewport
    this->cur_vp_width_ = fbo->getWidth();
    this->cur_vp_height_ = fbo->getHeight();
    this->cur_view_attrib_[0] = 0.0f;
    this->cur_view_attrib_[1] = 0.0f;
    this->cur_view_attrib_[2] = static_cast<float>(this->cur_vp_width_);
    this->cur_view_attrib_[3] = static_cast<float>(this->cur_vp_height_);
    if (this->cur_view_attrib_[2] < 1.0f)
        this->cur_view_attrib_[2] = 1.0f;
    if (this->cur_view_attrib_[3] < 1.0f)
        this->cur_view_attrib_[3] = 1.0f;
    this->cur_view_attrib_[2] = 2.0f / this->cur_view_attrib_[2];
    this->cur_view_attrib_[3] = 2.0f / this->cur_view_attrib_[3];

    // ------------------------------------------------------------------------

    // Set OpenGL state ----------------------------------------------------
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS); // Necessary for early depth test in fragment shader (default)
    glEnable(GL_CLIP_DISTANCE0);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    bool retval = false;
    switch (current_render_mode) {
    case (RenderMode::SIMPLE):
        retval = this->renderSimple(call, mpdc);
        break;
    case (RenderMode::SIMPLE_CLUSTERED):
        retval = this->renderSimple(call, mpdc);
        break;
    case (RenderMode::GEOMETRY_SHADER):
        retval = this->renderGeometryShader(call, mpdc);
        break;
    case (RenderMode::SSBO_STREAM):
        retval = this->renderSSBO(call, mpdc);
        break;
    case (RenderMode::SPLAT):
        retval = this->renderSplat(call, mpdc);
        break;
    case (RenderMode::BUFFER_ARRAY):
        retval = this->renderBufferArray(call, mpdc);
        break;
    case (RenderMode::AMBIENT_OCCLUSION):
        retval = this->renderAmbientOcclusion(call, mpdc);
        break;
    case (RenderMode::OUTLINE):
        retval = this->renderOutline(call, mpdc);
        break;
    default:
        break;
    }

    // Reset default OpenGL state ---------------------------------------------
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glDisable(GL_CLIP_DISTANCE0);
    glDisable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDisable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ZERO);
    glDisable(GL_POINT_SPRITE);

    // Save some current data
    this->last_vp_height_ = this->cur_vp_height_;
    this->last_vp_width_ = this->cur_vp_width_;
    for (size_t i = 0; i < 4; i++) {
        this->old_clip_dat_[i] = this->cur_clip_dat_[i];
    }

    // timer.EndFrame();

    return retval;
}


bool SphereRenderer::renderSimple(core_gl::view::CallRender3DGL& call, MultiParticleDataCall* mpdc) {

    this->sphere_prgm_->use();
    this->enableFlagStorage(this->sphere_prgm_, mpdc);

    glUniform4fv(this->sphere_prgm_->getUniformLocation("viewAttr"), 1, glm::value_ptr(this->cur_view_attrib_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camIn"), 1, glm::value_ptr(this->cur_cam_view_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camRight"), 1, glm::value_ptr(this->cur_cam_right_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camUp"), 1, glm::value_ptr(this->cur_cam_up_));
    glUniform1f(
        this->sphere_prgm_->getUniformLocation("scaling"), this->radius_scaling_param_.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphere_prgm_->getUniformLocation("clipDat"), 1, glm::value_ptr(this->cur_clip_dat_));
    glUniform4fv(this->sphere_prgm_->getUniformLocation("clipCol"), 1, glm::value_ptr(this->cur_clip_col_));
    glUniform4fv(this->sphere_prgm_->getUniformLocation("lightDir"), 1, glm::value_ptr(this->cur_light_dir_));
    glUniformMatrix4fv(this->sphere_prgm_->getUniformLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_inv_));
    glUniformMatrix4fv(
        this->sphere_prgm_->getUniformLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_transp_));
    glUniformMatrix4fv(this->sphere_prgm_->getUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_));
    glUniformMatrix4fv(this->sphere_prgm_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_inv_));
    glUniformMatrix4fv(
        this->sphere_prgm_->getUniformLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_transp_));

    GLuint flag_parts_count = 0;
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (!this->enableShaderData(this->sphere_prgm_, parts)) {
            continue;
        }

        glUniform1ui(this->sphere_prgm_->getUniformLocation("flags_enabled"), GLuint(this->flags_enabled_));
        if (this->flags_enabled_) {
            glUniform1ui(this->sphere_prgm_->getUniformLocation("flags_offset"), flag_parts_count);
            glUniform4fv(this->sphere_prgm_->getUniformLocation("flag_selected_col"), 1,
                this->select_color_param_.Param<param::ColorParam>()->Value().data());
            glUniform4fv(this->sphere_prgm_->getUniformLocation("flag_softselected_col"), 1,
                this->soft_select_color_param_.Param<param::ColorParam>()->Value().data());
        }

        GLuint vao, vb, cb;
        if (this->render_mode_ == RenderMode::SIMPLE_CLUSTERED) {
            parts.GetVAOs(vao, vb, cb);
            if (parts.IsVAO()) {
                glBindVertexArray(vao);
                this->enableBufferData(
                    this->sphere_prgm_, parts, vb, parts.GetVertexData(), cb, parts.GetColourData(), true); // or false?
            }
        }
        if ((this->render_mode_ == RenderMode::SIMPLE) || (!parts.IsVAO())) {
            this->enableBufferData(this->sphere_prgm_, parts, 0, parts.GetVertexData(), 0, parts.GetColourData());
        }

#ifdef PROFILING
        perf_manager->start_timer(timers[1], this->GetCoreInstance()->GetFrameID());
#endif
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));
#ifdef PROFILING
        perf_manager->stop_timer(timers[1]);
#endif

        if (this->render_mode_ == RenderMode::SIMPLE_CLUSTERED) {
            if (parts.IsVAO()) {
                glBindVertexArray(0);
            }
        }
        this->disableBufferData(this->sphere_prgm_);
        this->disableShaderData();
        flag_parts_count += parts.GetCount();
    }

    this->disableFlagStorage(this->sphere_prgm_);
    glUseProgram(0); // this->sphere_prgm_.Disable();

    mpdc->Unlock();

    return true;
}


bool SphereRenderer::renderSSBO(core_gl::view::CallRender3DGL& call, MultiParticleDataCall* mpdc) {

#ifdef CHRONOTIMING
    std::vector<std::chrono::steady_clock::time_point> deltas;
    std::chrono::steady_clock::time_point before, after;
#endif

    // this->curr_buf_ = 0;
    GLuint flag_parts_count = 0;
    if (this->state_invalid_) {
        this->buf_array_.resize(mpdc->GetParticleListCount());
        this->col_buf_array_.resize(mpdc->GetParticleListCount());
    }
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (col_type_ != parts.GetColourDataType() || vert_type_ != parts.GetVertexDataType()) {
            this->new_shader_ = this->generateShader(parts, "sphere_ssbo");
        }
        if (this->new_shader_ == nullptr)
            return false;

        this->new_shader_->use();
        this->enableFlagStorage(this->new_shader_, mpdc);
        if (!this->enableShaderData(this->new_shader_, parts)) {
            continue;
        }

        glUniform1ui(this->new_shader_->getUniformLocation("flags_enabled"), GLuint(this->flags_enabled_));
        if (this->flags_enabled_) {
            glUniform1ui(this->new_shader_->getUniformLocation("flags_offset"), flag_parts_count);
            glUniform4fv(this->new_shader_->getUniformLocation("flag_selected_col"), 1,
                this->select_color_param_.Param<param::ColorParam>()->Value().data());
            glUniform4fv(this->new_shader_->getUniformLocation("flag_softselected_col"), 1,
                this->soft_select_color_param_.Param<param::ColorParam>()->Value().data());
        }

        glUniform4fv(this->new_shader_->getUniformLocation("viewAttr"), 1, glm::value_ptr(this->cur_view_attrib_));
        glUniform3fv(this->new_shader_->getUniformLocation("camIn"), 1, glm::value_ptr(this->cur_cam_view_));
        glUniform3fv(this->new_shader_->getUniformLocation("camRight"), 1, glm::value_ptr(this->cur_cam_right_));
        glUniform3fv(this->new_shader_->getUniformLocation("camUp"), 1, glm::value_ptr(this->cur_cam_up_));
        glUniform1f(this->new_shader_->getUniformLocation("scaling"),
            this->radius_scaling_param_.Param<param::FloatParam>()->Value());
        glUniform4fv(this->new_shader_->getUniformLocation("clipDat"), 1, glm::value_ptr(this->cur_clip_dat_));
        glUniform4fv(this->new_shader_->getUniformLocation("clipCol"), 1, glm::value_ptr(this->cur_clip_col_));
        glUniform4fv(this->new_shader_->getUniformLocation("lightDir"), 1, glm::value_ptr(this->cur_light_dir_));
        glUniformMatrix4fv(
            this->new_shader_->getUniformLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_inv_));
        glUniformMatrix4fv(
            this->new_shader_->getUniformLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_transp_));
        glUniformMatrix4fv(this->new_shader_->getUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_));
        glUniformMatrix4fv(
            this->new_shader_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_inv_));
        glUniformMatrix4fv(
            this->new_shader_->getUniformLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_transp_));

        unsigned int col_bytes, vert_bytes, col_stride, vert_stride;
        bool interleaved;
        const bool static_data = this->use_static_data_param_.Param<param::BoolParam>()->Value();
        this->getBytesAndStride(parts, col_bytes, vert_bytes, col_stride, vert_stride, interleaved);

        // does all data reside interleaved in the same memory?
        if (interleaved) {
            if (static_data) {
                auto& buf_a = this->buf_array_[i];
                if (this->state_invalid_ || (buf_a.GetNumChunks() == 0)) {
                    buf_a.SetDataWithSize(parts.GetVertexData(), vert_stride, vert_stride, parts.GetCount(),
                        (GLuint)(2 * 1024 * 1024 * 1024 - 1));
                    // 2 GB - khronos: Most implementations will let you allocate a size up to the limit of GPU memory.
                }
                const GLuint num_chunks = buf_a.GetNumChunks();

                for (GLuint x = 0; x < num_chunks; ++x) {
                    glUniform1i(this->new_shader_->getUniformLocation("instanceOffset"), 0);
                    auto actual_items = buf_a.GetNumItems(x);
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_a.GetHandle(x));
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbo_vertex_binding_point, buf_a.GetHandle(x));
                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, ssbo_vertex_binding_point, buf_a.GetHandle(x), 0,
                        buf_a.GetMaxNumItemsPerChunk() * vert_stride);
                    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(actual_items));
                    //bufA.SignalCompletion();
                }
            } else {
                const GLuint num_chunks = this->streamer_.SetDataWithSize(
                    parts.GetVertexData(), vert_stride, vert_stride, parts.GetCount(), 3, (GLuint)(32 * 1024 * 1024));
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->streamer_.GetHandle());
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbo_vertex_binding_point, this->streamer_.GetHandle());

                for (GLuint x = 0; x < num_chunks; ++x) {
                    GLuint num_items, sync;
                    GLsizeiptr dst_off, dst_len;
                    this->streamer_.UploadChunk(x, num_items, sync, dst_off, dst_len);
                    // streamer_.UploadChunk<float, float>(x, [](float f) -> float { return f + 100.0; },
                    //    numItems, sync, dstOff, dstLen);
                    // megamol::core::utility::log::Log::DefaultLog.WriteInfo("[SphereRenderer] Uploading chunk %u at %lu len %lu", x,
                    // dstOff, dstLen);
                    glUniform1i(this->new_shader_->getUniformLocation("instanceOffset"), 0);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    glBindBufferRange(
                        GL_SHADER_STORAGE_BUFFER, ssbo_vertex_binding_point, this->streamer_.GetHandle(), dst_off, dst_len);
                    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(num_items));
                    this->streamer_.SignalCompletion(sync);
                }
            }
        } else {
            if (static_data) {
                auto& buf_a = this->buf_array_[i];
                auto& col_a = this->col_buf_array_[i];
                if (this->state_invalid_ || (buf_a.GetNumChunks() == 0)) {
                    buf_a.SetDataWithSize(parts.GetVertexData(), vert_stride, vert_stride, parts.GetCount(),
                        (GLuint)(2 * 1024 * 1024 * 1024 - 1));
                    // 2 GB - khronos: Most implementations will let you allocate a size up to the limit of GPU memory.
                    col_a.SetDataWithItems(
                        parts.GetColourData(), col_stride, col_stride, parts.GetCount(), buf_a.GetMaxNumItemsPerChunk());
                }
                const GLuint num_chunks = buf_a.GetNumChunks();

                for (GLuint x = 0; x < num_chunks; ++x) {
                    glUniform1i(this->new_shader_->getUniformLocation("instanceOffset"), 0);
                    auto actual_items = buf_a.GetNumItems(x);
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_a.GetHandle(x));
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbo_vertex_binding_point, buf_a.GetHandle(x));
                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, ssbo_vertex_binding_point, buf_a.GetHandle(x), 0,
                        buf_a.GetMaxNumItemsPerChunk() * vert_stride);
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, col_a.GetHandle(x));
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbo_color_binding_point, col_a.GetHandle(x));
                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, ssbo_color_binding_point, col_a.GetHandle(x), 0,
                        col_a.GetMaxNumItemsPerChunk() * col_stride);
                    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(actual_items));
                    //bufA.SignalCompletion();
                    //colA.SignalCompletion();
                }
            } else {
                const GLuint num_chunks = this->streamer_.SetDataWithSize(
                    parts.GetVertexData(), vert_stride, vert_stride, parts.GetCount(), 3, (GLuint)(32 * 1024 * 1024));
                const GLuint col_size = this->col_streamer_.SetDataWithItems(parts.GetColourData(), col_stride, col_stride,
                    parts.GetCount(), 3, this->streamer_.GetMaxNumItemsPerChunk());
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->streamer_.GetHandle());
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbo_vertex_binding_point, this->streamer_.GetHandle());
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->col_streamer_.GetHandle());
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbo_color_binding_point, this->col_streamer_.GetHandle());

                for (GLuint x = 0; x < num_chunks; ++x) {
                    GLuint num_items, num_items2, sync, sync2;
                    GLsizeiptr dst_off, dst_len, dst_off2, dst_len2;
                    this->streamer_.UploadChunk(x, num_items, sync, dst_off, dst_len);
                    this->col_streamer_.UploadChunk(x, num_items2, sync2, dst_off2, dst_len2);
                    ASSERT(num_items == num_items2);
                    glUniform1i(this->new_shader_->getUniformLocation("instanceOffset"), 0);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    glBindBufferRange(
                        GL_SHADER_STORAGE_BUFFER, ssbo_vertex_binding_point, this->streamer_.GetHandle(), dst_off, dst_len);
                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, ssbo_color_binding_point, this->col_streamer_.GetHandle(),
                        dst_off2, dst_len2);
                    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(num_items));
                    this->streamer_.SignalCompletion(sync);
                    this->col_streamer_.SignalCompletion(sync2);
                }
            }
        }

        this->disableShaderData();
        this->disableFlagStorage(this->new_shader_);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
        glUseProgram(0); // this->new_shader_->Disable();

        flag_parts_count += parts.GetCount();

#ifdef CHRONOTIMING
        printf("waitSignal times:\n");
        for (auto d : deltas) {
            printf("%u, ", d);
        }
        printf("\n");
#endif
    }

    mpdc->Unlock();

    return true;
}


bool SphereRenderer::renderSplat(core_gl::view::CallRender3DGL& call, MultiParticleDataCall* mpdc) {

    // Set OpenGL state -----------------------------------------------
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_BLEND);
    // Should be default for splat rendering (Hint: Background colour should not be WHITE)
    glBlendFunc(GL_ONE, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);

    // Maybe for blending against white, remove pre-mult alpha and use this:
    // @gl.blendFuncSeparate @gl.SRC_ALPHA, @gl.ONE_MINUS_SRC_ALPHA, @gl.ONE, @gl.ONE_MINUS_SRC_ALPHA
    //glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    // glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->the_single_buffer_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbo_vertex_binding_point, this->the_single_buffer_);

    // this->curr_buf_ = 0;
    GLuint flag_parts_count = 0;
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (col_type_ != parts.GetColourDataType() || vert_type_ != parts.GetVertexDataType()) {
            this->new_shader_ = this->generateShader(parts, "sphere_splat");
        }
        if (this->new_shader_ == nullptr)
            return false;

        this->new_shader_->use();
        this->enableFlagStorage(this->new_shader_, mpdc);
        if (!this->enableShaderData(this->new_shader_, parts)) {
            continue;
        }

        glUniform1ui(this->new_shader_->getUniformLocation("flags_enabled"), GLuint(this->flags_enabled_));
        if (this->flags_enabled_) {
            glUniform1ui(this->new_shader_->getUniformLocation("flags_offset"), flag_parts_count);
            glUniform4fv(this->new_shader_->getUniformLocation("flag_selected_col"), 1,
                this->select_color_param_.Param<param::ColorParam>()->Value().data());
            glUniform4fv(this->new_shader_->getUniformLocation("flag_softselected_col"), 1,
                this->soft_select_color_param_.Param<param::ColorParam>()->Value().data());
        }
        glUniform4fv(this->new_shader_->getUniformLocation("viewAttr"), 1, glm::value_ptr(this->cur_view_attrib_));
        glUniform3fv(this->new_shader_->getUniformLocation("camIn"), 1, glm::value_ptr(this->cur_cam_view_));
        glUniform3fv(this->new_shader_->getUniformLocation("camRight"), 1, glm::value_ptr(this->cur_cam_right_));
        glUniform3fv(this->new_shader_->getUniformLocation("camUp"), 1, glm::value_ptr(this->cur_cam_up_));
        glUniform1f(this->new_shader_->getUniformLocation("scaling"),
            this->radius_scaling_param_.Param<param::FloatParam>()->Value());
        glUniform4fv(this->new_shader_->getUniformLocation("clipDat"), 1, glm::value_ptr(this->cur_clip_dat_));
        glUniform4fv(this->new_shader_->getUniformLocation("clipCol"), 1, glm::value_ptr(this->cur_clip_col_));
        glUniform4fv(this->new_shader_->getUniformLocation("lightDir"), 1, glm::value_ptr(this->cur_light_dir_));
        glUniformMatrix4fv(
            this->new_shader_->getUniformLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_inv_));
        glUniformMatrix4fv(
            this->new_shader_->getUniformLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_transp_));
        glUniformMatrix4fv(this->new_shader_->getUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_));
        glUniformMatrix4fv(
            this->new_shader_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_inv_));
        glUniformMatrix4fv(
            this->new_shader_->getUniformLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_transp_));
        glUniform1f(this->new_shader_->getUniformLocation("alphaScaling"),
            this->alpha_scaling_param_.Param<param::FloatParam>()->Value());
        glUniform1i(this->new_shader_->getUniformLocation("attenuateSubpixel"),
            this->attenuate_subpixel_param_.Param<param::BoolParam>()->Value() ? 1 : 0);

        unsigned int col_bytes, vert_bytes, col_stride, vert_stride;
        bool interleaved;
        this->getBytesAndStride(parts, col_bytes, vert_bytes, col_stride, vert_stride, interleaved);

        // curr_buf_ = 0;
        UINT64 num_verts, vert_counter;
        // does all data reside interleaved in the same memory?
        if (interleaved) {

            num_verts = this->buf_size_ / vert_stride;
            const char* curr_vert = static_cast<const char*>(parts.GetVertexData());
            const char* curr_col = static_cast<const char*>(parts.GetColourData());
            vert_counter = 0;
            while (vert_counter < parts.GetCount()) {
                // GLuint vb = this->theBuffers[curr_buf_];
                void* mem = static_cast<char*>(the_single_mapped_mem_) + buf_size_ * this->curr_buf_;
                curr_col = col_stride == 0 ? curr_vert : curr_col;
                // currCol = currCol == 0 ? currVert : currCol;
                const char* whence = curr_vert < curr_col ? curr_vert : curr_col;
                UINT64 verts_this_time = vislib::math::Min(parts.GetCount() - vert_counter, num_verts);
                this->waitSingle(this->fences_[this->curr_buf_]);
                /// megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR, "Memcopying %u bytes from %016" PRIxPTR " to %016" PRIxPTR ". [%s, %s, line %d]\n", vertsThisTime * vertStride, whence, mem, __FILE__, __FUNCTION__, __LINE__);
                memcpy(mem, whence, verts_this_time * vert_stride);
                glFlushMappedNamedBufferRange(
                    this->the_single_buffer_, buf_size_ * this->curr_buf_, verts_this_time * vert_stride);
                // glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
                // glUniform1i(this->new_shader_->ParameterLocation("instanceOffset"), numVerts * curr_buf_);
                glUniform1i(this->new_shader_->getUniformLocation("instanceOffset"), 0);

                // this->the_single_buffer_, reinterpret_cast<const void *>(currCol - whence));
                // glBindBuffer(GL_ARRAY_BUFFER, 0);
                glBindBufferRange(GL_SHADER_STORAGE_BUFFER, ssbo_vertex_binding_point, this->the_single_buffer_,
                    buf_size_ * this->curr_buf_, buf_size_);
                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(verts_this_time));
                // glDrawArraysInstanced(GL_POINTS, 0, 1, vertsThisTime);
                this->lockSingle(this->fences_[this->curr_buf_]);

                this->curr_buf_ = (this->curr_buf_ + 1) % this->num_buffers_;
                vert_counter += verts_this_time;
                curr_vert += verts_this_time * vert_stride;
                curr_col += verts_this_time * col_stride;
            }
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_WARN,
                "[SphereRenderer] Splat mode does not support not interleaved data so far ...");
        }

        this->disableShaderData();
        this->disableFlagStorage(this->new_shader_);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
        glUseProgram(0); // new_shader_->Disable();

        flag_parts_count += parts.GetCount();
    }

    mpdc->Unlock();

    return true;
}


bool SphereRenderer::renderBufferArray(core_gl::view::CallRender3DGL& call, MultiParticleDataCall* mpdc) {

    this->sphere_prgm_->use();
    this->enableFlagStorage(this->sphere_prgm_, mpdc);

    glUniform4fv(this->sphere_prgm_->getUniformLocation("viewAttr"), 1, glm::value_ptr(this->cur_view_attrib_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camIn"), 1, glm::value_ptr(this->cur_cam_view_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camRight"), 1, glm::value_ptr(this->cur_cam_right_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camUp"), 1, glm::value_ptr(this->cur_cam_up_));
    glUniform1f(
        this->sphere_prgm_->getUniformLocation("scaling"), this->radius_scaling_param_.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphere_prgm_->getUniformLocation("clipDat"), 1, glm::value_ptr(this->cur_clip_dat_));
    glUniform4fv(this->sphere_prgm_->getUniformLocation("clipCol"), 1, glm::value_ptr(this->cur_clip_col_));
    glUniform4fv(this->sphere_prgm_->getUniformLocation("lightDir"), 1, glm::value_ptr(this->cur_light_dir_));
    glUniformMatrix4fv(this->sphere_prgm_->getUniformLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_inv_));
    glUniformMatrix4fv(
        this->sphere_prgm_->getUniformLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_transp_));
    glUniformMatrix4fv(this->sphere_prgm_->getUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_));
    glUniformMatrix4fv(this->sphere_prgm_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_inv_));
    glUniformMatrix4fv(
        this->sphere_prgm_->getUniformLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_transp_));

    //this->curr_buf_ = 0;
    GLuint flag_parts_count = 0;
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (!this->enableShaderData(this->sphere_prgm_, parts)) {
            continue;
        }

        glUniform1ui(this->sphere_prgm_->getUniformLocation("flags_enabled"), GLuint(this->flags_enabled_));
        if (this->flags_enabled_) {
            glUniform4fv(this->sphere_prgm_->getUniformLocation("flag_selected_col"), 1,
                this->select_color_param_.Param<param::ColorParam>()->Value().data());
            glUniform4fv(this->sphere_prgm_->getUniformLocation("flag_softselected_col"), 1,
                this->soft_select_color_param_.Param<param::ColorParam>()->Value().data());
        }

        unsigned int col_bytes, vert_bytes, col_stride, vert_stride;
        bool interleaved;
        this->getBytesAndStride(parts, col_bytes, vert_bytes, col_stride, vert_stride, interleaved);

        UINT64 num_verts, vert_counter;
        // does all data reside interleaved in the same memory?
        if (interleaved) {

            num_verts = this->buf_size_ / vert_stride;
            const char* curr_vert = static_cast<const char*>(parts.GetVertexData());
            const char* curr_col = static_cast<const char*>(parts.GetColourData());
            vert_counter = 0;
            while (vert_counter < parts.GetCount()) {
                // GLuint vb = this->theBuffers[curr_buf_];
                void* mem = static_cast<char*>(this->the_single_mapped_mem_) + num_verts * vert_stride * this->curr_buf_;
                curr_col = col_stride == 0 ? curr_vert : curr_col;
                // currCol = currCol == 0 ? currVert : currCol;
                const char* whence = curr_vert < curr_col ? curr_vert : curr_col;
                UINT64 verts_this_time = vislib::math::Min(parts.GetCount() - vert_counter, num_verts);
                this->waitSingle(this->fences_[this->curr_buf_]);
                /// megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR, "Memcopying %u bytes from %016" PRIxPTR " to %016" PRIxPTR ". [%s, %s, line %d]\n", vertsThisTime * vertStride, whence, mem, __FILE__, __FUNCTION__, __LINE__);
                memcpy(mem, whence, verts_this_time * vert_stride);
                glFlushMappedNamedBufferRange(
                    this->the_single_buffer_, num_verts * this->curr_buf_, verts_this_time * vert_stride);
                // glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);

                if (this->flags_enabled_) {
                    // Adapting flag offset to ring buffer gl_VertexID
                    glUniform1ui(this->sphere_prgm_->getUniformLocation("flags_offset"),
                        flag_parts_count - static_cast<GLuint>(num_verts * this->curr_buf_));
                }
                this->enableBufferData(this->sphere_prgm_, parts, this->the_single_buffer_,
                    reinterpret_cast<const void*>(curr_vert - whence), this->the_single_buffer_,
                    reinterpret_cast<const void*>(curr_col - whence));

                glDrawArrays(
                    GL_POINTS, static_cast<GLint>(num_verts * this->curr_buf_), static_cast<GLsizei>(verts_this_time));
                this->lockSingle(this->fences_[this->curr_buf_]);

                this->curr_buf_ = (this->curr_buf_ + 1) % this->num_buffers_;
                vert_counter += verts_this_time;
                curr_vert += verts_this_time * vert_stride;
                curr_col += verts_this_time * col_stride;
            }
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_WARN,
                "[SphereRenderer] BufferArray mode does not support not interleaved data so far ...");
        }

        this->disableBufferData(this->sphere_prgm_);
        this->disableShaderData();
        flag_parts_count += parts.GetCount();
    }

    this->disableFlagStorage(this->sphere_prgm_);
    glUseProgram(0); // this->sphere_prgm_.Disable();

    mpdc->Unlock();

    return true;
}


bool SphereRenderer::renderGeometryShader(core_gl::view::CallRender3DGL& call, MultiParticleDataCall* mpdc) {

    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    /// Default GL_LESS works, too?
    // glDepthFunc(GL_LEQUAL);

    /// If enabled and a vertex shader is active, it specifies that the GL will choose between front and
    /// back colors based on the polygon's face direction of which the vertex being shaded is a part.
    /// It has no effect on points or lines and has significant negative performance impact.
    // glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);

    this->sphere_geometry_prgm_->use();
    this->enableFlagStorage(this->sphere_geometry_prgm_, mpdc);

    // Set shader variables
    glUniform4fv(this->sphere_geometry_prgm_->getUniformLocation("viewAttr"), 1, glm::value_ptr(this->cur_view_attrib_));
    glUniform3fv(this->sphere_geometry_prgm_->getUniformLocation("camIn"), 1, glm::value_ptr(this->cur_cam_view_));
    glUniform3fv(this->sphere_geometry_prgm_->getUniformLocation("camRight"), 1, glm::value_ptr(this->cur_cam_right_));
    glUniform3fv(this->sphere_geometry_prgm_->getUniformLocation("camUp"), 1, glm::value_ptr(this->cur_cam_up_));
    glUniform1f(this->sphere_geometry_prgm_->getUniformLocation("scaling"),
        this->radius_scaling_param_.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphere_geometry_prgm_->getUniformLocation("clipDat"), 1, glm::value_ptr(this->cur_clip_dat_));
    glUniform4fv(this->sphere_geometry_prgm_->getUniformLocation("clipCol"), 1, glm::value_ptr(this->cur_clip_col_));
    glUniform4fv(this->sphere_geometry_prgm_->getUniformLocation("lightDir"), 1, glm::value_ptr(this->cur_light_dir_));
    glUniformMatrix4fv(
        this->sphere_geometry_prgm_->getUniformLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_inv_));
    glUniformMatrix4fv(
        this->sphere_geometry_prgm_->getUniformLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_transp_));
    glUniformMatrix4fv(this->sphere_geometry_prgm_->getUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_));
    glUniformMatrix4fv(
        this->sphere_geometry_prgm_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_inv_));
    glUniformMatrix4fv(
        this->sphere_geometry_prgm_->getUniformLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_transp_));

    GLuint flag_parts_count = 0;
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (!this->enableShaderData(this->sphere_geometry_prgm_, parts)) {
            continue;
        }

        glUniform1ui(this->sphere_geometry_prgm_->getUniformLocation("flags_enabled"), GLuint(this->flags_enabled_));
        if (this->flags_enabled_) {
            glUniform1ui(this->sphere_geometry_prgm_->getUniformLocation("flags_offset"), flag_parts_count);
            glUniform4fv(this->sphere_geometry_prgm_->getUniformLocation("flag_selected_col"), 1,
                this->select_color_param_.Param<param::ColorParam>()->Value().data());
            glUniform4fv(this->sphere_geometry_prgm_->getUniformLocation("flag_softselected_col"), 1,
                this->soft_select_color_param_.Param<param::ColorParam>()->Value().data());
        }

        this->enableBufferData(this->sphere_geometry_prgm_, parts, 0, parts.GetVertexData(), 0, parts.GetColourData());

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

        this->disableBufferData(this->sphere_geometry_prgm_);
        this->disableShaderData();
        flag_parts_count += parts.GetCount();
    }

    this->disableFlagStorage(this->sphere_geometry_prgm_);
    glUseProgram(0); // this->sphere_geometry_prgm_.Disable();

    // glDisable(GL_VERTEX_PROGRAM_TWO_SIDE);
    // glDepthFunc(GL_LESS);

    mpdc->Unlock();

    return true;
}


bool SphereRenderer::renderAmbientOcclusion(core_gl::view::CallRender3DGL& call, MultiParticleDataCall* mpdc) {

    // We need to regenerate the shader if certain settings are changed
    if (this->enable_lighting_slot_.IsDirty() || this->ao_cone_apex_slot_.IsDirty()) {

        this->ao_cone_apex_slot_.ResetDirty();
        this->enable_lighting_slot_.ResetDirty();

        this->createResources();
    }

    // Rebuild the g_buffer_ if neccessary
    this->rebuildGBuffer();

    // Render the particles' geometry
    bool high_precision = this->use_hp_textures_slot_.Param<param::BoolParam>()->Value();

    // Choose shader
    bool use_geo = this->enable_geometry_shader_.Param<param::BoolParam>()->Value();
    std::shared_ptr<glowl::GLSLProgram> the_shader = use_geo ? this->sphere_geometry_prgm_ : this->sphere_prgm_;

    // Rebuild and reupload working data if neccessary
    this->rebuildWorkingData(call, mpdc, the_shader);

    GLint prev_fbo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);

    glBindFramebuffer(GL_FRAMEBUFFER, this->g_buffer_.fbo);
    GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, bufs);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindFragDataLocation(the_shader->getHandle(), 0, "outColor");
    glBindFragDataLocation(the_shader->getHandle(), 1, "outNormal");

    the_shader->use();
    this->enableFlagStorage(the_shader, mpdc);

    glUniformMatrix4fv(the_shader->getUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_));
    glUniformMatrix4fv(the_shader->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_inv_));
    glUniformMatrix4fv(the_shader->getUniformLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_inv_));
    glUniformMatrix4fv(the_shader->getUniformLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_transp_));
    glUniformMatrix4fv(the_shader->getUniformLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_transp_));
    glUniform1f(the_shader->getUniformLocation("scaling"), this->radius_scaling_param_.Param<param::FloatParam>()->Value());
    glUniform4fv(the_shader->getUniformLocation("viewAttr"), 1, glm::value_ptr(this->cur_view_attrib_));
    glUniform3fv(the_shader->getUniformLocation("camRight"), 1, glm::value_ptr(this->cur_cam_right_));
    glUniform3fv(the_shader->getUniformLocation("camUp"), 1, glm::value_ptr(this->cur_cam_up_));
    glUniform3fv(the_shader->getUniformLocation("camIn"), 1, glm::value_ptr(this->cur_cam_view_));
    glUniform4fv(the_shader->getUniformLocation("clipDat"), 1, glm::value_ptr(this->cur_clip_dat_));
    glUniform4fv(the_shader->getUniformLocation("clipCol"), 1, glm::value_ptr(this->cur_clip_col_));
    glUniform1i(the_shader->getUniformLocation("inUseHighPrecision"), (int)high_precision);

    GLuint flag_parts_count = 0;
    for (unsigned int i = 0; i < this->gpu_data_.size(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (!this->enableShaderData(the_shader, parts)) {
            continue;
        }

        glUniform1ui(the_shader->getUniformLocation("flags_enabled"), GLuint(this->flags_enabled_));
        if (this->flags_enabled_) {
            glUniform1ui(the_shader->getUniformLocation("flags_offset"), flag_parts_count);
            glUniform4fv(the_shader->getUniformLocation("flag_selected_col"), 1,
                this->select_color_param_.Param<param::ColorParam>()->Value().data());
            glUniform4fv(the_shader->getUniformLocation("flag_softselected_col"), 1,
                this->soft_select_color_param_.Param<param::ColorParam>()->Value().data());
        }

        glBindVertexArray(this->gpu_data_[i].vertex_array);

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

        this->disableShaderData();
        flag_parts_count += parts.GetCount();
    }

    glBindVertexArray(0);
    this->disableFlagStorage(the_shader);
    glUseProgram(0); // the_shader.Disable();

    glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);

    // Deferred rendering pass
    this->renderDeferredPass(call);

    return true;
}


bool SphereRenderer::renderOutline(core_gl::view::CallRender3DGL& call, MultiParticleDataCall* mpdc) {

    this->sphere_prgm_->use();
    this->enableFlagStorage(this->sphere_prgm_, mpdc);

    glUniform4fv(this->sphere_prgm_->getUniformLocation("viewAttr"), 1, glm::value_ptr(this->cur_view_attrib_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camIn"), 1, glm::value_ptr(this->cur_cam_view_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camRight"), 1, glm::value_ptr(this->cur_cam_right_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camUp"), 1, glm::value_ptr(this->cur_cam_up_));
    glUniform1f(
        this->sphere_prgm_->getUniformLocation("scaling"), this->radius_scaling_param_.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphere_prgm_->getUniformLocation("clipDat"), 1, glm::value_ptr(this->cur_clip_dat_));
    glUniform4fv(this->sphere_prgm_->getUniformLocation("clipCol"), 1, glm::value_ptr(this->cur_clip_col_));
    glUniform4fv(this->sphere_prgm_->getUniformLocation("lightDir"), 1, glm::value_ptr(this->cur_light_dir_));
    glUniformMatrix4fv(this->sphere_prgm_->getUniformLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_inv_));
    glUniformMatrix4fv(
        this->sphere_prgm_->getUniformLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_transp_));
    glUniformMatrix4fv(this->sphere_prgm_->getUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_));
    glUniformMatrix4fv(this->sphere_prgm_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_inv_));
    glUniformMatrix4fv(
        this->sphere_prgm_->getUniformLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_transp_));

    glUniform1f(this->sphere_prgm_->getUniformLocation("outlineWidth"),
        this->outline_width_slot_.Param<param::FloatParam>()->Value());

    GLuint flag_parts_count = 0;
    for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        if (!this->enableShaderData(this->sphere_prgm_, parts)) {
            continue;
        }

        glUniform1ui(this->sphere_prgm_->getUniformLocation("flags_enabled"), GLuint(this->flags_enabled_));
        if (this->flags_enabled_) {
            glUniform1ui(this->sphere_prgm_->getUniformLocation("flags_offset"), flag_parts_count);
            glUniform4fv(this->sphere_prgm_->getUniformLocation("flag_selected_col"), 1,
                this->select_color_param_.Param<param::ColorParam>()->Value().data());
            glUniform4fv(this->sphere_prgm_->getUniformLocation("flag_softselected_col"), 1,
                this->soft_select_color_param_.Param<param::ColorParam>()->Value().data());
        }

        this->enableBufferData(this->sphere_prgm_, parts, 0, parts.GetVertexData(), 0, parts.GetColourData());

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

        this->disableBufferData(this->sphere_prgm_);
        this->disableShaderData();
        flag_parts_count += parts.GetCount();
    }

    this->disableFlagStorage(this->sphere_prgm_);
    glUseProgram(0); // this->sphere_prgm_.Disable();

    mpdc->Unlock();

    return true;
}


bool SphereRenderer::enableBufferData(const std::shared_ptr<glowl::GLSLProgram> prgm,
    const MultiParticleDataCall::Particles& parts, GLuint vert_buf, const void* vert_ptr, GLuint col_buf,
    const void* col_ptr, bool create_buffer_data) {

    GLuint vert_attrib_loc = glGetAttribLocation(prgm->getHandle(), "inPosition");
    GLuint col_attrib_loc = glGetAttribLocation(prgm->getHandle(), "inColor");
    GLuint col_idx_attrib_loc = glGetAttribLocation(prgm->getHandle(), "inColIdx");

    const void* color_ptr = col_ptr;
    const void* vertex_ptr = vert_ptr;
    if (create_buffer_data) {
        color_ptr = nullptr;
        vertex_ptr = nullptr;
    }

    unsigned int part_count = static_cast<unsigned int>(parts.GetCount());

    // colour
    glBindBuffer(GL_ARRAY_BUFFER, col_buf);
    switch (parts.GetColourDataType()) {
    case MultiParticleDataCall::Particles::COLDATA_NONE:
        break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
        if (create_buffer_data) {
            glBufferData(GL_ARRAY_BUFFER, part_count * (std::max)(parts.GetColourDataStride(), 3u),
                parts.GetColourData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(col_attrib_loc);
        glVertexAttribPointer(col_attrib_loc, 3, GL_UNSIGNED_BYTE, GL_TRUE, parts.GetColourDataStride(), color_ptr);
        break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
        if (create_buffer_data) {
            glBufferData(GL_ARRAY_BUFFER, part_count * (std::max)(parts.GetColourDataStride(), 4u),
                parts.GetColourData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(col_attrib_loc);
        glVertexAttribPointer(col_attrib_loc, 4, GL_UNSIGNED_BYTE, GL_TRUE, parts.GetColourDataStride(), color_ptr);
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
        if (create_buffer_data) {
            glBufferData(GL_ARRAY_BUFFER,
                part_count * (std::max)(parts.GetColourDataStride(), static_cast<unsigned int>(3 * sizeof(float))),
                parts.GetColourData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(col_attrib_loc);
        glVertexAttribPointer(col_attrib_loc, 3, GL_FLOAT, GL_TRUE, parts.GetColourDataStride(), color_ptr);
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
        if (create_buffer_data) {
            glBufferData(GL_ARRAY_BUFFER,
                part_count * (std::max)(parts.GetColourDataStride(), static_cast<unsigned int>(4 * sizeof(float))),
                parts.GetColourData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(col_attrib_loc);
        glVertexAttribPointer(col_attrib_loc, 4, GL_FLOAT, GL_TRUE, parts.GetColourDataStride(), color_ptr);
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
    case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I: {
        if (create_buffer_data) {
            glBufferData(GL_ARRAY_BUFFER,
                part_count * (std::max)(parts.GetColourDataStride(), static_cast<unsigned int>(1 * sizeof(float))),
                parts.GetColourData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(col_idx_attrib_loc);
        if (parts.GetColourDataType() == MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
            glVertexAttribPointer(col_idx_attrib_loc, 1, GL_FLOAT, GL_FALSE, parts.GetColourDataStride(), color_ptr);
        } else {
            glVertexAttribPointer(col_idx_attrib_loc, 1, GL_DOUBLE, GL_FALSE, parts.GetColourDataStride(), color_ptr);
        }
    } break;
    case MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA:
        if (create_buffer_data) {
            glBufferData(GL_ARRAY_BUFFER,
                part_count *
                    (std::max)(parts.GetColourDataStride(), static_cast<unsigned int>(4 * sizeof(unsigned short))),
                parts.GetColourData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(col_attrib_loc);
        glVertexAttribPointer(col_attrib_loc, 4, GL_UNSIGNED_SHORT, GL_TRUE, parts.GetColourDataStride(), color_ptr);
        break;
    default:
        break;
    }

    // radius and position
    glBindBuffer(GL_ARRAY_BUFFER, vert_buf);
    switch (parts.GetVertexDataType()) {
    case MultiParticleDataCall::Particles::VERTDATA_NONE:
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        if (create_buffer_data) {
            glBufferData(GL_ARRAY_BUFFER,
                part_count * (std::max)(parts.GetVertexDataStride(), static_cast<unsigned int>(3 * sizeof(float))),
                parts.GetVertexData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(vert_attrib_loc);
        glVertexAttribPointer(vert_attrib_loc, 3, GL_FLOAT, GL_FALSE, parts.GetVertexDataStride(), vertex_ptr);
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
        if (create_buffer_data) {
            glBufferData(GL_ARRAY_BUFFER,
                part_count * (std::max)(parts.GetVertexDataStride(), static_cast<unsigned int>(4 * sizeof(float))),
                parts.GetVertexData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(vert_attrib_loc);
        glVertexAttribPointer(vert_attrib_loc, 4, GL_FLOAT, GL_FALSE, parts.GetVertexDataStride(), vertex_ptr);
        break;
    case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
        if (create_buffer_data) {
            glBufferData(GL_ARRAY_BUFFER,
                part_count * (std::max)(parts.GetVertexDataStride(), static_cast<unsigned int>(3 * sizeof(double))),
                parts.GetVertexData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(vert_attrib_loc);
        glVertexAttribPointer(vert_attrib_loc, 3, GL_DOUBLE, GL_FALSE, parts.GetVertexDataStride(), vertex_ptr);
        break;
    case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
        if (create_buffer_data) {
            glBufferData(GL_ARRAY_BUFFER,
                part_count * (std::max)(parts.GetVertexDataStride(), static_cast<unsigned int>(3 * sizeof(short))),
                parts.GetVertexData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(vert_attrib_loc);
        glVertexAttribPointer(vert_attrib_loc, 3, GL_SHORT, GL_FALSE, parts.GetVertexDataStride(), vertex_ptr);
        break;
    default:
        break;
    }

    return true;
}


bool SphereRenderer::disableBufferData(const std::shared_ptr<glowl::GLSLProgram> prgm) {

    GLuint vert_attrib_loc = glGetAttribLocation(prgm->getHandle(), "inPosition");
    GLuint col_attrib_loc = glGetAttribLocation(prgm->getHandle(), "inColor");
    GLuint col_idx_attrib_loc = glGetAttribLocation(prgm->getHandle(), "inColIdx");

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(vert_attrib_loc);
    glDisableVertexAttribArray(col_attrib_loc);
    glDisableVertexAttribArray(col_idx_attrib_loc);

    return true;
}


bool SphereRenderer::enableShaderData(
    std::shared_ptr<glowl::GLSLProgram> prgm, const MultiParticleDataCall::Particles& parts) {

    // colour
    bool use_global_color = false;
    bool use_tf = false;
    switch (parts.GetColourDataType()) {
    case MultiParticleDataCall::Particles::COLDATA_NONE: {
        glUniform4f(prgm->getUniformLocation("globalCol"), static_cast<float>(parts.GetGlobalColour()[0]) / 255.0f,
            static_cast<float>(parts.GetGlobalColour()[1]) / 255.0f,
            static_cast<float>(parts.GetGlobalColour()[2]) / 255.0f, 1.0f);
        use_global_color = true;
    } break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
    case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I: {
        this->enableTransferFunctionTexture(prgm);
        use_tf = true;
    } break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
    case MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA: {
    } break;
    default: {
        glUniform4f(prgm->getUniformLocation("globalCol"), 0.5f, 0.5f, 0.5f, 1.0f);
        use_global_color = true;
    } break;
    }
    glUniform1i(prgm->getUniformLocation("useGlobalCol"), static_cast<GLint>(use_global_color));
    glUniform1i(prgm->getUniformLocation("useTf"), static_cast<GLint>(use_tf));

    // radius and position
    switch (parts.GetVertexDataType()) {
    case MultiParticleDataCall::Particles::VERTDATA_NONE:
        return false;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
    case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ: {
        glUniform1f(prgm->getUniformLocation("constRad"), parts.GetGlobalRadius());
    } break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
        glUniform1f(prgm->getUniformLocation("constRad"), -1.0f);
        break;
    case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
        glUniform1f(prgm->getUniformLocation("constRad"), parts.GetGlobalRadius());
    default:
        return false;
    }

    return true;
}


bool SphereRenderer::disableShaderData(void) {

    return this->disableTransferFunctionTexture();
}


bool SphereRenderer::enableTransferFunctionTexture(std::shared_ptr<glowl::GLSLProgram> prgm) {

    core_gl::view::CallGetTransferFunctionGL* cgtf = this->get_tf_slot_.CallAs<core_gl::view::CallGetTransferFunctionGL>();
    if ((cgtf != nullptr) && (*cgtf)(0)) {
        // TODO: how to do with unique?
        //cgtf->BindConvenience(prgm, GL_TEXTURE0, 0);
    } else {
        glEnable(GL_TEXTURE_1D);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_1D, this->grey_tf_);
        glUniform1i(prgm->getUniformLocation("tfTexture"), 0);
        glUniform2fv(prgm->getUniformLocation("tfRange"), 1, static_cast<GLfloat*>(this->range_.data()));
    }
    return true;
}


bool SphereRenderer::disableTransferFunctionTexture(void) {

    core_gl::view::CallGetTransferFunctionGL* cgtf = this->get_tf_slot_.CallAs<core_gl::view::CallGetTransferFunctionGL>();
    if (cgtf != nullptr) {
        cgtf->UnbindConvenience();
    } else {
        glBindTexture(GL_TEXTURE_1D, 0);
        glDisable(GL_TEXTURE_1D);
    }
    return true;
}


bool SphereRenderer::enableFlagStorage(const std::shared_ptr<glowl::GLSLProgram> prgm, MultiParticleDataCall* mpdc) {

    if (!this->flags_available_)
        return false;
    if (mpdc == nullptr)
        return false;

    this->flags_enabled_ = false;

    auto flagc = this->read_flags_slot_.CallAs<core_gl::FlagCallRead_GL>();
    if (flagc == nullptr)
        return false;

    if ((*flagc)(core_gl::FlagCallRead_GL::CallGetData)) {
        if (flagc->hasUpdate()) {
            uint32_t parts_count = 0;
            uint32_t part_list_count = static_cast<uint32_t>(mpdc->GetParticleListCount());
            for (uint32_t i = 0; i < part_list_count; i++) {
                MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);
                parts_count += static_cast<uint32_t>(parts.GetCount());
            }
            flagc->getData()->validateFlagCount(parts_count);
        }
    }
    flagc->getData()->flags->bindBase(GL_SHADER_STORAGE_BUFFER, ssbo_flags_binding_point);
    this->flags_enabled_ = true;

    return true;
}


bool SphereRenderer::disableFlagStorage(const std::shared_ptr<glowl::GLSLProgram> prgm) {

    if (!this->flags_available_)
        return false;

    if (this->flags_enabled_) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    }
    this->flags_enabled_ = false;
    return true;
}


bool SphereRenderer::makeColorString(const MultiParticleDataCall::Particles& parts, std::string& out_code,
    std::string& out_declaration, bool interleaved) {

    bool ret = true;

    switch (parts.GetColourDataType()) {
    case MultiParticleDataCall::Particles::COLDATA_NONE:
        out_declaration = "";
        out_code = "    inColor = globalCol;\n";
        break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[SphereRenderer] Cannot pack an unaligned RGB color into an SSBO! Giving up.");
        ret = false;
        break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
        out_declaration = "    uint color;\n";
        if (interleaved) {
            out_code =
                "    inColor = unpackUnorm4x8(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE "+ instanceOffset].color);\n";
        } else {
            out_code = "    inColor = unpackUnorm4x8(theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE
                      "+ instanceOffset].color);\n";
        }
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
        out_declaration = "    float r; float g; float b;\n";
        if (interleaved) {
            out_code =
                "    inColor = vec4(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].r,\n"
                "                       theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].g,\n"
                "                       theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].b, 1.0); \n";
        } else {
            out_code =
                "    inColor = vec4(theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].r,\n"
                "                       theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].g,\n"
                "                       theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].b, 1.0); \n";
        }
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
        out_declaration = "    float r; float g; float b; float a;\n";
        if (interleaved) {
            out_code = "    inColor = vec4(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].r,\n"
                      "                       theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].g,\n"
                      "                       theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].b,\n"
                      "                       theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].a); \n";
        } else {
            out_code = "    inColor = vec4(theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].r,\n"
                      "                       theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].g,\n"
                      "                       theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].b,\n"
                      "                       theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].a); \n";
        }
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
        out_declaration = "    float colorIndex;\n";
        if (interleaved) {
            out_code = "    inColIdx = theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].colorIndex; \n";
        } else {
            out_code = "    inColIdx = theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].colorIndex; \n";
        }
    } break;
    case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I: {
        out_declaration = "    double colorIndex;\n";
        if (interleaved) {
            out_code =
                "    inColIdx = float(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].colorIndex); \n";
        } else {
            out_code = "    inColIdx = float(theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE
                      " + instanceOffset].colorIndex); \n";
        }
    } break;
    case MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA: {
        out_declaration = "    uint col1; uint col2;\n";
        if (interleaved) {
            out_code = "    inColor.xy = unpackUnorm2x16(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE
                      "+ instanceOffset].col1);\n"
                      "    inColor.zw = unpackUnorm2x16(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE
                      "+ instanceOffset].col2);\n";
        } else {
            out_code = "    inColor.xy = unpackUnorm2x16(theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE
                      "+ instanceOffset].col1);\n"
                      "    inColor.zw = unpackUnorm2x16(theColBuffer[" SSBO_GENERATED_SHADER_INSTANCE
                      "+ instanceOffset].col2);\n";
        }
    } break;
    default:
        out_declaration = "";
        out_code = "    inColor = globalCol;\n";
        break;
    }
    // out_code = "    inColor = vec4(0.2, 0.7, 1.0, 1.0);";

    return ret;
}


bool SphereRenderer::makeVertexString(const MultiParticleDataCall::Particles& parts, std::string& out_code,
    std::string& out_declaration, bool interleaved) {

    bool ret = true;

    switch (parts.GetVertexDataType()) {
    case MultiParticleDataCall::Particles::VERTDATA_NONE:
        out_declaration = "";
        out_code = "";
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        out_declaration = "    float posX; float posY; float posZ;\n";
        if (interleaved) {
            out_code = "    inPosition = vec4(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posX,\n"
                      "                 theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posY,\n"
                      "                 theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posZ, 1.0); \n"
                      "    rad = constRad;";
        } else {
            out_code =
                "    inPosition = vec4(thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posX,\n"
                "                 thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posY,\n"
                "                 thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posZ, 1.0); \n"
                "    rad = constRad;";
        }
        break;
    case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
        out_declaration = "    uint posX1; uint posX2; uint posY1; uint posY2; uint posZ1; uint posZ2;\n";
        if (interleaved) {
            out_code = "    uvec2 thex = uvec2(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posX1, "
                      "theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posX2);\n"
                      "    uvec2 they = uvec2(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posY1, "
                      "theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posY2);\n"
                      "    uvec2 thez = uvec2(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posZ1, "
                      "theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posZ2);\n"
                      "    inPosition = inPosition = vec4(float(packDouble2x32(thex)), float(packDouble2x32(they)), "
                      "float(packDouble2x32(thez)), 1.0);\n"
                      "    rad = constRad;";
        } else {
            out_code = "    uvec2 thex = uvec2(thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posX1, "
                      "thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posX2);\n"
                      "    uvec2 they = uvec2(thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posY1, "
                      "thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posY2);\n"
                      "    uvec2 thez = uvec2(thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posZ1, "
                      "thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posZ2);\n"
                      "    inPosition = inPosition = vec4(float(packDouble2x32(thex)), float(packDouble2x32(they)), "
                      "float(packDouble2x32(thez)), 1.0);\n"
                      "    rad = constRad;";
        }
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
        out_declaration = "    float posX; float posY; float posZ; float posR;\n";
        if (interleaved) {
            out_code = "    inPosition = vec4(theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posX,\n"
                      "                 theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posY,\n"
                      "                 theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posZ, 1.0); \n"
                      "    rad = theBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posR;";
        } else {
            out_code =
                "    inPosition = vec4(thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posX,\n"
                "                 thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posY,\n"
                "                 thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posZ, 1.0); \n"
                "    rad = thePosBuffer[" SSBO_GENERATED_SHADER_INSTANCE " + instanceOffset].posR;";
        }
        break;
    default:
        out_declaration = "";
        out_code = "";
        break;
    }

    return ret;
}


std::shared_ptr<glowl::GLSLProgram> SphereRenderer::makeShader(
    const std::string& prgm_name) {

    std::shared_ptr<glowl::GLSLProgram> sh;

    try {
        std::string vert_path = "sphere_renderer/" + prgm_name + ".vert.glsl";
        std::string frag_path = "sphere_renderer/" + prgm_name + ".frag.glsl";

        // should be safe to use shader_options_flags_ since only ssbo and splat use call makeShader
        // and both use shader_options_flags_
        sh = core::utility::make_glowl_shader(prgm_name, *shader_options_flags_, vert_path, frag_path);
    }
    catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, ("SphereRenderer: " + std::string(e.what())).c_str());
    }


    return sh;
}


std::shared_ptr<glowl::GLSLProgram> SphereRenderer::generateShader(
    const MultiParticleDataCall::Particles& parts, const std::string& prgm_name) {

    int c = parts.GetColourDataType();
    int p = parts.GetVertexDataType();

    unsigned int col_bytes, vert_bytes, col_stride, vert_stride;
    bool interleaved;
    this->getBytesAndStride(parts, col_bytes, vert_bytes, col_stride, vert_stride, interleaved);

    shader_map::iterator i = this->the_shaders_.find(std::make_tuple(c, p, interleaved));
    if (i == this->the_shaders_.end()) {
        std::string vert_code, col_code, vert_decl, col_decl, decl;
        makeVertexString(parts, vert_code, vert_decl, interleaved);
        makeColorString(parts, col_code, col_decl, interleaved);

        if (interleaved) {

            decl = "\nstruct SphereParams {\n";

            if (parts.GetColourData() < parts.GetVertexData()) {
                decl += col_decl;
                decl += vert_decl;
            } else {
                decl += vert_decl;
                decl += col_decl;
            }
            decl += "};\n";

            decl += "layout(" SSBO_GENERATED_SHADER_ALIGNMENT ", binding = " + std::to_string(ssbo_vertex_binding_point) +
                    ") buffer shader_data {\n"
                    "    SphereParams theBuffer[];\n"
                    "};\n";

        } else {
            // we seem to have separate buffers for vertex and color data

            decl = "\nstruct SpherePosParams {\n" + vert_decl;
            decl += "};\n";

            decl += "\nstruct SphereColParams {\n" + col_decl;
            decl += "};\n";

            decl += "layout(" SSBO_GENERATED_SHADER_ALIGNMENT ", binding = " + std::to_string(ssbo_vertex_binding_point) +
                    ") buffer shader_data {\n"
                    "    SpherePosParams thePosBuffer[];\n"
                    "};\n";
            decl += "layout(" SSBO_GENERATED_SHADER_ALIGNMENT ", binding = " + std::to_string(ssbo_color_binding_point) +
                    ") buffer shader_data2 {\n"
                    "    SphereColParams theColBuffer[];\n"
                    "};\n";
        }

        std::string code = "\n";
        code += col_code;
        code += vert_code;

        generateShaderFile(prgm_name + "_declaration_snippet", decl);
        generateShaderFile(prgm_name + "_code_snippet", code);

        this->the_shaders_.emplace(std::make_pair(std::make_tuple(c, p, interleaved), makeShader(prgm_name)));
        i = this->the_shaders_.find(std::make_tuple(c, p, interleaved));
    }

    return i->second;
}


void SphereRenderer::getBytesAndStride(const MultiParticleDataCall::Particles& parts, unsigned int& out_col_bytes,
    unsigned int& out_vert_bytes, unsigned int& out_col_stride, unsigned int& out_vert_stride, bool& out_interleaved) {

    out_vert_bytes = MultiParticleDataCall::Particles::VertexDataSize[parts.GetVertexDataType()];
    out_col_bytes = MultiParticleDataCall::Particles::ColorDataSize[parts.GetColourDataType()];

    out_col_stride = parts.GetColourDataStride();
    out_col_stride = out_col_stride < out_col_bytes ? out_col_bytes : out_col_stride;
    out_vert_stride = parts.GetVertexDataStride();
    out_vert_stride = out_vert_stride < out_vert_bytes ? out_vert_bytes : out_vert_stride;

    out_interleaved = (std::abs(reinterpret_cast<const ptrdiff_t>(parts.GetColourData()) -
                               reinterpret_cast<const ptrdiff_t>(parts.GetVertexData())) <= out_vert_stride &&
                         out_vert_stride == out_col_stride) ||
                     out_col_stride == 0;
}


void SphereRenderer::getGLSLVersion(int& out_major, int& out_minor) const {

    out_major = -1;
    out_minor = -1;
    std::string glslVerStr((char*)glGetString(GL_SHADING_LANGUAGE_VERSION));
    std::size_t found = glslVerStr.find(".");
    if (found != std::string::npos) {
        out_major = std::atoi(glslVerStr.substr(0, 1).c_str());
        out_minor = std::atoi(glslVerStr.substr(found + 1, 1).c_str());
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_WARN,
            "[SphereRenderer] No valid GL_SHADING_LANGUAGE_VERSION string found: %s", glslVerStr.c_str());
    }
}


#if defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)

void SphereRenderer::lockSingle(GLsync& out_sync_obj) {

    if (out_sync_obj) {
        glDeleteSync(out_sync_obj);
    }
    out_sync_obj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}


void SphereRenderer::waitSingle(const GLsync& sync_obj) {

    if (sync_obj) {
        while (1) {
            GLenum wait = glClientWaitSync(sync_obj, GL_SYNC_FLUSH_COMMANDS_BIT, 1);
            if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
                return;
            }
        }
    }
}

#endif // defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)


// ##### Ambient Occlusion ################################################# //

bool SphereRenderer::rebuildGBuffer() {

    if (!this->trigger_rebuild_g_buffer_ && (this->cur_vp_width_ == this->last_vp_width_) &&
        (this->cur_vp_height_ == this->last_vp_height_) && !this->use_hp_textures_slot_.IsDirty()) {
        return true;
    }

    this->use_hp_textures_slot_.ResetDirty();

    bool high_precision = this->use_hp_textures_slot_.Param<param::BoolParam>()->Value();

    glBindTexture(GL_TEXTURE_2D, this->g_buffer_.color);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, this->cur_vp_width_, this->cur_vp_height_, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    glBindTexture(GL_TEXTURE_2D, this->g_buffer_.normals);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, high_precision ? GL_RGBA32F : GL_RGBA, this->cur_vp_width_, this->cur_vp_height_, 0, GL_RGB,
        GL_UNSIGNED_BYTE, nullptr);

    glBindTexture(GL_TEXTURE_2D, this->g_buffer_.depth);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, this->cur_vp_width_, this->cur_vp_height_, 0, GL_DEPTH_COMPONENT,
        GL_UNSIGNED_BYTE, nullptr);

    glBindTexture(GL_TEXTURE_2D, 0);

    // Configure the framebuffer object
    GLint prev_fbo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);

    glBindFramebuffer(GL_FRAMEBUFFER, this->g_buffer_.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->g_buffer_.color, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, this->g_buffer_.normals, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, this->g_buffer_.depth, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Framebuffer not complete. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);

    this->trigger_rebuild_g_buffer_ = false;

    return true;
}


void SphereRenderer::rebuildWorkingData(core_gl::view::CallRender3DGL& call, MultiParticleDataCall* mpdc,
    const std::shared_ptr<glowl::GLSLProgram> prgm) {

    // Upload new data if neccessary
    if (this->state_invalid_) {
        unsigned int parts_count = mpdc->GetParticleListCount();

        // Add buffers if neccessary
        for (unsigned int i = static_cast<unsigned int>(this->gpu_data_.size()); i < parts_count; i++) {
            gpuParticleDataType data;
            glGenVertexArrays(1, &(data.vertex_array));
            glGenBuffers(1, &(data.vertex_vbo));
            glGenBuffers(1, &(data.color_vbo));
            this->gpu_data_.push_back(data);
        }

        // Remove buffers if neccessary
        while (this->gpu_data_.size() > parts_count) {
            gpuParticleDataType& data = this->gpu_data_.back();
            glDeleteVertexArrays(1, &(data.vertex_array));
            glDeleteBuffers(1, &(data.vertex_vbo));
            glDeleteBuffers(1, &(data.color_vbo));
            this->gpu_data_.pop_back();
        }

        // Reupload buffers
        for (unsigned int i = 0; i < parts_count; i++) {
            MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

            glBindVertexArray(this->gpu_data_[i].vertex_array);
            this->enableBufferData(prgm, parts, this->gpu_data_[i].vertex_vbo, parts.GetVertexData(),
                this->gpu_data_[i].color_vbo, parts.GetColourData(), true);
            glBindVertexArray(0);
            this->disableBufferData(prgm);
        }
    }

    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
    // Check if voxelization is even needed
    if (this->vol_gen_ == nullptr) {
        this->vol_gen_ = new misc::MDAOVolumeGenerator();
        auto so = shader_options;
        this->vol_gen_->SetShaderSourceFactory(&so);
        this->vol_gen_->Init(frontend_resources.get<frontend_resources::OpenGL_Context>());
    }

    // Recreate the volume if neccessary
    bool equal_clip_data = true;
    for (size_t i = 0; i < 4; i++) {
        if (this->old_clip_dat_[i] != this->cur_clip_dat_[i]) {
            equal_clip_data = false;
            break;
        }
    }
    if ((vol_gen_ != nullptr) && (this->state_invalid_ || this->ao_vol_size_slot_.IsDirty() || !equal_clip_data)) {
        this->ao_vol_size_slot_.ResetDirty();

        int vol_size = this->ao_vol_size_slot_.Param<param::IntParam>()->Value();

        vislib::math::Dimension<float, 3> dims = this->cur_clip_box_.GetSize();

        // Calculate the extensions of the volume by using the specified number of voxels for the longest edge
        float longest_edge = this->cur_clip_box_.LongestEdge();
        dims.Scale(static_cast<float>(vol_size) / longest_edge);

        // The X size must be a multiple of 4, so we might have to correct that a little
        dims.SetWidth(ceil(dims.GetWidth() / 4.0f) * 4.0f);
        dims.SetHeight(ceil(dims.GetHeight()));
        dims.SetDepth(ceil(dims.GetDepth()));
        this->amb_cone_constants_[0] = std::min(dims.Width(), std::min(dims.Height(), dims.Depth()));
        this->amb_cone_constants_[1] = ceil(std::log2(static_cast<float>(vol_size))) - 1.0f;

        // Set resolution accordingly
        this->vol_gen_->SetResolution(dims.GetWidth(), dims.GetHeight(), dims.GetDepth());

        // Insert all particle lists
        this->vol_gen_->ClearVolume();
        this->vol_gen_->StartInsertion(this->cur_clip_box_,
            glm::vec4(this->cur_clip_dat_[0], this->cur_clip_dat_[1], this->cur_clip_dat_[2], this->cur_clip_dat_[3]));

        for (unsigned int i = 0; i < this->gpu_data_.size(); i++) {
            float global_radius = 0.0f;
            if (mpdc->AccessParticles(i).GetVertexDataType() != MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR)
                global_radius = mpdc->AccessParticles(i).GetGlobalRadius();

            this->vol_gen_->InsertParticles(static_cast<unsigned int>(mpdc->AccessParticles(i).GetCount()), global_radius,
                this->gpu_data_[i].vertex_array);
        }
        this->vol_gen_->EndInsertion();

        this->vol_gen_->RecreateMipmap();
    }
}


void SphereRenderer::renderDeferredPass(core_gl::view::CallRender3DGL& call) {

    bool enable_lighting = this->enable_lighting_slot_.Param<param::BoolParam>()->Value();
    bool high_precision = this->use_hp_textures_slot_.Param<param::BoolParam>()->Value();

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, this->g_buffer_.depth);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, this->g_buffer_.normals);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->g_buffer_.color);
    if (vol_gen_ != nullptr) {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_3D, this->vol_gen_->GetVolumeTextureHandle());
        glActiveTexture(GL_TEXTURE0);
    }

    this->lighting_prgm_->use();

    ao_dir_ubo_->bind((GLuint)AO_DIR_UBO_BINDING_POINT);

    this->lighting_prgm_->setUniform("inColorTex", static_cast<int>(0));
    this->lighting_prgm_->setUniform("inNormalsTex", static_cast<int>(1));
    this->lighting_prgm_->setUniform("inDepthTex", static_cast<int>(2));
    this->lighting_prgm_->setUniform("inDensityTex", static_cast<int>(3));

    this->lighting_prgm_->setUniform("inWidth", static_cast<float>(this->cur_vp_width_));
    this->lighting_prgm_->setUniform("inHeight", static_cast<float>(this->cur_vp_height_));
    glUniformMatrix4fv(this->lighting_prgm_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_inv_));
    this->lighting_prgm_->setUniform("inUseHighPrecision", high_precision);
    if (enable_lighting) {
        this->lighting_prgm_->setUniform("inObjLightDir", this->cur_light_dir_);
        this->lighting_prgm_->setUniform("inObjCamPos", this->cur_cam_pos_);
    }
    this->lighting_prgm_->setUniform("inAOOffset", this->ao_offset_slot_.Param<param::FloatParam>()->Value());
    this->lighting_prgm_->setUniform("inAOStrength", this->ao_strength_slot_.Param<param::FloatParam>()->Value());
    this->lighting_prgm_->setUniform("inAOConeLength", this->ao_cone_length_slot_.Param<param::FloatParam>()->Value());
    this->lighting_prgm_->setUniform("inAmbVolShortestEdge", this->amb_cone_constants_[0]);
    this->lighting_prgm_->setUniform("inAmbVolMaxLod", this->amb_cone_constants_[1]);
    glm::vec3 cur_clip_box_coords = glm::vec3(this->cur_clip_box_.GetLeftBottomBack().GetX(),
        this->cur_clip_box_.GetLeftBottomBack().GetY(), this->cur_clip_box_.GetLeftBottomBack().GetZ());
    this->lighting_prgm_->setUniform("inBoundsMin", cur_clip_box_coords);
    glm::vec3 cur_clip_box_size = glm::vec3(this->cur_clip_box_.GetSize().GetWidth(),
        this->cur_clip_box_.GetSize().GetHeight(), this->cur_clip_box_.GetSize().GetDepth());
    this->lighting_prgm_->setUniform("inBoundsSize", cur_clip_box_size);

    // Draw screen filling 'quad' (2 triangle, front facing: CCW)
    std::vector<GLfloat> vertices = {-1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f};
    GLuint vert_attrib_loc = glGetAttribLocation(this->lighting_prgm_->getHandle(), "inPosition");
    glEnableVertexAttribArray(vert_attrib_loc);
    glVertexAttribPointer(vert_attrib_loc, 2, GL_FLOAT, GL_TRUE, 0, vertices.data());
    glDrawArrays(GL_TRIANGLES, static_cast<GLint>(0), static_cast<GLsizei>(vertices.size() / 2));
    glDisableVertexAttribArray(vert_attrib_loc);

    glUseProgram(0); // this->lighting_prgm_.Disable();

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_3D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_TEXTURE_3D);
}


void SphereRenderer::generate3ConeDirections(std::vector<glm::vec4>& directions, float apex) {

    directions.clear();

    float edge_length = 2.0f * tan(0.5f * apex);
    float height = sqrt(1.0f - edge_length * edge_length / 12.0f);
    float radius = sqrt(3.0f) / 3.0f * edge_length;

    for (int i = 0; i < 3; i++) {
        float angle = static_cast<float>(i) / 3.0f * 2.0f * static_cast<float>(M_PI);

        glm::vec3 center(cos(angle) * radius, height, sin(angle) * radius);
        center = glm::normalize(center);
        directions.push_back(glm::vec4(center.x, center.y, center.z, edge_length));
    }
}


std::string SphereRenderer::generateDirectionShaderArrayString(
    const std::vector<glm::vec4>& directions, const std::string& directions_name) {

    std::stringstream result;

    std::string upper_dir_name = directions_name;
    std::transform(upper_dir_name.begin(), upper_dir_name.end(), upper_dir_name.begin(), ::toupper);

    result << "\n#define NUM_" << upper_dir_name << " " << directions.size() << std::endl;
    result << "\nconst vec4 " << directions_name << "[NUM_" << upper_dir_name << "] = vec4[NUM_" << upper_dir_name << "]("
           << std::endl;

    for (auto iter = directions.begin(); iter != directions.end(); iter++) {
        result << "\tvec4(" << (*iter)[0] << ", " << (*iter)[1] << ", " << (*iter)[2] << ", " << (*iter)[3] << ")";
        if (iter + 1 != directions.end())
            result << ",";
        result << std::endl;
    }
    result << ");" << std::endl;

    return result.str();
}

bool SphereRenderer::generateShaderFile(const std::string& file_name, const std::string& code) {
    //std::string full_path = "shaders/sphere_renderer/inc/" + file_name + ".inc.glsl";
    std::string full_path = "../../../plugins/moldyn_gl/shaders/sphere_renderer/inc/" + file_name + ".inc.glsl";

    std::ofstream shader_file;
    shader_file.open(full_path);
    shader_file << code;
    shader_file.close();

    return true;
}
