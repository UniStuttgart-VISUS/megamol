/*/*
 * SphereRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 *
 */

#include "SphereRenderer.h"

#include "mmstd/light/DistantLight.h"
#include "mmstd_gl/flags/FlagCallsGL.h"


#include "OpenGL_Context.h"
#include "geometry_calls/VolumetricDataCall.h"


using namespace megamol::core;
using namespace megamol::geocalls;
using namespace megamol::moldyn_gl::rendering;
using namespace vislib_gl::graphics::gl;


//#define CHRONOTIMING

#define SSBO_GENERATED_SHADER_INSTANCE "gl_VertexID" // or "gl_InstanceID"
#define SSBO_GENERATED_SHADER_ALIGNMENT "std430"     // "std430"

// Beware of changing the binding points
// Need to be changed in shaders accordingly
const GLuint ssbo_flags_binding_point = 2;
const GLuint ssbo_vertex_binding_point = 3;
const GLuint ssbo_color_binding_point = 4;

SphereRenderer::SphereRenderer()
        : mmstd_gl::Renderer3DModuleGL()
        , get_data_slot_("getdata", "Connects to the data source")
        , get_tf_slot_("gettransferfunction", "The slot for the transfer function module")
        , get_clip_plane_slot_("getclipplane", "The slot for the clipping plane module")
        , read_flags_slot_("readFlags", "The slot for reading the selection flags")
        , get_lights_slot_("lights", "Lights are retrieved over this slot.")
        , get_voxels_("voxels", "Connects to the voxel generator")
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
        , shading_mode_(ShadingMode::FORWARD)
        , grey_tf_(0)
        , range_()
        , flags_enabled_(false)
        , flags_available_(false)
        , sphere_prgm_()
        , sphere_geometry_prgm_()
        , lighting_prgm_()
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
        , shading_mode_param_("shaderMode", "The shading mode for selected render modes.")
        , radius_scaling_param_("scaling", "Scaling factor for particle radii.")
        , force_time_slot_(
              "forceTime", "Flag to force the time code to the specified value. Set to true when rendering a video.")
        , use_local_bbox_param_("useLocalBbox", "Enforce usage of local bbox for camera setup")
        , select_color_param_("flag storage::selectedColor", "Color for selected spheres in flag storage.")
        , soft_select_color_param_(
              "flag storage::softSelectedColor", "Color for soft selected spheres in flag storage.")
        , alpha_scaling_param_("splat::alphaScaling", "Splat: Scaling factor for particle alpha.")
        , attenuate_subpixel_param_(
              "splat::attenuateSubpixel", "Splat: Attenuate alpha of points that should have subpixel size.")
        , use_static_data_param_(
              "ssbo::staticData", "SSBO: Upload data only once per hash change and keep data static on GPU")
        , outline_width_slot_("outline::width", "Width of the outline in pixels") {

    this->get_data_slot_.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->get_data_slot_.SetNecessity(AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->get_data_slot_);

    this->get_tf_slot_.SetCompatibleCall<mmstd_gl::CallGetTransferFunctionGLDescription>();
    this->get_tf_slot_.SetNecessity(AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->get_tf_slot_);

    this->get_lights_slot_.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->get_lights_slot_.SetNecessity(AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->get_lights_slot_);

    this->get_voxels_.SetCompatibleCall<VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->get_voxels_);

    this->get_clip_plane_slot_.SetCompatibleCall<view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->get_clip_plane_slot_);

    this->read_flags_slot_.SetCompatibleCall<mmstd_gl::FlagCallRead_GLDescription>();
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
    rmp->SetTypePair(RenderMode::OUTLINE, this->getRenderModeString(RenderMode::OUTLINE).c_str());
    this->render_mode_param_ << rmp;
    this->MakeSlotAvailable(&this->render_mode_param_);
    rmp = nullptr;

    // Initialising enum param with all possible modes (needed for configurator)
    // (Removing not available render modes later in create function)
    param::EnumParam* smp = new param::EnumParam(this->shading_mode_);
    smp->SetTypePair(ShadingMode::FORWARD, this->getShadingModeString(ShadingMode::FORWARD).c_str());
    smp->SetTypePair(ShadingMode::DEFERRED, this->getShadingModeString(ShadingMode::DEFERRED).c_str());
    this->shading_mode_param_ << smp;
    this->MakeSlotAvailable(&this->shading_mode_param_);
    smp = nullptr;

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

    this->outline_width_slot_ << (new core::param::FloatParam(2.0f, 0.0f));
    this->MakeSlotAvailable(&this->outline_width_slot_);
}


SphereRenderer::~SphereRenderer() {
    this->Release();
}


bool SphereRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {

    auto cr = &call;
    if (cr == nullptr)
        return false;

    MultiParticleDataCall* c_particle = this->get_data_slot_.CallAs<MultiParticleDataCall>();
    if ((c_particle != nullptr)) {
        c_particle->SetFrameID(
            static_cast<unsigned int>(cr->Time()), this->force_time_slot_.Param<param::BoolParam>()->Value());
        if (!(*c_particle)(1))
            return false;
        cr->SetTimeFramesCount(c_particle->FrameCount());
        auto const plcount = c_particle->GetParticleListCount();
        if (this->use_local_bbox_param_.Param<param::BoolParam>()->Value() && plcount > 0) {
            auto bbox = c_particle->AccessParticles(0).GetBBox();
            auto cbbox = bbox;
            cbbox.Grow(c_particle->AccessParticles(0).GetGlobalRadius());
            for (unsigned pidx = 1; pidx < plcount; ++pidx) {
                auto temp = c_particle->AccessParticles(pidx).GetBBox();
                bbox.Union(temp);
                temp.Grow(c_particle->AccessParticles(pidx).GetGlobalRadius());
                cbbox.Union(temp);
            }
            cr->AccessBoundingBoxes().SetBoundingBox(bbox);
            cr->AccessBoundingBoxes().SetClipBox(cbbox);
        } else {
            cr->AccessBoundingBoxes() = c_particle->AccessBoundingBoxes();
        }

    } else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
    }
    this->cur_clip_box_ = cr->AccessBoundingBoxes().ClipBox();

    return true;
}


bool SphereRenderer::create() {

    ASSERT(IsAvailable());

    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    if (ogl_ctx.isVersionGEQ(1, 4) == 0) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[SphereRenderer] No render mode available. OpenGL version 1.4 or greater is required.");
        return false;
    }
    if (!ogl_ctx.isExtAvailable("GL_ARB_explicit_attrib_location")) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[SphereRenderer] No render mode is available. Extension "
            "GL_ARB_explicit_attrib_location is not available.");
        return false;
    }
    if (!ogl_ctx.isExtAvailable("GL_ARB_conservative_depth")) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[SphereRenderer] No render mode is available. Extension GL_ARB_conservative_depth is not available.");
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
    if (this->isRenderModeAvailable(RenderMode::OUTLINE)) {
        this->render_mode_param_.Param<param::EnumParam>()->SetTypePair(
            RenderMode::OUTLINE, this->getRenderModeString(RenderMode::OUTLINE).c_str());
    }
    this->MakeSlotAvailable(&this->render_mode_param_);

    // Check initial render mode
    if (!this->isRenderModeAvailable(this->render_mode_)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
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

#ifdef MEGAMOL_USE_PROFILING
    perf_manager_ = const_cast<frontend_resources::PerformanceManager*>(
        &frontend_resources.get<frontend_resources::PerformanceManager>());
    frontend_resources::PerformanceManager::basic_timer_config upload_timer, render_timer;
    upload_timer.name = "upload";
    upload_timer.api = frontend_resources::PerformanceManager::query_api::OPENGL;
    render_timer.name = "render";
    render_timer.api = frontend_resources::PerformanceManager::query_api::OPENGL;
    timers_ = perf_manager_->add_timers(this, {upload_timer, render_timer});
#endif

    return true;
}


void SphereRenderer::release() {

#ifdef MEGAMOL_USE_PROFILING
    perf_manager_->remove_timers(timers_);
#endif

    this->resetOpenGLResources();
}


bool SphereRenderer::resetOpenGLResources() {
    if (this->grey_tf_ != 0) {
        glDeleteTextures(1, &this->grey_tf_);
    }
    this->grey_tf_ = 0;

    this->the_single_mapped_mem_ = nullptr;

    this->the_shaders_.clear();

    this->curr_buf_ = 0;
    this->buf_size_ = (32 * 1024 * 1024);
    this->num_buffers_ = 3;

    this->col_type_ = SimpleSphericalParticles::ColourDataType::COLDATA_NONE;
    this->vert_type_ = SimpleSphericalParticles::VertexDataType::VERTDATA_NONE;

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


void SphereRenderer::resetConditionalParameters() {

    this->select_color_param_.Param<param::ColorParam>()->SetGUIVisible(false);
    this->soft_select_color_param_.Param<param::ColorParam>()->SetGUIVisible(false);

    // Set all render mode dependent parameter to GUI invisible
    // SPLAT
    this->alpha_scaling_param_.Param<param::FloatParam>()->SetGUIVisible(false);
    this->attenuate_subpixel_param_.Param<param::BoolParam>()->SetGUIVisible(false);
    // SSBO
    this->use_static_data_param_.Param<param::BoolParam>()->SetGUIVisible(false);
    // Outlining
    this->outline_width_slot_.Param<param::FloatParam>()->SetGUIVisible(false);
}


bool SphereRenderer::createResources() {

    this->resetConditionalParameters();
    this->resetOpenGLResources();

    this->state_invalid_ = true;

    if (!this->isRenderModeAvailable(this->render_mode_)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[SphereRenderer] Render mode: '%s' is not available - falling back to render mode '%s'.",
            (this->getRenderModeString(this->render_mode_)).c_str(),
            (this->getRenderModeString(RenderMode::SIMPLE)).c_str());
        this->render_mode_ = RenderMode::SIMPLE; // Fallback render mode ...
        return false;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
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
    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());
    shader_options_flags_ = std::make_unique<msf::ShaderFactoryOptionsOpenGL>(shader_options);

    std::string flags_shader_snippet;
    if (this->flags_available_) {
        shader_options_flags_->addDefinition("FLAGS_AVAILABLE");
    }

    try {
        switch (this->render_mode_) {

        case (RenderMode::SIMPLE):
        case (RenderMode::SIMPLE_CLUSTERED): {
            if (shading_mode_ == ShadingMode::DEFERRED) {
                shader_options_flags_->addDefinition("DEFERRED_SHADING");
            }

            sphere_prgm_.reset();
            sphere_prgm_ = core::utility::make_glowl_shader("sphere_simple", *shader_options_flags_,
                "moldyn_gl/sphere_renderer/sphere_simple.vert.glsl",
                "moldyn_gl/sphere_renderer/sphere_simple.frag.glsl");

            glBindAttribLocation(this->sphere_prgm_->getHandle(), 0, "inPosition");
            glBindAttribLocation(this->sphere_prgm_->getHandle(), 1, "inColor");
            glBindAttribLocation(this->sphere_prgm_->getHandle(), 2, "inColIdx");
        } break;

        case (RenderMode::GEOMETRY_SHADER): {
            sphere_geometry_prgm_.reset();
            sphere_geometry_prgm_ = core::utility::make_glowl_shader("sphere_geometry", *shader_options_flags_,
                "moldyn_gl/sphere_renderer/sphere_geometry.vert.glsl",
                "moldyn_gl/sphere_renderer/sphere_geometry.geom.glsl",
                "moldyn_gl/sphere_renderer/sphere_geometry.frag.glsl");

            glBindAttribLocation(this->sphere_geometry_prgm_->getHandle(), 0, "inPosition");
            glBindAttribLocation(this->sphere_geometry_prgm_->getHandle(), 1, "inColor");
            glBindAttribLocation(this->sphere_geometry_prgm_->getHandle(), 2, "inColIdx");
        } break;

        case (RenderMode::SSBO_STREAM): {
            if (shading_mode_ == ShadingMode::DEFERRED) {
                shader_options_flags_->addDefinition("DEFERRED_SHADING");
            }

            this->use_static_data_param_.Param<param::BoolParam>()->SetGUIVisible(true);

            glGenVertexArrays(1, &this->vert_array_);
            glBindVertexArray(this->vert_array_);
            glBindVertexArray(0);
        } break;

        case (RenderMode::SPLAT): {
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
        } break;

        case (RenderMode::BUFFER_ARRAY): {
            if (shading_mode_ == ShadingMode::DEFERRED) {
                shader_options_flags_->addDefinition("DEFERRED_SHADING");
            }

            sphere_prgm_.reset();
            sphere_prgm_ = core::utility::make_glowl_shader("sphere_bufferarray", *shader_options_flags_,
                "moldyn_gl/sphere_renderer/sphere_bufferarray.vert.glsl",
                "moldyn_gl/sphere_renderer/sphere_bufferarray.frag.glsl");

            glBindAttribLocation(this->sphere_prgm_->getHandle(), 0, "inPosition");
            glBindAttribLocation(this->sphere_prgm_->getHandle(), 1, "inColor");
            glBindAttribLocation(this->sphere_prgm_->getHandle(), 2, "inColIdx");

            glGenVertexArrays(1, &this->vert_array_);
            glBindVertexArray(this->vert_array_);
            glGenBuffers(1, &this->the_single_buffer_);
            glBindBuffer(GL_ARRAY_BUFFER, this->the_single_buffer_);
            glBufferStorage(
                GL_ARRAY_BUFFER, this->buf_size_ * this->num_buffers_, nullptr, single_buffer_creation_bits_);
            this->the_single_mapped_mem_ = glMapNamedBufferRange(
                this->the_single_buffer_, 0, this->buf_size_ * this->num_buffers_, single_buffer_mapping_bits_);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindVertexArray(0);
        } break;

        case RenderMode::OUTLINE: {
            this->outline_width_slot_.Param<param::FloatParam>()->SetGUIVisible(true);

            // Create the sphere shader
            sphere_prgm_ = core::utility::make_glowl_shader("sphere_outline", *shader_options_flags_,
                "moldyn_gl/sphere_renderer/sphere_outline.vert.glsl",
                "moldyn_gl/sphere_renderer/sphere_outline.frag.glsl");

            glBindAttribLocation(this->sphere_prgm_->getHandle(), 0, "inPosition");
            glBindAttribLocation(this->sphere_prgm_->getHandle(), 1, "inColor");
            glBindAttribLocation(this->sphere_prgm_->getHandle(), 2, "inColIdx");
        } break;

        default:
            return false;
        }
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to compile sphere shader: %s. [%s, %s, line %d]\n", std::string(e.what()).c_str(), __FILE__,
            __FUNCTION__, __LINE__);
        return false;
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
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(warnstr.c_str());
    }

    return (warnstr.empty());
}


bool SphereRenderer::isFlagStorageAvailable() {
    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();

    auto flagc = this->read_flags_slot_.CallAs<mmstd_gl::FlagCallRead_GL>();

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
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(warnstr.c_str());
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
    case (RenderMode::OUTLINE):
        mode = "Outline";
        break;
    default:
        mode = "unknown";
        break;
    }

    return mode;
}

std::string megamol::moldyn_gl::rendering::SphereRenderer::getShadingModeString(ShadingMode sm) {
    std::string mode;

    switch (sm) {
    case (ShadingMode::FORWARD):
        mode = "Forward";
        break;
    case (ShadingMode::DEFERRED):
        mode = "Deferred";
        break;
    default:
        mode = "unknown";
        break;
    }

    return mode;
}


bool SphereRenderer::Render(mmstd_gl::CallRender3DGL& call) {
    // timer.BeginFrame();

    auto cgtf = this->get_tf_slot_.CallAs<mmstd_gl::CallGetTransferFunctionGL>();

    // Get data
    float scaling = 1.0f;
    MultiParticleDataCall* mpdc = this->getData(static_cast<unsigned int>(call.Time()), scaling);
    if (mpdc == nullptr)
        return false;
    // Check if we got a new data set
    const SIZE_T hash = mpdc->DataHash();
    const unsigned int frame_id = mpdc->FrameID();
    this->state_invalid_ = ((hash != this->old_hash_) || (frame_id != this->old_frame_id_));

    // Check for flag storage before render mode because info is needed for resource creation
    isFlagStorageAvailable();

    // Checking for changed render mode
    auto current_render_mode = static_cast<RenderMode>(this->render_mode_param_.Param<param::EnumParam>()->Value());
    auto current_shading_mode = static_cast<ShadingMode>(this->shading_mode_param_.Param<param::EnumParam>()->Value());
    if (this->init_resources_ || (current_render_mode != this->render_mode_) ||
        (current_shading_mode != this->shading_mode_)) {
        this->render_mode_ = current_render_mode;
        this->shading_mode_ = current_shading_mode;
        init_resources_ = false;
        if (!this->createResources()) {
            return false;
        }
    }

    // Update current state variables -----------------------------------------

    // Update data set range_ (only if new data set was loaded, not on frame loading)
    if (hash != this->old_hash_) {                           // or (this->state_invalid_) {
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


bool SphereRenderer::renderSimple(mmstd_gl::CallRender3DGL& call, MultiParticleDataCall* mpdc) {

    this->sphere_prgm_->use();
    this->enableFlagStorage(this->sphere_prgm_, mpdc);

    glUniform4fv(this->sphere_prgm_->getUniformLocation("viewAttr"), 1, glm::value_ptr(this->cur_view_attrib_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camIn"), 1, glm::value_ptr(this->cur_cam_view_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camRight"), 1, glm::value_ptr(this->cur_cam_right_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camUp"), 1, glm::value_ptr(this->cur_cam_up_));
    glUniform1f(this->sphere_prgm_->getUniformLocation("scaling"),
        this->radius_scaling_param_.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphere_prgm_->getUniformLocation("clipDat"), 1, glm::value_ptr(this->cur_clip_dat_));
    glUniform4fv(this->sphere_prgm_->getUniformLocation("clipCol"), 1, glm::value_ptr(this->cur_clip_col_));
    glUniform4fv(this->sphere_prgm_->getUniformLocation("lightDir"), 1, glm::value_ptr(this->cur_light_dir_));
    glUniformMatrix4fv(this->sphere_prgm_->getUniformLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_inv_));
    glUniformMatrix4fv(
        this->sphere_prgm_->getUniformLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_transp_));
    glUniformMatrix4fv(this->sphere_prgm_->getUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_));
    glUniformMatrix4fv(
        this->sphere_prgm_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_inv_));
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

#ifdef MEGAMOL_USE_PROFILING
        perf_manager_->start_timer(timers_[1]);
#endif
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));
#ifdef MEGAMOL_USE_PROFILING
        perf_manager_->stop_timer(timers_[1]);
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


bool SphereRenderer::renderSSBO(mmstd_gl::CallRender3DGL& call, MultiParticleDataCall* mpdc) {

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
                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, ssbo_vertex_binding_point, this->streamer_.GetHandle(),
                        dst_off, dst_len);
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
                    col_a.SetDataWithItems(parts.GetColourData(), col_stride, col_stride, parts.GetCount(),
                        buf_a.GetMaxNumItemsPerChunk());
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
                const GLuint col_size = this->col_streamer_.SetDataWithItems(parts.GetColourData(), col_stride,
                    col_stride, parts.GetCount(), 3, this->streamer_.GetMaxNumItemsPerChunk());
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
                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, ssbo_vertex_binding_point, this->streamer_.GetHandle(),
                        dst_off, dst_len);
                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, ssbo_color_binding_point,
                        this->col_streamer_.GetHandle(), dst_off2, dst_len2);
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


bool SphereRenderer::renderSplat(mmstd_gl::CallRender3DGL& call, MultiParticleDataCall* mpdc) {

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
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
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


bool SphereRenderer::renderBufferArray(mmstd_gl::CallRender3DGL& call, MultiParticleDataCall* mpdc) {

    this->sphere_prgm_->use();
    this->enableFlagStorage(this->sphere_prgm_, mpdc);

    glUniform4fv(this->sphere_prgm_->getUniformLocation("viewAttr"), 1, glm::value_ptr(this->cur_view_attrib_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camIn"), 1, glm::value_ptr(this->cur_cam_view_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camRight"), 1, glm::value_ptr(this->cur_cam_right_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camUp"), 1, glm::value_ptr(this->cur_cam_up_));
    glUniform1f(this->sphere_prgm_->getUniformLocation("scaling"),
        this->radius_scaling_param_.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphere_prgm_->getUniformLocation("clipDat"), 1, glm::value_ptr(this->cur_clip_dat_));
    glUniform4fv(this->sphere_prgm_->getUniformLocation("clipCol"), 1, glm::value_ptr(this->cur_clip_col_));
    glUniform4fv(this->sphere_prgm_->getUniformLocation("lightDir"), 1, glm::value_ptr(this->cur_light_dir_));
    glUniformMatrix4fv(this->sphere_prgm_->getUniformLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_inv_));
    glUniformMatrix4fv(
        this->sphere_prgm_->getUniformLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_transp_));
    glUniformMatrix4fv(this->sphere_prgm_->getUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_));
    glUniformMatrix4fv(
        this->sphere_prgm_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_inv_));
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
                void* mem =
                    static_cast<char*>(this->the_single_mapped_mem_) + num_verts * vert_stride * this->curr_buf_;
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
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
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


bool SphereRenderer::renderGeometryShader(mmstd_gl::CallRender3DGL& call, MultiParticleDataCall* mpdc) {

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
    glUniform4fv(
        this->sphere_geometry_prgm_->getUniformLocation("viewAttr"), 1, glm::value_ptr(this->cur_view_attrib_));
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
    glUniformMatrix4fv(
        this->sphere_geometry_prgm_->getUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_));
    glUniformMatrix4fv(
        this->sphere_geometry_prgm_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_inv_));
    glUniformMatrix4fv(this->sphere_geometry_prgm_->getUniformLocation("MVPtransp"), 1, GL_FALSE,
        glm::value_ptr(this->cur_mvp_transp_));

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

bool SphereRenderer::renderOutline(mmstd_gl::CallRender3DGL& call, MultiParticleDataCall* mpdc) {

    this->sphere_prgm_->use();
    this->enableFlagStorage(this->sphere_prgm_, mpdc);

    glUniform4fv(this->sphere_prgm_->getUniformLocation("viewAttr"), 1, glm::value_ptr(this->cur_view_attrib_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camIn"), 1, glm::value_ptr(this->cur_cam_view_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camRight"), 1, glm::value_ptr(this->cur_cam_right_));
    glUniform3fv(this->sphere_prgm_->getUniformLocation("camUp"), 1, glm::value_ptr(this->cur_cam_up_));
    glUniform1f(this->sphere_prgm_->getUniformLocation("scaling"),
        this->radius_scaling_param_.Param<param::FloatParam>()->Value());
    glUniform4fv(this->sphere_prgm_->getUniformLocation("clipDat"), 1, glm::value_ptr(this->cur_clip_dat_));
    glUniform4fv(this->sphere_prgm_->getUniformLocation("clipCol"), 1, glm::value_ptr(this->cur_clip_col_));
    glUniform4fv(this->sphere_prgm_->getUniformLocation("lightDir"), 1, glm::value_ptr(this->cur_light_dir_));
    glUniformMatrix4fv(this->sphere_prgm_->getUniformLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_inv_));
    glUniformMatrix4fv(
        this->sphere_prgm_->getUniformLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(this->cur_mv_transp_));
    glUniformMatrix4fv(this->sphere_prgm_->getUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_));
    glUniformMatrix4fv(
        this->sphere_prgm_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_inv_));
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


bool SphereRenderer::disableShaderData() {

    return this->disableTransferFunctionTexture();
}


bool SphereRenderer::enableTransferFunctionTexture(std::shared_ptr<glowl::GLSLProgram> prgm) {

    mmstd_gl::CallGetTransferFunctionGL* cgtf = this->get_tf_slot_.CallAs<mmstd_gl::CallGetTransferFunctionGL>();
    if ((cgtf != nullptr) && (*cgtf)(0)) {
        cgtf->BindConvenience(*prgm, GL_TEXTURE0, 0);
    } else {
        glEnable(GL_TEXTURE_1D);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_1D, this->grey_tf_);
        glUniform1i(prgm->getUniformLocation("tfTexture"), 0);
        glUniform2fv(prgm->getUniformLocation("tfRange"), 1, static_cast<GLfloat*>(this->range_.data()));
    }
    return true;
}


bool SphereRenderer::disableTransferFunctionTexture() {

    mmstd_gl::CallGetTransferFunctionGL* cgtf = this->get_tf_slot_.CallAs<mmstd_gl::CallGetTransferFunctionGL>();
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

    auto flagc = this->read_flags_slot_.CallAs<mmstd_gl::FlagCallRead_GL>();
    if (flagc == nullptr)
        return false;

    if ((*flagc)(mmstd_gl::FlagCallRead_GL::CallGetData)) {
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
    std::string& out_declaration, bool interleaved, msf::ShaderFactoryOptionsOpenGL& shader_options) {

    bool ret = true;

    switch (parts.GetColourDataType()) {
    case MultiParticleDataCall::Particles::COLDATA_NONE:
        shader_options.addDefinition("COLDATA_NONE");
        break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[SphereRenderer] Cannot pack an unaligned RGB color into an SSBO! Giving up.");
        ret = false;
        break;
    case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
        shader_options.addDefinition("COLDATA_UINT8_RGBA");
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
        shader_options.addDefinition("COLDATA_FLOAT_RGB");
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
        shader_options.addDefinition("COLDATA_FLOAT_RGBA");
        break;
    case MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
        shader_options.addDefinition("COLDATA_FLOAT_I");
        break;
    case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I:
        shader_options.addDefinition("COLDATA_DOUBLE_I");
        break;
    case MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA:
        shader_options.addDefinition("COLDATA_USHORT_RGBA");
        break;
    default:
        shader_options.addDefinition("COLDATA_DEFAULT");
        break;
    }
    // out_code = "    inColor = vec4(0.2, 0.7, 1.0, 1.0);";

    return ret;
}


bool SphereRenderer::makeVertexString(const MultiParticleDataCall::Particles& parts, std::string& out_code,
    std::string& out_declaration, bool interleaved, msf::ShaderFactoryOptionsOpenGL& shader_options) {

    bool ret = true;

    switch (parts.GetVertexDataType()) {
    case MultiParticleDataCall::Particles::VERTDATA_NONE:
        shader_options.addDefinition("VERTDATA_NONE");
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        shader_options.addDefinition("VERTDATA_FLOAT_XYZ");
        break;
    case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
        shader_options.addDefinition("VERTDATA_DOUBLE_XYZ");
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
        shader_options.addDefinition("VERTDATA_FLOAT_XYZR");
        break;
    default:
        shader_options.addDefinition("VERTDATA_DEFAULT");
        break;
    }

    return ret;
}


std::shared_ptr<glowl::GLSLProgram> SphereRenderer::makeShader(
    const std::string& prgm_name, const msf::ShaderFactoryOptionsOpenGL& shader_options) {

    std::shared_ptr<glowl::GLSLProgram> sh;

    try {
        std::string vert_path = "moldyn_gl/sphere_renderer/" + prgm_name + ".vert.glsl";
        std::string frag_path = "moldyn_gl/sphere_renderer/" + prgm_name + ".frag.glsl";

        sh = core::utility::make_glowl_shader(prgm_name, shader_options, vert_path, frag_path);
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to compile sphere shader: %s. [%s, %s, line %d]\n", std::string(e.what()).c_str(), __FILE__,
            __FUNCTION__, __LINE__);
    }


    return sh;
}


std::shared_ptr<glowl::GLSLProgram> SphereRenderer::generateShader(
    const MultiParticleDataCall::Particles& parts, const std::string& prgm_name) {

    msf::ShaderFactoryOptionsOpenGL ssbo_splat_shader_options = *shader_options_flags_;

    int c = parts.GetColourDataType();
    int p = parts.GetVertexDataType();

    unsigned int col_bytes, vert_bytes, col_stride, vert_stride;
    bool interleaved;
    this->getBytesAndStride(parts, col_bytes, vert_bytes, col_stride, vert_stride, interleaved);

    shader_map::iterator i = this->the_shaders_.find(std::make_tuple(c, p, interleaved));
    if (i == this->the_shaders_.end()) {
        std::string vert_code, col_code, vert_decl, col_decl, decl;
        makeVertexString(parts, vert_code, vert_decl, interleaved, ssbo_splat_shader_options);
        makeColorString(parts, col_code, col_decl, interleaved, ssbo_splat_shader_options);

        ssbo_splat_shader_options.addDefinition("SSBO_GENERATED_SHADER_ALIGNMENT", SSBO_GENERATED_SHADER_ALIGNMENT);
        ssbo_splat_shader_options.addDefinition("SSBO_GENERATED_SHADER_INSTANCE", SSBO_GENERATED_SHADER_INSTANCE);
        ssbo_splat_shader_options.addDefinition("SSBO_VERTEX_BINDING_POINT", std::to_string(ssbo_vertex_binding_point));
        ssbo_splat_shader_options.addDefinition("SSBO_COLOR_BINDING_POINT", std::to_string(ssbo_color_binding_point));

        if (interleaved) {
            ssbo_splat_shader_options.addDefinition("INTERLEAVED");

            if (parts.GetColourData() < parts.GetVertexData()) {
                ssbo_splat_shader_options.addDefinition("COL_LOWER_VERT");
            }
        }

        this->the_shaders_.emplace(
            std::make_pair(std::make_tuple(c, p, interleaved), makeShader(prgm_name, ssbo_splat_shader_options)));
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
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
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
