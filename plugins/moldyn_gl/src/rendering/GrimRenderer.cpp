/*
 * GrimRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "GrimRenderer.h"

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmstd/light/DistantLight.h"

#include <glm/ext.hpp>

#include "OpenGL_Context.h"


using namespace megamol::core;
using namespace megamol::moldyn_gl::rendering;


// #define SPEAK_CELL_USAGE 1
#define SPEAK_VRAM_CACHE_USAGE 1
//#define VRAM_UPLOAD_QUOTA 0
#define VRAM_UPLOAD_QUOTA 25
//#define VRAM_UPLOAD_QUOTA 100

//#define SUPSAMP_LOOP 1
//#define SUPSAMP_LOOPCNT 1
//#define SUPSAMP_LOOPCNT 2
//#define SUPSAMP_LOOPCNT 4
//#define SUPSAMP_LOOPCNT 16
//#define SUPSAMP_LOOPCNT 64


/****************************************************************************/
// CellInfo

GrimRenderer::CellInfo::CellInfo() {

    glGenOcclusionQueriesNV(1, &this->oQuery);
}


GrimRenderer::CellInfo::~CellInfo() {

    glDeleteOcclusionQueriesNV(1, &this->oQuery);
    this->cache.clear();
}

/****************************************************************************/
// GrimRenderer

GrimRenderer::GrimRenderer()
        : mmstd_gl::Renderer3DModuleGL()
        , sphere_shader_()
        , vanilla_sphere_shader_()
        , init_depth_shader_()
        , init_depth_map_shader_()
        , depth_mip_shader_()
        , point_shader_()
        , init_depth_point_shader_()
        , vert_cnt_shader_()
        , vert_cnt_shader_2_()
        , fbo_()
        , get_data_slot_("getdata", "Connects to the data source")
        , get_tf_slot_("gettransferfunction", "Connects to the transfer function module")
        , get_lights_slot_("lights", "Lights are retrieved over this slot.")
        , use_cell_cull_slot_("use_cell_cull", "Flag to activate per cell culling")
        , use_vert_cull_slot_("use_vert_cull", "Flag to activate per vertex culling")
        , speak_cell_perc_slot_("speak_cell_perc", "Flag to activate output of percentage of culled cells")
        , speak_vert_count_slot_("speak_vert_count", "Flag to activate output of number of vertices")
        , deferred_shading_slot_("deferred_shading", "De-/Activates deferred shading with normal generation")
        , grey_tf_(0)
        , cell_dists_()
        , cell_infos_(0)
        , cache_size_(0)
        , cache_size_used_(0)
        , deferred_sphere_shader_()
        , deferred_vanilla_sphere_shader_()
        , deferred_point_shader_()
        , deferred_shader_()
        , inhash_(0) {

    this->get_data_slot_.SetCompatibleCall<moldyn::ParticleGridDataCallDescription>();
    this->MakeSlotAvailable(&this->get_data_slot_);

    this->get_tf_slot_.SetCompatibleCall<mmstd_gl::CallGetTransferFunctionGLDescription>();
    this->MakeSlotAvailable(&this->get_tf_slot_);

    this->get_lights_slot_.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->MakeSlotAvailable(&this->get_lights_slot_);

    this->use_cell_cull_slot_ << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->use_cell_cull_slot_);

    this->use_vert_cull_slot_.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->use_vert_cull_slot_);

    this->speak_cell_perc_slot_ << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->speak_cell_perc_slot_);

    this->speak_vert_count_slot_ << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->speak_vert_count_slot_);

    this->deferred_shading_slot_ << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->deferred_shading_slot_);
    this->deferred_shading_slot_.ForceSetDirty();

    this->cache_size_ = 6 * 1024 * 1024 * 1024; // TODO: Any way to get this better?
    //this->cache_size_ = 256 * 1024 * 1024; // TODO: Any way to get this better?
    //this->cache_size_ = 256 * 1024; // TODO: Any way to get this better?
    //this->cache_size_ = 1; // TODO: Any way to get this better?
}


GrimRenderer::~GrimRenderer() {

    this->Release();
}


bool GrimRenderer::create() {

    ASSERT(IsAvailable());

    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    // TODO: RequiredExtensions for glowl::GLSLProgram and glowl::fbo_
    if (!ogl_ctx.isExtAvailable("GL_NV_occlusion_query") || !ogl_ctx.isExtAvailable("GL_ARB_multitexture") ||
        !ogl_ctx.isExtAvailable("GL_ARB_vertex_buffer_object") ||
        !ogl_ctx.areExtAvailable(vislib_gl::graphics::gl::FramebufferObject::RequiredExtensions()))
        return false;

    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

    try {
        sphere_shader_ = core::utility::make_glowl_shader("sphere_shader", shader_options,
            "moldyn_gl/grim_renderer/sphere.vert.glsl", "moldyn_gl/grim_renderer/sphere.frag.glsl");

        vanilla_sphere_shader_ = core::utility::make_glowl_shader("vanilla_sphere_shader", shader_options,
            "moldyn_gl/grim_renderer/vanilla_sphere.vert.glsl", "moldyn_gl/grim_renderer/vanilla_sphere.frag.glsl");

        init_depth_shader_ = core::utility::make_glowl_shader("init_depth_shader", shader_options,
            "moldyn_gl/grim_renderer/init_depth.vert.glsl", "moldyn_gl/grim_renderer/init_depth.frag.glsl");

        init_depth_map_shader_ = core::utility::make_glowl_shader("init_depth_map_shader", shader_options,
            "moldyn_gl/grim_renderer/init_depth_map.vert.glsl", "moldyn_gl/grim_renderer/init_depth_map.frag.glsl");

        depth_mip_shader_ = core::utility::make_glowl_shader("depth_mip_shader", shader_options,
            "moldyn_gl/grim_renderer/depth_mip.vert.glsl", "moldyn_gl/grim_renderer/depth_mip.frag.glsl");

        point_shader_ = core::utility::make_glowl_shader("point_shader", shader_options,
            "moldyn_gl/grim_renderer/point.vert.glsl", "moldyn_gl/grim_renderer/point.frag.glsl");

        init_depth_point_shader_ = core::utility::make_glowl_shader("init_depth_point_shader", shader_options,
            "moldyn_gl/grim_renderer/init_depth_point.vert.glsl", "moldyn_gl/grim_renderer/init_depth_point.frag.glsl");

        vert_cnt_shader_ = core::utility::make_glowl_shader("vert_cnt_shader", shader_options,
            "moldyn_gl/grim_renderer/vert_cnt.vert.glsl", "moldyn_gl/grim_renderer/vert_cnt.frag.glsl");

        vert_cnt_shader_2_ = core::utility::make_glowl_shader("vert_cnt_shader_2", shader_options,
            "moldyn_gl/grim_renderer/vert_cnt_2.vert.glsl", "moldyn_gl/grim_renderer/vert_cnt_2.frag.glsl");

        deferred_sphere_shader_ = core::utility::make_glowl_shader("deferred_sphere_shader", shader_options,
            "moldyn_gl/grim_renderer/deferred_sphere.vert.glsl", "moldyn_gl/grim_renderer/deferred_sphere.frag.glsl");

        deferred_vanilla_sphere_shader_ = core::utility::make_glowl_shader("deferred_vanilla_sphere_shader",
            shader_options, "moldyn_gl/grim_renderer/deferred_vanilla_sphere.vert.glsl",
            "moldyn_gl/grim_renderer/deferred_vanilla_sphere.frag.glsl");

        deferred_point_shader_ = core::utility::make_glowl_shader("deferred_point_shader", shader_options,
            "moldyn_gl/grim_renderer/deferred_point.vert.glsl", "moldyn_gl/grim_renderer/deferred_point.frag.glsl");

        deferred_shader_ = core::utility::make_glowl_shader("deferred_shader", shader_options,
            "moldyn_gl/grim_renderer/deferred.vert.glsl", "moldyn_gl/grim_renderer/deferred.frag.glsl");

    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to compile grim shader: %s. [%s, %s, line %d]\n", std::string(e.what()).c_str(), __FILE__,
            __FUNCTION__, __LINE__);

        return false;
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

    this->fbo_.Create(1, 1); // year, right.
    this->depthmap_[0].Create(1, 1);
    this->depthmap_[1].Create(1, 1);

    return true;
}


bool GrimRenderer::GetExtents(megamol::mmstd_gl::CallRender3DGL& call) {

    auto cr = &call;
    if (cr == NULL)
        return false;

    moldyn::ParticleGridDataCall* pgdc = this->get_data_slot_.CallAs<moldyn::ParticleGridDataCall>();
    if (pgdc == NULL)
        return false;
    if (!(*pgdc)(1))
        return false;

    cr->SetTimeFramesCount(pgdc->FrameCount());
    cr->AccessBoundingBoxes() = pgdc->AccessBoundingBoxes();

    return true;
}


void GrimRenderer::release() {
    fbo_.Release();
    depthmap_[0].Release();
    depthmap_[1].Release();
    ds_fbo_.Release();
    glDeleteTextures(1, &this->grey_tf_);
    this->cell_dists_.clear();
    this->cell_infos_.clear();
}


void GrimRenderer::set_cam_uniforms(std::shared_ptr<glowl::GLSLProgram> shader, glm::mat4 view_matrix_inv,
    glm::mat4 view_matrix_inv_transp, glm::mat4 mvp_matrix, glm::mat4 mvp_matrix_transp, glm::mat4 mvp_matrix_inv,
    glm::vec4 cam_pos, glm::vec4 curlight_dir) {

    glUniformMatrix4fv(shader->getUniformLocation("mv_inv"), 1, GL_FALSE, glm::value_ptr(view_matrix_inv));
    glUniformMatrix4fv(
        shader->getUniformLocation("mv_inv_transp"), 1, GL_FALSE, glm::value_ptr(view_matrix_inv_transp));
    glUniformMatrix4fv(shader->getUniformLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp_matrix));
    glUniformMatrix4fv(shader->getUniformLocation("mvp_transp"), 1, GL_FALSE, glm::value_ptr(mvp_matrix_transp));
    glUniformMatrix4fv(shader->getUniformLocation("mvp_inv"), 1, GL_FALSE, glm::value_ptr(mvp_matrix_inv));
    glUniform4fv(shader->getUniformLocation("light_dir"), 1, glm::value_ptr(curlight_dir));
    glUniform4fv(shader->getUniformLocation("cam_pos"), 1, glm::value_ptr(cam_pos));
}


bool GrimRenderer::Render(megamol::mmstd_gl::CallRender3DGL& call) {

    auto cr = &call;
    if (cr == NULL)
        return false;

    moldyn::ParticleGridDataCall* pgdc = this->get_data_slot_.CallAs<moldyn::ParticleGridDataCall>();
    if (pgdc == NULL)
        return false;

    static unsigned int tod = 0;
    unsigned int todi = vislib::sys::GetTicksOfDay();
    bool speak = false;
    if ((todi < tod) || (todi > tod + 1000)) {
        speak = true;
        tod = todi;
    }

    bool use_cell_cull = this->use_cell_cull_slot_.Param<param::BoolParam>()->Value();
    bool use_vert_cull = this->use_vert_cull_slot_.Param<param::BoolParam>()->Value();
    bool speak_cell_perc = speak /*&& use_cell_cull*/ && this->speak_cell_perc_slot_.Param<param::BoolParam>()->Value();
    bool speak_vert_count = /*speak && */ this->speak_vert_count_slot_.Param<param::BoolParam>()->Value();
    bool deferred_shading = this->deferred_shading_slot_.Param<param::BoolParam>()->Value();
    auto da_sphere_shader = use_vert_cull ? this->sphere_shader_ : this->vanilla_sphere_shader_;
    auto da_point_shader = this->point_shader_;
    if (deferred_shading) {
        da_sphere_shader = use_vert_cull ? this->deferred_sphere_shader_ : this->deferred_vanilla_sphere_shader_;
        da_point_shader = this->deferred_point_shader_;
    }
    unsigned int cial = glGetAttribLocationARB(da_sphere_shader->getHandle(), "colIdx");
    unsigned int cial2 = glGetAttribLocationARB(da_point_shader->getHandle(), "colIdx");

    // ask for extend to calculate the data scaling
    pgdc->SetFrameID(static_cast<unsigned int>(cr->Time()));
    if (!(*pgdc)(1))
        return false;

    const float scaling = 1.0f;

    // fetch real data
    pgdc->SetFrameID(static_cast<unsigned int>(cr->Time()));
    if (!(*pgdc)(0))
        return false;
    if (this->inhash_ != pgdc->DataHash()) {
        this->inhash_ = pgdc->DataHash();
        // invalidate ALL VBOs
        SIZE_T cnt = this->cell_infos_.size();
        for (SIZE_T i = 0; i < cnt; i++) {
            SIZE_T cnt2 = this->cell_infos_[i].cache.size();
            for (SIZE_T j = 0; j < cnt2; j++) {
                glDeleteBuffersARB(2, this->cell_infos_[i].cache[j].data);
                this->cell_infos_[i].cache[j].data[0] = 0;
                this->cell_infos_[i].cache[j].data[1] = 0;
            }
        }
        this->cache_size_used_ = 0;
    }

    unsigned int cell_cnt = pgdc->CellsCount();
    unsigned int type_cnt = pgdc->TypesCount();

    // Camera
    core::view::Camera cam = call.GetCamera();
    auto cam_pose = cam.get<core::view::Camera::Pose>();
    auto cam_intrinsics = cam.get<core::view::Camera::PerspectiveParameters>();
    auto fbo_ = call.GetFramebuffer();

    glm::mat4 view_matrix = cam.getViewMatrix();
    glm::mat4 proj_matrix = cam.getProjectionMatrix();
    glm::mat4 view_matrix_inv = glm::inverse(view_matrix);
    glm::mat4 view_matrix_inv_transp = glm::transpose(view_matrix_inv);
    glm::mat4 mvp_matrix = proj_matrix * view_matrix;
    glm::mat4 mvp_matrix_transp = glm::transpose(mvp_matrix);
    glm::mat4 mvp_matrix_inv = glm::inverse(mvp_matrix);
    glm::vec4 cam_pos = glm::vec4(cam_pose.position, 1.0f);
    glm::vec4 cam_view = glm::vec4(cam_pose.direction, 1.0f);
    glm::vec4 cam_up = glm::vec4(cam_pose.up, 1.0f);
    glm::vec4 cam_right = glm::vec4(glm::cross(cam_pose.direction, cam_pose.up), 1.0f);
    float half_aperture_angle = cam_intrinsics.fovy / 2.0f;

    // Lights
    glm::vec4 curlight_dir = {0.0f, 0.0f, 0.0f, 1.0f};

    auto call_light = get_lights_slot_.CallAs<core::view::light::CallLight>();
    if (call_light != nullptr) {
        if (!(*call_light)(0)) {
            return false;
        }

        auto lights = call_light->getData();
        auto distant_lights = lights.get<core::view::light::DistantLightType>();

        if (distant_lights.size() > 1) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GrimRenderer] Only one single 'Distant Light' source is supported by this renderer");
        } else if (distant_lights.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GrimRenderer] No 'Distant Light' found");
        }

        for (auto const& light : distant_lights) {
            auto use_eyedir = light.eye_direction;
            if (use_eyedir) {
                curlight_dir = cam_view;
            } else {
                auto light_dir = light.direction;
                if (light_dir.size() == 3) {
                    curlight_dir[0] = light_dir[0];
                    curlight_dir[1] = light_dir[1];
                    curlight_dir[2] = light_dir[2];
                }
                if (light_dir.size() == 4) {
                    curlight_dir[3] = light_dir[3];
                }
                /// View Space Lighting. Comment line to change to Object Space Lighting.
                // this->curlight_dir = this->curMVtransp * this->curlight_dir;
            }
            /// TODO Implement missing distant light parameters:
            // light.second.dl_angularDiameter;
            // light.second.lightColor;
            // light.second.lightIntensity;
        }
    }

    // update fbo_ size, if required ///////////////////////////////////////////
    if ((this->fbo_.GetWidth() != fbo_->getWidth()) || (this->fbo_.GetHeight() != fbo_->getHeight()) ||
        this->deferred_shading_slot_.IsDirty()) {
        this->deferred_shading_slot_.ResetDirty();

        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 1, -1, "grim-fbo_-resize");
        this->fbo_.Release();
        this->fbo_.Create(fbo_->getWidth(), fbo_->getHeight(), GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, // colour buffer
            vislib_gl::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE,
            GL_DEPTH_COMPONENT24); // depth buffer

        unsigned int dmw = vislib::math::NextPowerOfTwo(fbo_->getWidth());
        unsigned int dmh = vislib::math::NextPowerOfTwo(fbo_->getHeight());
        dmh += dmh / 2;
        if ((this->depthmap_[0].GetWidth() != dmw) || (this->depthmap_[0].GetHeight() != dmh)) {
            for (int i = 0; i < 2; i++) {
                this->depthmap_[i].Release();
                this->depthmap_[i].Create(dmw, dmh, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT,
                    vislib_gl::graphics::gl::FramebufferObject::ATTACHMENT_DISABLED);
            }
        }

        this->ds_fbo_.Release();
        if (deferred_shading) {
            // attachments:
            //  colour (RGBA-byte)
            //  normal (RGBA-float16; xyz + confidence)
            //  depth (24 bit)
            //  stencil (none)
            vislib_gl::graphics::gl::FramebufferObject::ColourAttachParams cap[3];
            cap[0].format = GL_RGBA;
            cap[0].internalFormat = GL_RGBA8;
            cap[0].type = GL_UNSIGNED_BYTE;
            cap[1].format = GL_RGBA;
            cap[1].internalFormat = GL_RGBA16F;
            cap[1].type = GL_HALF_FLOAT;
            cap[2].format = GL_RGBA;
            cap[2].internalFormat = GL_RGBA32F;
            cap[2].type = GL_FLOAT;
            vislib_gl::graphics::gl::FramebufferObject::DepthAttachParams dap;
            dap.state = vislib_gl::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE;
            dap.format = GL_DEPTH_COMPONENT24;
            vislib_gl::graphics::gl::FramebufferObject::StencilAttachParams sap;
            sap.state = vislib_gl::graphics::gl::FramebufferObject::ATTACHMENT_DISABLED;
            sap.format = GL_STENCIL_INDEX;

            try {
                if (!this->ds_fbo_.Create(fbo_->getWidth(), fbo_->getHeight(), 3, cap, dap, sap)) {
                    throw vislib::Exception("ds_fbo_.Create failed\n", __FILE__, __LINE__);
                }
            } catch (vislib::Exception ex) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("Failed to created ds_fbo_: %s", ex.GetMsgA());
            }
        }
        glPopDebugGroup();
    }

    if (this->cell_dists_.size() != cell_cnt) {
        this->cell_dists_.resize(cell_cnt);
        this->cell_infos_.resize(cell_cnt);
        for (unsigned int i = 0; i < cell_cnt; i++) {
            this->cell_dists_[i].First() = i;
            this->cell_infos_[i].wasvisible = true; // TODO: refine with Reina-Approach (wtf?)
            this->cell_infos_[i].maxrad = 0.0f;
            this->cell_infos_[i].cache.clear();
            this->cell_infos_[i].cache.resize(type_cnt);
            for (unsigned int j = 0; j < type_cnt; j++) {
                this->cell_infos_[i].maxrad = glm::max(
                    this->cell_infos_[i].maxrad, pgdc->Cells()[i].AccessParticleLists()[j].GetMaxRadius() * scaling);
            }
        }
        this->cache_size_used_ = 0;
    }

    // depth-sort of cells ////////////////////////////////////////////////////

    float view_dist = 0.5f * fbo_->getHeight() / tanf(half_aperture_angle);

    std::vector<vislib::Pair<unsigned int, float>>& dists = this->cell_dists_;
    std::vector<CellInfo>& infos = this->cell_infos_;
    // -> The usage of these references is required in order to get performance !!! WTF !!!

    for (unsigned int i = 0; i < cell_cnt; i++) {
        unsigned int idx = dists[i].First();
        const moldyn::ParticleGridDataCall::GridCell& cell = pgdc->Cells()[idx];
        CellInfo& info = infos[idx];
        const vislib::math::Cuboid<float>& bbox = cell.GetBoundingBox();

        glm::vec3 cell_pos((bbox.Left() + bbox.Right()) * 0.5f * scaling, (bbox.Bottom() + bbox.Top()) * 0.5f * scaling,
            (bbox.Back() + bbox.Front()) * 0.5f * scaling);

        glm::vec3 cell_dist_v = cell_pos - glm::vec3(cam_pos);
        float cell_dist = glm::dot(glm::vec3(cam_view), cell_dist_v);

        dists[i].Second() = cell_dist;

        // calculate view size of the max sphere
        float sphere_img_rad = info.maxrad * view_dist / cell_dist;
        info.dots = (sphere_img_rad < 0.75f);

        info.isvisible = true;
        // Testing against the viewing frustum would be nice, but I don't care
    }
    std::sort(dists.begin(), dists.end(), GrimRenderer::depthSort);

    // init depth points //////////////////////////////////////////////////////
#ifdef _WIN32
#pragma region Depthbuffer initialization
#endif // _WIN32

    glLineWidth(5.0f);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    // upload to gpu-cache
    int vram_upload_quota = VRAM_UPLOAD_QUOTA; // upload no more then X VBO per frame

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 2, -1, "grim-init-depth-points");
    // z-buffer-filling
#if defined(DEBUG) || defined(_DEBUG)
    UINT oldlevel = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_NONE);
#endif

    this->fbo_.Enable();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#ifdef SPEAK_CELL_USAGE
    printf("[initd1");
#endif
    this->init_depth_point_shader_->use();
    set_cam_uniforms(this->init_depth_point_shader_, view_matrix_inv, view_matrix_inv_transp, mvp_matrix,
        mvp_matrix_transp, mvp_matrix_inv, cam_pos, curlight_dir);
    glPointSize(1.0f);
    for (int i = cell_cnt - 1; i >= 0; i--) { // front to back
        unsigned int idx = dists[i].First();
        const moldyn::ParticleGridDataCall::GridCell* cell = &pgdc->Cells()[idx];
        CellInfo& info = infos[idx];
        if (!info.wasvisible)
            continue;
        // only draw cells which were visible last frame
        if (!info.dots)
            continue;

#ifdef SPEAK_CELL_USAGE
        printf("-%d", i);
#endif
        for (unsigned int j = 0; j < type_cnt; j++) {
            const moldyn::ParticleGridDataCall::Particles& parts = cell->AccessParticleLists()[j];
            const moldyn::ParticleGridDataCall::ParticleType& ptype = pgdc->Types()[j];
            CellInfo::CacheItem& ci = info.cache[j];
            unsigned int vbpp = 1, cbpp = 1;
            switch (ptype.GetVertexDataType()) {
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                vbpp = 3 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                vbpp = 4 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
                vbpp = 3 * sizeof(short);
                break;
            default:
                continue;
            }
            switch (ptype.GetColourDataType()) {
            case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                cbpp = 3;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                cbpp = 4;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                cbpp = 3 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                cbpp = 4 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
                cbpp = sizeof(float);
                break;
            default:
                break;
            }

            if ((ci.data[0] == 0) && (vram_upload_quota > 0) && (parts.GetCount() > 0) &&
                (((vbpp + cbpp) * parts.GetCount()) < (this->cache_size_ - this->cache_size_used_))) {
                // upload
                glGetError();
                glGenBuffersARB(2, ci.data);
                if (glGetError() != GL_NO_ERROR) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError("glGenBuffersARB failed");
                    throw vislib::Exception("glGenBuffersARB failed", __FILE__, __LINE__);
                }
                vram_upload_quota--;
                glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                glGetError();
                if (parts.GetVertexDataStride() == 0) {
                    GLenum err;
                    glBufferDataARB(GL_ARRAY_BUFFER, vbpp * parts.GetCount(), parts.GetVertexData(), GL_STATIC_DRAW);
                    if ((err = glGetError()) != GL_NO_ERROR) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError("glBufferDataARB failed: %u", err);
                        throw vislib::Exception("glBufferDataARB failed", __FILE__, __LINE__);
                    }
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Currently only data without stride is supported for caching");
                    throw vislib::Exception(
                        "Currently only data without stride is supported for caching", __FILE__, __LINE__);
                }
                this->cache_size_used_ += vbpp * parts.GetCount();
                glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                if (parts.GetColourDataStride() == 0) {
                    GLenum err;
                    glBufferDataARB(GL_ARRAY_BUFFER, cbpp * parts.GetCount(), parts.GetColourData(), GL_STATIC_DRAW);
                    if ((err = glGetError()) != GL_NO_ERROR) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError("glBufferDataARB failed: %u", err);
                        throw vislib::Exception("glBufferDataARB failed", __FILE__, __LINE__);
                    }
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Currently only data without stride is supported for caching");
                    throw vislib::Exception(
                        "Currently only data without stride is supported for caching", __FILE__, __LINE__);
                }
                this->cache_size_used_ += cbpp * parts.GetCount();
#ifdef SPEAK_VRAM_CACHE_USAGE
                printf("VRAM-Cache: Add[%d; %u] %u/%u\n", i, j, this->cache_size_used_, this->cache_size_);
#endif // SPEAK_VRAM_CACHE_USAGE
            }

            // radius and position
            switch (ptype.GetVertexDataType()) {
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
                continue;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                glEnableClientState(GL_VERTEX_ARRAY);
                if (ci.data[0] != 0) {
                    glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                    glVertexPointer(3, GL_FLOAT, 0, NULL);
                } else {
                    glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                }
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                glEnableClientState(GL_VERTEX_ARRAY);
                if (ci.data[0] != 0) {
                    glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                    glVertexPointer(3, GL_FLOAT, 16, NULL);
                } else {
                    glVertexPointer(3, GL_FLOAT, glm::max(16U, parts.GetVertexDataStride()), parts.GetVertexData());
                }
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ: {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "GrimRenderer: vertices with short coords are deprecated!");
            } break;

            default:
                continue;
            }
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
            glDisableClientState(GL_VERTEX_ARRAY);
        }
    }
#ifdef SPEAK_CELL_USAGE
    printf("]\n");
#endif
    glUseProgram(0); //this->init_depth_point_shader_.Disable();
    glPopDebugGroup();

    // init depth disks ///////////////////////////////////////////////////////
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 3, -1, "grim-init-depth-disks");

    float viewport_stuff[4] = {0.0f, 0.0f, static_cast<float>(fbo_->getWidth()), static_cast<float>(fbo_->getHeight())};
    float default_point_size = glm::max(viewport_stuff[2], viewport_stuff[3]);
    if (viewport_stuff[2] < 1.0f)
        viewport_stuff[2] = 1.0f;
    if (viewport_stuff[3] < 1.0f)
        viewport_stuff[3] = 1.0f;
    viewport_stuff[2] = 2.0f / viewport_stuff[2];
    viewport_stuff[3] = 2.0f / viewport_stuff[3];

    glPointSize(default_point_size);

    this->init_depth_shader_->use();
    set_cam_uniforms(this->init_depth_shader_, view_matrix_inv, view_matrix_inv_transp, mvp_matrix, mvp_matrix_transp,
        mvp_matrix_inv, cam_pos, curlight_dir);

    glUniform4fv(this->init_depth_shader_->getUniformLocation("viewAttr"), 1, viewport_stuff);
    glUniform3fv(this->init_depth_shader_->getUniformLocation("camIn"), 1, glm::value_ptr(cam_view));
    glUniform3fv(this->init_depth_shader_->getUniformLocation("camRight"), 1, glm::value_ptr(cam_right));
    glUniform3fv(this->init_depth_shader_->getUniformLocation("camUp"), 1, glm::value_ptr(cam_up));

    // no clipping plane for now
    glColor4ub(192, 192, 192, 255);
    glDisableClientState(GL_COLOR_ARRAY);

#ifdef SPEAK_CELL_USAGE
    printf("[initd2");
#endif
    for (int i = cell_cnt - 1; i >= 0; i--) { // front to back
        unsigned int idx = dists[i].First();
        const moldyn::ParticleGridDataCall::GridCell* cell = &pgdc->Cells()[idx];
        CellInfo& info = infos[idx];
        if (!info.wasvisible)
            continue;
        // only draw cells which were visible last frame
        if (info.dots)
            continue;

        //glColor4ub(192, 192, 192, 255);
        float a = static_cast<float>(i) / static_cast<float>(cell_cnt - 1);
        ASSERT((a >= 0.0) && (a <= 1.0f));
        glColor3f(1.0f - a, 0.0f, a);
        if (info.dots) {
            glColor3ub(255, 0, 0);
        }

#ifdef SPEAK_CELL_USAGE
        printf("-%d", i);
#endif
        for (unsigned int j = 0; j < type_cnt; j++) {
            const moldyn::ParticleGridDataCall::Particles& parts = cell->AccessParticleLists()[j];
            const moldyn::ParticleGridDataCall::ParticleType& ptype = pgdc->Types()[j];
            CellInfo::CacheItem& ci = info.cache[j];
            unsigned int vbpp = 1, cbpp = 1;
            switch (ptype.GetVertexDataType()) {
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                vbpp = 3 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                vbpp = 4 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
                vbpp = 3 * sizeof(short);
                break;
            default:
                continue;
            }
            switch (ptype.GetColourDataType()) {
            case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                cbpp = 3;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                cbpp = 4;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                cbpp = 3 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                cbpp = 4 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
                cbpp = sizeof(float);
                break;
            default:
                break;
            }

            if ((ci.data[0] == 0) && (vram_upload_quota > 0) && (parts.GetCount() > 0) &&
                (((vbpp + cbpp) * parts.GetCount()) < (this->cache_size_ - this->cache_size_used_))) {
                // upload
                glGetError();
                glGenBuffersARB(2, ci.data);
                if (glGetError() != GL_NO_ERROR) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError("glGenBuffersARB failed");
                    throw vislib::Exception("glGenBuffersARB failed", __FILE__, __LINE__);
                }
                vram_upload_quota--;
                glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                glGetError();
                if (parts.GetVertexDataStride() == 0) {
                    GLenum err;
                    glBufferDataARB(GL_ARRAY_BUFFER, vbpp * parts.GetCount(), parts.GetVertexData(), GL_STATIC_DRAW);
                    if ((err = glGetError()) != GL_NO_ERROR) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError("glBufferDataARB failed: %u", err);
                        throw vislib::Exception("glBufferDataARB failed", __FILE__, __LINE__);
                    }
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Currently only data without stride is supported for caching");
                    throw vislib::Exception(
                        "Currently only data without stride is supported for caching", __FILE__, __LINE__);
                }
                this->cache_size_used_ += vbpp * parts.GetCount();
                glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                if (parts.GetColourDataStride() == 0) {
                    GLenum err;
                    glBufferDataARB(GL_ARRAY_BUFFER, cbpp * parts.GetCount(), parts.GetColourData(), GL_STATIC_DRAW);
                    if ((err = glGetError()) != GL_NO_ERROR) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError("glBufferDataARB failed: %u", err);
                        throw vislib::Exception("glBufferDataARB failed", __FILE__, __LINE__);
                    }
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Currently only data without stride is supported for caching");
                    throw vislib::Exception(
                        "Currently only data without stride is supported for caching", __FILE__, __LINE__);
                }
                this->cache_size_used_ += cbpp * parts.GetCount();
#ifdef SPEAK_VRAM_CACHE_USAGE
                printf("VRAM-Cache: Add[%d; %u] %u/%u\n", i, j, this->cache_size_used_, this->cache_size_);
#endif // SPEAK_VRAM_CACHE_USAGE
            }

            // radius and position
            switch (ptype.GetVertexDataType()) {
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
                continue;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                glEnableClientState(GL_VERTEX_ARRAY);
                glUniform4f(this->init_depth_shader_->getUniformLocation("inConsts1"), ptype.GetGlobalRadius(), 0.0f,
                    0.0f, 0.0f);
                if (ci.data[0] != 0) {
                    glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                    glVertexPointer(3, GL_FLOAT, 0, NULL);
                } else {
                    glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                }
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                glEnableClientState(GL_VERTEX_ARRAY);
                glUniform4f(this->init_depth_shader_->getUniformLocation("inConsts1"), -1.0f, 0.0f, 0.0f, 0.0f);
                if (ci.data[0] != 0) {
                    glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                    glVertexPointer(4, GL_FLOAT, 0, NULL);
                } else {
                    glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                }
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ: {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "GrimRenderer: vertices with short coords are deprecated!");
            } break;

            default:
                continue;
            }
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
            glDisableClientState(GL_VERTEX_ARRAY);
        }

        //glBegin(GL_LINES);
        //glVertex3f(cell->GetBoundingBox().Left(), cell->GetBoundingBox().Bottom(), cell->GetBoundingBox().Back());
        //glVertex3f(cell->GetBoundingBox().Right(), cell->GetBoundingBox().Top(), cell->GetBoundingBox().Front());
        //glVertex3f(cell->GetBoundingBox().Right(), cell->GetBoundingBox().Bottom(), cell->GetBoundingBox().Back());
        //glVertex3f(cell->GetBoundingBox().Left(), cell->GetBoundingBox().Top(), cell->GetBoundingBox().Front());
        //glVertex3f(cell->GetBoundingBox().Left(), cell->GetBoundingBox().Top(), cell->GetBoundingBox().Back());
        //glVertex3f(cell->GetBoundingBox().Right(), cell->GetBoundingBox().Bottom(), cell->GetBoundingBox().Front());
        //glVertex3f(cell->GetBoundingBox().Right(), cell->GetBoundingBox().Top(), cell->GetBoundingBox().Back());
        //glVertex3f(cell->GetBoundingBox().Left(), cell->GetBoundingBox().Bottom(), cell->GetBoundingBox().Front());
        //glEnd();
    }
#ifdef SPEAK_CELL_USAGE
    printf("]\n");
#endif

    glUseProgram(0); // this->init_depth_shader_.Disable();
    glPopDebugGroup();

#ifdef _WIN32
#pragma endregion Depthbuffer initialization
#endif // _WIN32

    // issue queries //////////////////////////////////////////////////////
#ifdef _WIN32
#pragma region issue occlusion queries for all cells to find hidden ones
#endif // _WIN32

    if (use_cell_cull) {
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 4, -1, "grim-issue-queries");

        // occlusion queries ftw
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glDepthMask(GL_FALSE);
        glDisable(GL_CULL_FACE);

        // this shader is so simple it should also work for the boxes.
        this->init_depth_point_shader_->use();
        set_cam_uniforms(this->init_depth_point_shader_, view_matrix_inv, view_matrix_inv_transp, mvp_matrix,
            mvp_matrix_transp, mvp_matrix_inv, cam_pos, curlight_dir);

        // also disable texturing and any fancy shading features
        for (int i = cell_cnt - 1; i >= 0; i--) { // front to back
            const moldyn::ParticleGridDataCall::GridCell& cell = pgdc->Cells()[i];
            CellInfo& info = infos[i];
            const vislib::math::Cuboid<float>& bbox = cell.GetBoundingBox();
            if (!info.isvisible)
                continue; // frustum culling

            glBeginOcclusionQueryNV(info.oQuery);

            // render bounding box for cell idx
            glBegin(GL_QUADS);

            glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Top(), bbox.Back());
            glVertex3f(bbox.Left(), bbox.Top(), bbox.Back());

            glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Front());
            glVertex3f(bbox.Left(), bbox.Top(), bbox.Front());
            glVertex3f(bbox.Right(), bbox.Top(), bbox.Front());
            glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Front());

            glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Front());
            glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Front());

            glVertex3f(bbox.Left(), bbox.Top(), bbox.Front());
            glVertex3f(bbox.Left(), bbox.Top(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Top(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Top(), bbox.Front());

            glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Back());
            glVertex3f(bbox.Left(), bbox.Top(), bbox.Back());
            glVertex3f(bbox.Left(), bbox.Top(), bbox.Front());
            glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Front());

            glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Front());
            glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Top(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Top(), bbox.Front());

            glEnd();

            glEndOcclusionQueryNV();
        }

        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);
        // reenable other state
        glPopDebugGroup();
    }
#ifdef _WIN32
#pragma endregion issue occlusion queries
#endif // _WIN32

    this->fbo_.Disable();
    // END Depth buffer initialized

    glEnable(GL_CULL_FACE);

    // depth mipmap ///////////////////////////////////////////////////////////
#ifdef _WIN32
#pragma region depth buffer mipmaping
#endif // _WIN32

    int maxLevel = 0;
    if (use_vert_cull) {
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 5, -1, "grim-depth-mipmap");
        // create depth mipmap
        this->depthmap_[0].Enable();

        //glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glEnable(GL_TEXTURE_2D);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        glActiveTextureARB(GL_TEXTURE0_ARB);
        this->fbo_.BindDepthTexture();

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        this->init_depth_map_shader_->use();
        set_cam_uniforms(this->init_depth_map_shader_, view_matrix_inv, view_matrix_inv_transp, mvp_matrix,
            mvp_matrix_transp, mvp_matrix_inv, cam_pos, curlight_dir);
        this->init_depth_map_shader_->setUniform("datex", 0);

        glBegin(GL_QUADS);
        float xf = float(this->fbo_.GetWidth()) / float(this->depthmap_[0].GetWidth());
        float yf = float(this->fbo_.GetHeight()) / float(this->depthmap_[0].GetHeight());
        glVertex2f(-1.0f, -1.0f);
        glVertex2f(-1.0f + 2.0f * xf, -1.0f);
        glVertex2f(-1.0f + 2.0f * xf, -1.0f + 2.0f * yf);
        glVertex2f(-1.0f, -1.0f + 2.0f * yf);
        glEnd();

        glUseProgram(0); // this->init_depth_map_shader_.Disable();

        int lw = this->depthmap_[0].GetWidth() / 2;
        int ly = this->depthmap_[0].GetHeight() * 2 / 3;
        int lh = ly / 2;
        int ls = vislib::math::Min(lh, lw);

        this->depth_mip_shader_->use();
        set_cam_uniforms(this->depth_mip_shader_, view_matrix_inv, view_matrix_inv_transp, mvp_matrix,
            mvp_matrix_transp, mvp_matrix_inv, cam_pos, curlight_dir);
        this->depth_mip_shader_->setUniform("datex", 0);
        this->depth_mip_shader_->setUniform("src", 0, 0);
        this->depth_mip_shader_->setUniform("dst", 0, ly);

        maxLevel = 1; // we created one! hui!
        glBegin(GL_QUADS);
        glVertex2f(-1.0f + 2.0f * 0.0f, -1.0f + 2.0f * float(ly) / float(this->depthmap_[0].GetHeight()));
        glVertex2f(-1.0f + 2.0f * float(this->fbo_.GetWidth() / 2) / float(this->depthmap_[0].GetWidth()),
            -1.0f + 2.0f * float(ly) / float(this->depthmap_[0].GetHeight()));
        glVertex2f(-1.0f + 2.0f * float(this->fbo_.GetWidth() / 2) / float(this->depthmap_[0].GetWidth()),
            -1.0f + 2.0f * float(ly + this->fbo_.GetHeight() / 2) / float(this->depthmap_[0].GetHeight()));
        glVertex2f(-1.0f + 2.0f * 0.0f,
            -1.0f + 2.0f * float(ly + this->fbo_.GetHeight() / 2) / float(this->depthmap_[0].GetHeight()));
        glEnd();

        this->depthmap_[0].Disable();

        int lx = lw;
        while (ls > 1) {
            this->depthmap_[maxLevel % 2].Enable();
            this->depthmap_[1 - (maxLevel % 2)].BindColourTexture();

            this->depth_mip_shader_->setUniform("src", lx - lw, ly);
            this->depth_mip_shader_->setUniform("dst", lx, ly);

            lw /= 2;
            lh /= 2;
            ls /= 2;

            float x1, x2, y1, y2;

            x1 = float(lx) / float(this->depthmap_[0].GetWidth());
            x2 = float(lx + lw) / float(this->depthmap_[0].GetWidth());
            y1 = float(ly) / float(this->depthmap_[0].GetHeight());
            y2 = float(ly + lh) / float(this->depthmap_[0].GetHeight());

            glBegin(GL_QUADS);
            glVertex2f(-1.0f + 2.0f * x1, -1.0f + 2.0f * y1);
            glVertex2f(-1.0f + 2.0f * x2, -1.0f + 2.0f * y1);
            glVertex2f(-1.0f + 2.0f * x2, -1.0f + 2.0f * y2);
            glVertex2f(-1.0f + 2.0f * x1, -1.0f + 2.0f * y2);
            glEnd();

            this->depthmap_[maxLevel % 2].Disable();
            glBindTexture(GL_TEXTURE_2D, 0);

            lx += lw;
            maxLevel++;
        }

        glUseProgram(0); // this->depth_mip_shader_.Disable();

        this->depthmap_[0].Enable();
        this->init_depth_map_shader_->use();
        set_cam_uniforms(this->init_depth_map_shader_, view_matrix_inv, view_matrix_inv_transp, mvp_matrix,
            mvp_matrix_transp, mvp_matrix_inv, cam_pos, curlight_dir);
        this->init_depth_map_shader_->setUniform("datex", 0);
        this->depthmap_[1].BindColourTexture();

        lw = this->depthmap_[0].GetWidth() / 2;
        ly = this->depthmap_[0].GetHeight() * 2 / 3;
        lh = ly / 2;
        ls = vislib::math::Min(lh, lw);
        lx = lw;
        while (ls > 1) {

            lw /= 2;
            lh /= 2;
            ls /= 2;

            float x1, x2, y1, y2;

            x1 = float(lx) / float(this->depthmap_[0].GetWidth());
            x2 = float(lx + lw) / float(this->depthmap_[0].GetWidth());
            y1 = float(ly) / float(this->depthmap_[0].GetHeight());
            y2 = float(ly + lh) / float(this->depthmap_[0].GetHeight());

            glBegin(GL_QUADS);
            glVertex2f(-1.0f + 2.0f * x1, -1.0f + 2.0f * y1);
            glVertex2f(-1.0f + 2.0f * x2, -1.0f + 2.0f * y1);
            glVertex2f(-1.0f + 2.0f * x2, -1.0f + 2.0f * y2);
            glVertex2f(-1.0f + 2.0f * x1, -1.0f + 2.0f * y2);
            glEnd();

            lx += lw;

            // and skip one
            lw /= 2;
            lh /= 2;
            ls /= 2;
            lx += lw;
        }

        glUseProgram(0); // this->init_depth_map_shader_.Disable();
        this->depthmap_[0].Disable();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        glBindTexture(GL_TEXTURE_2D, 0);
        glPopDebugGroup();
        // END generation of depth-max mipmap
    }
#ifdef _WIN32
#pragma endregion depth buffer mipmaping
#endif // _WIN32

#if defined(DEBUG) || defined(_DEBUG)
    vislib::Trace::GetInstance().SetLevel(oldlevel);
#endif

    // image output ///////////////////////////////////////////////////////////
    unsigned int vis_cnt = 0;
    unsigned int vis_part = 0;

    if (speak_vert_count) {
        //
        // outputs the number of vertices surviving the vertex culling stage
        // usually not done when just drawing pictures
        //
        // THIS WILL NOT GENERATE ANY VISIBLE IMAGE OUTPUT !!!
        //
#ifdef _WIN32
#pragma region speak_vert_count
#endif // _WIN32

        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 5, -1, "grim-count-visible-points");
        GLuint all_query;
        glGenOcclusionQueriesNV(1, &all_query);
        glBeginOcclusionQueryNV(all_query);

        glDisable(GL_DEPTH_TEST);

        glPointSize(1.0f);
        if (use_vert_cull) {
            this->vert_cnt_shader_2_->use();
            set_cam_uniforms(this->vert_cnt_shader_2_, view_matrix_inv, view_matrix_inv_transp, mvp_matrix,
                mvp_matrix_transp, mvp_matrix_inv, cam_pos, curlight_dir);

            glUniform4fv(this->vert_cnt_shader_2_->getUniformLocation("viewAttr"), 1, viewport_stuff);
            glUniform3fv(this->vert_cnt_shader_2_->getUniformLocation("camIn"), 1, glm::value_ptr(cam_view));
            glUniform3fv(this->vert_cnt_shader_2_->getUniformLocation("camRight"), 1, glm::value_ptr(cam_right));
            glUniform3fv(this->vert_cnt_shader_2_->getUniformLocation("camUp"), 1, glm::value_ptr(cam_up));
            this->vert_cnt_shader_2_->setUniform("depthTexParams", (GLint) this->depthmap_[0].GetWidth(),
                (GLint) (this->depthmap_[0].GetHeight() * 2 / 3), (GLint) maxLevel);

            glEnable(GL_TEXTURE_2D);
            glActiveTextureARB(GL_TEXTURE2_ARB);
            this->depthmap_[0].BindColourTexture();
            this->vert_cnt_shader_2_->setUniform("depthTex", 2);
            glActiveTextureARB(GL_TEXTURE0_ARB);

            glColor3ub(128, 128, 128);
            glDisableClientState(GL_COLOR_ARRAY);
        } else {
            this->vert_cnt_shader_->use();
            set_cam_uniforms(this->vert_cnt_shader_, view_matrix_inv, view_matrix_inv_transp, mvp_matrix,
                mvp_matrix_transp, mvp_matrix_inv, cam_pos, curlight_dir);
        }

#ifdef SPEAK_CELL_USAGE
        printf("[vertCnt");
#endif
        for (int i = 0; i < static_cast<int>(cell_cnt); i++) { // front to back
            const moldyn::ParticleGridDataCall::GridCell& cell = pgdc->Cells()[i];
            CellInfo& info = infos[i];
            unsigned int pixel_count;
            if (!info.isvisible)
                continue; // frustum culling

            if (use_cell_cull) {
                glGetOcclusionQueryuivNV(info.oQuery, GL_PIXEL_COUNT_NV, &pixel_count);
                info.isvisible = (pixel_count > 0);
                //printf("pixel_count of cell %u is %u\n", idx, pixel_count);
                if (!info.isvisible)
                    continue; // occlusion culling
            } else {
                info.isvisible = true;
            }
            vis_cnt++;

#ifdef SPEAK_CELL_USAGE
            printf("-%d", i);
#endif
            for (unsigned int j = 0; j < type_cnt; j++) {
                const moldyn::ParticleGridDataCall::Particles& parts = cell.AccessParticleLists()[j];
                const moldyn::ParticleGridDataCall::ParticleType& ptype = pgdc->Types()[j];
                CellInfo::CacheItem& ci = info.cache[j];
                float min_c = 0.0f, max_c = 0.0f;
                unsigned int col_tab_size = 0;
                vis_part += parts.GetCount();

                // radius and position
                switch (ptype.GetVertexDataType()) {
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
                    continue;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    if (use_vert_cull) {
                        glUniform4f(this->vert_cnt_shader_2_->getUniformLocation("inConsts1"), ptype.GetGlobalRadius(),
                            min_c, max_c, float(col_tab_size));
                    }
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                        glVertexPointer(3, GL_FLOAT, 0, NULL);
                    } else {
                        glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    if (use_vert_cull) {
                        glUniform4f(this->vert_cnt_shader_2_->getUniformLocation("inConsts1"), -1.0f, min_c, max_c,
                            float(col_tab_size));
                    }
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                        glVertexPointer(4, GL_FLOAT, 0, NULL);
                    } else {
                        glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ: {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "GrimRenderer: vertices with short coords are deprecated!");
                } break;

                default:
                    continue;
                }

                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));
                glBindBufferARB(GL_ARRAY_BUFFER, 0);
                //glDisableClientState(GL_COLOR_ARRAY);
                glDisableClientState(GL_VERTEX_ARRAY);
            }
        }
        glUseProgram(0); // (use_vert_cull ? this->vert_cnt_shader_2_ : this->vert_cnt_shader_).Disable();
#ifdef SPEAK_CELL_USAGE
        printf("]\n");
#endif

        unsigned int total_schnitzels = 0;
        glEndOcclusionQueryNV();
        glFlush();
        glGetOcclusionQueryuivNV(all_query, GL_PIXEL_COUNT_NV, &total_schnitzels);
        glDeleteOcclusionQueriesNV(1, &all_query);

        if (speak && speak_vert_count) {
            unsigned int total_spheres = 0;
            for (int i = 0; i < static_cast<int>(cell_cnt); i++) {
                const moldyn::ParticleGridDataCall::GridCell& cell = pgdc->Cells()[i];
                for (unsigned int j = 0; j < type_cnt; j++) {
                    const moldyn::ParticleGridDataCall::Particles& parts = cell.AccessParticleLists()[j];
                    total_spheres += parts.GetCount();
                }
            }
            printf("VERTEX COUNT: %u (%f%%)\n", static_cast<unsigned int>(total_schnitzels),
                static_cast<float>(total_schnitzels) / static_cast<float>(total_spheres) * 100.0f);
        }
        glPopDebugGroup();
#ifdef _WIN32
#pragma endregion speak_vert_count
#endif // _WIN32

    } else {
        //
        // GENERATE VISIBLE IMAGE OUTPUT
        //
        if (deferred_shading) {
#if defined(DEBUG) || defined(_DEBUG)
            UINT oldlevel = vislib::Trace::GetInstance().GetLevel();
            vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_NONE);
#endif
            this->ds_fbo_.EnableMultiple(
                3, GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClearDepth(1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // not sure about this one
#if defined(DEBUG) || defined(_DEBUG)
            vislib::Trace::GetInstance().SetLevel(oldlevel);
#endif
        } else {

            // REACTIVATE TARGET fbo_
            cr->GetFramebuffer()->bind();
        }

        glEnable(GL_DEPTH_TEST);
        glPointSize(1.0f);
        glDisableClientState(GL_COLOR_ARRAY);

        // draw points ///////////////////////////////////////////////////////
#ifdef SPEAK_CELL_USAGE
        printf("[drawd");
#endif
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 6, -1, "grim-draw-dots");
        // draw visible data (dots)
        da_point_shader->use();
        set_cam_uniforms(da_point_shader, view_matrix_inv, view_matrix_inv_transp, mvp_matrix, mvp_matrix_transp,
            mvp_matrix_inv, cam_pos, curlight_dir);
        for (int i = cell_cnt - 1; i >= 0; i--) { // front to back
            const moldyn::ParticleGridDataCall::GridCell& cell = pgdc->Cells()[i];
            CellInfo& info = infos[i];
            unsigned int pixel_count;
            if (!info.isvisible)
                continue; // frustum culling
            if (!info.dots)
                continue;

            if (use_cell_cull) {
                glGetOcclusionQueryuivNV(info.oQuery, GL_PIXEL_COUNT_NV, &pixel_count);
                info.isvisible = (pixel_count > 0);
                //printf("pixel_count of cell %u is %u\n", idx, pixel_count);
                if (!info.isvisible)
                    continue; // occlusion culling
            } else {
                info.isvisible = true;
            }
            vis_cnt++;

#ifdef SPEAK_CELL_USAGE
            printf("-%d", i);
#endif
            for (unsigned int j = 0; j < type_cnt; j++) {
                const moldyn::ParticleGridDataCall::Particles& parts = cell.AccessParticleLists()[j];
                const moldyn::ParticleGridDataCall::ParticleType& ptype = pgdc->Types()[j];
                CellInfo::CacheItem& ci = info.cache[j];
                float min_c = 0.0f, max_c = 0.0f;
                unsigned int col_tab_size = 0;
                vis_part += parts.GetCount();

                // colour
                switch (ptype.GetColourDataType()) {
                case geocalls::MultiParticleDataCall::Particles::COLDATA_NONE:
                    glColor3ubv(ptype.GetGlobalColour());
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                    glEnableClientState(GL_COLOR_ARRAY);
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                        glColorPointer(3, GL_UNSIGNED_BYTE, 0, NULL);
                    } else {
                        glColorPointer(3, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), parts.GetColourData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                    glEnableClientState(GL_COLOR_ARRAY);
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                        glColorPointer(4, GL_UNSIGNED_BYTE, 0, NULL);
                    } else {
                        glColorPointer(4, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), parts.GetColourData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                    glEnableClientState(GL_COLOR_ARRAY);
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                        glColorPointer(3, GL_FLOAT, 0, NULL);
                    } else {
                        glColorPointer(3, GL_FLOAT, parts.GetColourDataStride(), parts.GetColourData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                    glEnableClientState(GL_COLOR_ARRAY);
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                        glColorPointer(4, GL_FLOAT, 0, NULL);
                    } else {
                        glColorPointer(4, GL_FLOAT, parts.GetColourDataStride(), parts.GetColourData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                    glEnableVertexAttribArrayARB(cial2);
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                        glVertexAttribPointerARB(cial2, 1, GL_FLOAT, GL_FALSE, 0, NULL);
                    } else {
                        glVertexAttribPointerARB(
                            cial2, 1, GL_FLOAT, GL_FALSE, parts.GetColourDataStride(), parts.GetColourData());
                    }

                    // Bind transfer function texture
                    glEnable(GL_TEXTURE_1D);
                    mmstd_gl::CallGetTransferFunctionGL* cgtf =
                        this->get_tf_slot_.CallAs<mmstd_gl::CallGetTransferFunctionGL>();
                    if ((cgtf != NULL) && ((*cgtf)())) {
                        glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                        col_tab_size = cgtf->TextureSize();
                    } else {
                        glBindTexture(GL_TEXTURE_1D, this->grey_tf_);
                        col_tab_size = 2;
                    }

                    glUniform1i(da_point_shader->getUniformLocation("colTab"), 0);
                    min_c = ptype.GetMinColourIndexValue();
                    max_c = ptype.GetMaxColourIndexValue();
                    glColor3ub(127, 127, 127);
                } break;
                default:
                    glColor3ub(127, 127, 127);
                    break;
                }

                // radius and position
                switch (ptype.GetVertexDataType()) {
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
                    continue;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4f(da_point_shader->getUniformLocation("inConsts1"), ptype.GetGlobalRadius(), min_c, max_c,
                        float(col_tab_size));
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                        glVertexPointer(3, GL_FLOAT, 0, NULL);
                    } else {
                        glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4f(
                        da_point_shader->getUniformLocation("inConsts1"), -1.0f, min_c, max_c, float(col_tab_size));
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                        glVertexPointer(3, GL_FLOAT, 16, NULL);
                    } else {
                        glVertexPointer(
                            3, GL_FLOAT, vislib::math::Max(16U, parts.GetVertexDataStride()), parts.GetVertexData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ: {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "GrimRenderer: vertices with short coords are deprecated!");
                } break;

                default:
                    continue;
                }
                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));
                glBindBufferARB(GL_ARRAY_BUFFER, 0);
                glDisableClientState(GL_COLOR_ARRAY);
                glDisableClientState(GL_VERTEX_ARRAY);
                glDisableVertexAttribArrayARB(cial2);
                glDisable(GL_TEXTURE_1D);
            }
        }
#ifdef SPEAK_CELL_USAGE
        printf("]\n");
#endif
        glUseProgram(0); // da_point_shader->Disable();
        glPopDebugGroup();

        // draw spheres ///////////////////////////////////////////////////////
#ifdef SPEAK_CELL_USAGE
        printf("[draws");
#endif
        // draw visible data (spheres)
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 7, -1, "grim-draw-spheres");
        da_sphere_shader->use();
        set_cam_uniforms(da_sphere_shader, view_matrix_inv, view_matrix_inv_transp, mvp_matrix, mvp_matrix_transp,
            mvp_matrix_inv, cam_pos, curlight_dir);
#ifdef SUPSAMP_LOOP
        for (int supsamppass = 0; supsamppass < SUPSAMP_LOOPCNT; supsamppass++) {
#endif // SUPSAMP_LOOP

            glUniform4fv(da_sphere_shader->getUniformLocation("viewAttr"), 1, viewport_stuff);
            glUniform3fv(da_sphere_shader->getUniformLocation("camIn"), 1, glm::value_ptr(cam_view));
            glUniform3fv(da_sphere_shader->getUniformLocation("camRight"), 1, glm::value_ptr(cam_right));
            glUniform3fv(da_sphere_shader->getUniformLocation("camUp"), 1, glm::value_ptr(cam_up));
            glUniform1i(da_sphere_shader->getUniformLocation("use_shading"), static_cast<int>(!deferred_shading));

            if (use_vert_cull) {
                da_sphere_shader->setUniform("depthTexParams", (GLint) this->depthmap_[0].GetWidth(),
                    (GLint) (this->depthmap_[0].GetHeight() * 2 / 3), (GLint) maxLevel);
                glEnable(GL_TEXTURE_2D);
                glActiveTextureARB(GL_TEXTURE2_ARB);
                this->depthmap_[0].BindColourTexture();
                da_sphere_shader->setUniform("depthTex", 2);
                glActiveTextureARB(GL_TEXTURE0_ARB);
            } else {
                da_sphere_shader->setUniform("clipDat", 0.0f, 0.0f, 0.0f, 0.0f);
                da_sphere_shader->setUniform("clipCol", 0.0f, 0.0f, 0.0f);
            }
            glPointSize(default_point_size);

            for (int i = cell_cnt - 1; i >= 0; i--) { // front to back
                unsigned int idx = dists[i].First();
                const moldyn::ParticleGridDataCall::GridCell& cell = pgdc->Cells()[idx];
                CellInfo& info = infos[idx];

                unsigned int pixel_count;
                if (!info.isvisible)
                    continue; // frustum culling
                if (info.dots)
                    continue;

                if (use_cell_cull) {
                    glGetOcclusionQueryuivNV(info.oQuery, GL_PIXEL_COUNT_NV, &pixel_count);
                    info.isvisible = (pixel_count > 0);
                    //printf("pixel_count of cell %u is %u\n", idx, pixel_count);
                    if (!info.isvisible)
                        continue; // occlusion culling
                } else {
                    info.isvisible = true;
                }
                vis_cnt++;

#ifdef SPEAK_CELL_USAGE
                printf("-%d", i);
#endif
                for (unsigned int j = 0; j < type_cnt; j++) {
                    const moldyn::ParticleGridDataCall::Particles& parts = cell.AccessParticleLists()[j];
                    const moldyn::ParticleGridDataCall::ParticleType& ptype = pgdc->Types()[j];
                    CellInfo::CacheItem& ci = info.cache[j];
                    float min_c = 0.0f, max_c = 0.0f;
                    unsigned int col_tab_size = 0;
                    vis_part += parts.GetCount();

                    // colour
                    switch (ptype.GetColourDataType()) {
                    case geocalls::MultiParticleDataCall::Particles::COLDATA_NONE:
                        glColor3ubv(ptype.GetGlobalColour());
                        break;
                    case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                        glEnableClientState(GL_COLOR_ARRAY);
                        if (ci.data[0] != 0) {
                            glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                            glColorPointer(3, GL_UNSIGNED_BYTE, 0, NULL);
                        } else {
                            glColorPointer(3, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), parts.GetColourData());
                        }
                        break;
                    case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                        glEnableClientState(GL_COLOR_ARRAY);
                        if (ci.data[0] != 0) {
                            glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                            glColorPointer(4, GL_UNSIGNED_BYTE, 0, NULL);
                        } else {
                            glColorPointer(4, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), parts.GetColourData());
                        }
                        break;
                    case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                        glEnableClientState(GL_COLOR_ARRAY);
                        if (ci.data[0] != 0) {
                            glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                            glColorPointer(3, GL_FLOAT, 0, NULL);
                        } else {
                            glColorPointer(3, GL_FLOAT, parts.GetColourDataStride(), parts.GetColourData());
                        }
                        break;
                    case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                        glEnableClientState(GL_COLOR_ARRAY);
                        if (ci.data[0] != 0) {
                            glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                            glColorPointer(4, GL_FLOAT, 0, NULL);
                        } else {
                            glColorPointer(4, GL_FLOAT, parts.GetColourDataStride(), parts.GetColourData());
                        }
                        break;
                    case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                        glEnableVertexAttribArrayARB(cial);
                        if (ci.data[0] != 0) {
                            glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                            glVertexAttribPointerARB(cial, 1, GL_FLOAT, GL_FALSE, 0, NULL);
                        } else {
                            glVertexAttribPointerARB(
                                cial, 1, GL_FLOAT, GL_FALSE, parts.GetColourDataStride(), parts.GetColourData());
                        }

                        // Bind transfer function texture
                        glEnable(GL_TEXTURE_1D);
                        mmstd_gl::CallGetTransferFunctionGL* cgtf =
                            this->get_tf_slot_.CallAs<mmstd_gl::CallGetTransferFunctionGL>();
                        if ((cgtf != NULL) && ((*cgtf)())) {
                            glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                            col_tab_size = cgtf->TextureSize();
                        } else {
                            glBindTexture(GL_TEXTURE_1D, this->grey_tf_);
                            col_tab_size = 2;
                        }

                        glUniform1i(da_sphere_shader->getUniformLocation("colTab"), 0);
                        min_c = ptype.GetMinColourIndexValue();
                        max_c = ptype.GetMaxColourIndexValue();
                        glColor3ub(127, 127, 127);
                    } break;
                    default:
                        glColor3ub(127, 127, 127);
                        break;
                    }

                    // radius and position
                    switch (ptype.GetVertexDataType()) {
                    case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
                        continue;
                    case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                        glEnableClientState(GL_VERTEX_ARRAY);
                        glUniform4f(da_sphere_shader->getUniformLocation("inConsts1"), ptype.GetGlobalRadius(), min_c,
                            max_c, float(col_tab_size));
                        if (ci.data[0] != 0) {
                            glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                            glVertexPointer(3, GL_FLOAT, 0, NULL);
                        } else {
                            glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                        }
                        break;
                    case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                        glEnableClientState(GL_VERTEX_ARRAY);
                        glUniform4f(da_sphere_shader->getUniformLocation("inConsts1"), -1.0f, min_c, max_c,
                            float(col_tab_size));
                        if (ci.data[0] != 0) {
                            glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                            glVertexPointer(4, GL_FLOAT, 0, NULL);
                        } else {
                            glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                        }
                        break;
                    case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ: {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "GrimRenderer: vertices with short coords are deprecated!");
                    } break;

                    default:
                        continue;
                    }

                    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));
                    glBindBufferARB(GL_ARRAY_BUFFER, 0);
                    glDisableClientState(GL_COLOR_ARRAY);
                    glDisableClientState(GL_VERTEX_ARRAY);
                    glDisableVertexAttribArrayARB(cial);
                    glDisable(GL_TEXTURE_1D);
                }
            }
#ifdef SPEAK_CELL_USAGE
            printf("]\n");
#endif

#ifdef SUPSAMP_LOOP
        }
#endif // SUPSAMP_LOOP

        if (deferred_shading) {
            this->ds_fbo_.Disable();
        }

        glUseProgram(0); // da_sphere_shader->Disable();
        glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
        glDisable(GL_TEXTURE_2D);

        glPopDebugGroup();
    }

    if (speak_cell_perc) {
        printf("CELLS VISIBLE: %f%%\n", float(vis_cnt * 100) / float(cell_cnt));
        printf("PARTICLES IN VISIBLE CELLS: %u\n", vis_part);
    }

    // remove unused cache item ///////////////////////////////////////////////
    if ((this->cache_size_used_ * 5 / 4) > this->cache_size_) {
        for (int i = cell_cnt - 1; i >= 0; i--) { // front to back
            unsigned int idx = dists[i].First();
            const moldyn::ParticleGridDataCall::GridCell& cell = pgdc->Cells()[idx];
            CellInfo& info = infos[idx];

            if (info.wasvisible)
                continue; // this one is still required

            for (unsigned int j = 0; j < type_cnt; j++) {
                const moldyn::ParticleGridDataCall::Particles& parts = cell.AccessParticleLists()[j];
                const moldyn::ParticleGridDataCall::ParticleType& ptype = pgdc->Types()[j];
                CellInfo::CacheItem& ci = info.cache[j];

                if ((ci.data[0] == 0) || (parts.GetCount() == 0))
                    continue; // not cached or no data

                unsigned int vbpp = 1, cbpp = 1;
                switch (ptype.GetVertexDataType()) {
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    vbpp = 3 * sizeof(float);
                    break;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    vbpp = 4 * sizeof(float);
                    break;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
                    vbpp = 3 * sizeof(short);
                    break;
                default:
                    continue;
                }
                switch (ptype.GetColourDataType()) {
                case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                    cbpp = 3;
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                    cbpp = 4;
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                    cbpp = 3 * sizeof(float);
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                    cbpp = 4 * sizeof(float);
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
                    cbpp = sizeof(float);
                    break;
                default:
                    break;
                }

                glDeleteBuffersARB(2, ci.data);
                ci.data[0] = ci.data[1] = 0;

                this->cache_size_used_ -= (vbpp + cbpp) * parts.GetCount();
#ifdef SPEAK_VRAM_CACHE_USAGE
                printf("VRAM-Cache: Del[%d; %u] %u/%u\n", i, j, this->cache_size_used_, this->cache_size_);
#endif // SPEAK_VRAM_CACHE_USAGE
            }
        }
    }

    // deferred shading ///////////////////////////////////////////////////////
    if (deferred_shading) {
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 8, -1, "grim-deferred-shading");
        cr->GetFramebuffer()->bind();

        glEnable(GL_TEXTURE_2D);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);

        this->deferred_shader_->use();
        // useless, everything is identity here
        //set_cam_uniforms(this->deferred_shader_, view_matrix_inv, view_matrix_inv_transp, mvp_matrix,
        //    mvp_matrix_transp, mvp_matrix_inv, cam_pos, curlight_dir);

        glActiveTextureARB(GL_TEXTURE0_ARB);
        this->ds_fbo_.BindColourTexture(0);
        glActiveTextureARB(GL_TEXTURE1_ARB);
        this->ds_fbo_.BindColourTexture(1);
        glActiveTextureARB(GL_TEXTURE2_ARB);
        this->ds_fbo_.BindColourTexture(2);

        this->deferred_shader_->setUniform("colour", 0);
        this->deferred_shader_->setUniform("normal", 1);
        this->deferred_shader_->setUniform("pos", 2);

        glm::vec3 test = cam_view;

        glm::vec3 ray(cam_view);
        glm::vec3 up(cam_up);
        glm::vec3 right(cam_right);

        //vislib::math::Vector<float, 4> light_dir;
        //vislib::math::ShallowVector<float, 3> lp(light_dir.PeekComponents());
        //lp = right;
        //lp *= -0.5f;
        //lp -= ray;
        //lp += up;
        //light_dir[3] = 0.0f;
        //this->deferred_shader_.SetParameterArray3("lightDir", 1, light_dir.PeekComponents());
        this->deferred_shader_->setUniform("lightDir", curlight_dir);

        up *= sinf(half_aperture_angle);
        right *=
            sinf(half_aperture_angle) * static_cast<float>(fbo_->getWidth()) / static_cast<float>(fbo_->getHeight());

        this->deferred_shader_->setUniform("ray", glm::vec3(cam_view.x, cam_view.y, cam_view.z));

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glColor3ub(255, 255, 255);
        glBegin(GL_QUADS);
        glNormal3fv(glm::value_ptr(ray - right - up));
        glTexCoord2f(0.0f, 0.0f);
        glVertex2i(-1, -1);
        glNormal3fv(glm::value_ptr(ray + right - up));
        glTexCoord2f(1.0f, 0.0f);
        glVertex2i(1, -1);
        glNormal3fv(glm::value_ptr(ray + right + up));
        glTexCoord2f(1.0f, 1.0f);
        glVertex2i(1, 1);
        glNormal3fv(glm::value_ptr(ray - right + up));
        glTexCoord2f(0.0f, 1.0f);
        glVertex2i(-1, 1);
        glEnd();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        glActiveTextureARB(GL_TEXTURE0_ARB);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTextureARB(GL_TEXTURE1_ARB);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTextureARB(GL_TEXTURE2_ARB);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTextureARB(GL_TEXTURE0_ARB);

        glUseProgram(0); // this->deferred_shader_.Disable();
        glPopDebugGroup();
    }

    //// DEBUG OUTPUT OF fbo_ --------------------------------------------------
    //cr->EnableOutputBuffer();
    //glEnable(GL_TEXTURE_2D);
    //glDisable(GL_LIGHTING);
    //glDisable(GL_DEPTH_TEST);

    ////this->fbo_.BindDepthTexture();
    ////this->fbo_.BindColourTexture();
    ////this->depthmap_[0].BindColourTexture();

    ////this->ds_fbo_.BindColourTexture(0);
    ////this->ds_fbo_.BindColourTexture(1);
    //this->ds_fbo_.BindColourTexture(2);
    ////this->ds_fbo_.BindDepthTexture();

    //glMatrixMode(GL_PROJECTION);
    //glPushMatrix();
    //glLoadIdentity();
    //glMatrixMode(GL_MODELVIEW);
    //glPushMatrix();
    //glLoadIdentity();
    //glColor3ub(255, 255, 255);
    //glBegin(GL_QUADS);
    //glTexCoord2f(0.0f, 0.0f);
    //glVertex2i(-1, -1);
    //glTexCoord2f(1.0f, 0.0f);
    //glVertex2i(1, -1);
    //glTexCoord2f(1.0f, 1.0f);
    //glVertex2i(1, 1);
    //glTexCoord2f(0.0f, 1.0f);
    //glVertex2i(-1, 1);
    //glEnd();
    //glMatrixMode(GL_PROJECTION);
    //glPopMatrix();
    //glMatrixMode(GL_MODELVIEW);
    //glPopMatrix();
    //glBindTexture(GL_TEXTURE_2D, 0);

    // done!
    pgdc->Unlock();

    for (int i = cell_cnt - 1; i >= 0; i--) {
        CellInfo& info = infos[i];
        info.wasvisible = info.isvisible;
    }

    // Reset default OpenGL state ---------------------------------------------
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glDisable(GL_CLIP_DISTANCE0);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDisable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_POINT_SPRITE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glDisable(GL_CULL_FACE);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_TEXTURE_1D);
    glDisable(GL_LIGHTING);
    glPointSize(1.0f);
    glLineWidth(1.0f);

    return true;
}


bool GrimRenderer::depthSort(
    const vislib::Pair<unsigned int, float>& lhs, const vislib::Pair<unsigned int, float>& rhs) {

    return (rhs.Second() < lhs.Second());

    //float d = rhs.Second() - lhs.Second();
    //if (d > vislib::math::FLOAT_EPSILON) return ;
    //if (d < -vislib::math::FLOAT_EPSILON) return -1;
    //return 0;
}
