/*
 * GlyphRenderer.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "GlyphRenderer.h"
#include "OpenGL_Context.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "inttypes.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmstd/renderer/CallClipPlane.h"
#include "mmstd_gl/flags/FlagCallsGL.h"
#include "mmstd_gl/renderer/CallGetTransferFunctionGL.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/String.h"
#include "vislib/assert.h"
#include "vislib/math/Quaternion.h"
#include <cstring>
#include <iostream>


#ifdef MEGAMOL_USE_PROFILING
#include "PerformanceManager.h"
#endif

using namespace megamol;
using namespace megamol::core;
using namespace megamol::moldyn_gl;
using namespace megamol::moldyn_gl::rendering;

//const uint32_t max_ssbo_size = 2 * 1024 * 1024 * 1024;

GlyphRenderer::GlyphRenderer(void)
        : mmstd_gl::Renderer3DModuleGL()
        , get_data_slot_("getData", "The slot to fetch the data")
        , get_tf_slot_("getTF", "The slot for the transfer function")
        , get_clip_plane_slot_("getClipPlane", "The slot for the clip plane")
        , read_flags_slot_("readFlags", "The slot for reading the selection flags")
        , glyph_param_("glyph", "Which glyph to render")
        , scale_param_("scaling", "TODO: scales the box??")
        , radius_scale_param_("radius_scaling", "scales the glyph radii")
        , orientation_param_("Orientation", "Selects along which axis the arrows are aligned")
        , length_filter_param_("lengthFilter", "Filters the arrows by length")
        , color_interpolation_param_(
              "colorInterpolation", "Interpolate between directional coloring (0) and glyph color (1)")
        , min_radius_param_("minRadius", "Sets the minimum radius length. Applied to each axis.")
        , color_mode_param_("colorMode", "Switch between global glyph and per axis color")
        , superquadric_exponent_param_("exponent", "Sets the exponent used in the implicit superquadric equation")
        , gizmo_arrow_thickness_("Thickness", "Sets the arrow thickness of the gizmo arrows") {

    this->get_data_slot_.SetCompatibleCall<geocalls::EllipsoidalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->get_data_slot_);

    this->get_tf_slot_.SetCompatibleCall<mmstd_gl::CallGetTransferFunctionGLDescription>();
    this->MakeSlotAvailable(&this->get_tf_slot_);

    this->get_clip_plane_slot_.SetCompatibleCall<view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->get_clip_plane_slot_);

    this->read_flags_slot_.SetCompatibleCall<mmstd_gl::FlagCallRead_GLDescription>();
    this->MakeSlotAvailable(&this->read_flags_slot_);

    param::EnumParam* gp = new param::EnumParam(0);
    gp->SetTypePair(Glyph::BOX, "Box");
    gp->SetTypePair(Glyph::ELLIPSOID, "Ellipsoid");
    gp->SetTypePair(Glyph::ARROW, "Arrow");
    gp->SetTypePair(Glyph::SUPERQUADRIC, "Superquadric");
    gp->SetTypePair(Glyph::GIZMO_ARROWGLYPH, "GizmoArrow");
    this->glyph_param_ << gp;
    this->MakeSlotAvailable(&this->glyph_param_);

    scale_param_ << new param::FloatParam(1.0f, 0.0f, 100.0f);
    this->MakeSlotAvailable(&this->scale_param_);

    radius_scale_param_ << new param::FloatParam(1.0f, 0.0f, 100.0f, 0.1f);
    this->MakeSlotAvailable(&this->radius_scale_param_);

    // currently only needed for arrow
    param::EnumParam* op = new param::EnumParam(3);
    op->SetTypePair(0, "x");
    op->SetTypePair(1, "y");
    op->SetTypePair(2, "z");
    op->SetTypePair(3, "largest radius");
    this->orientation_param_ << op;
    this->MakeSlotAvailable(&this->orientation_param_);

    length_filter_param_ << new param::FloatParam(0.0f, 0.0f);
    this->MakeSlotAvailable(&this->length_filter_param_);

    color_interpolation_param_ << new param::FloatParam(1.0f, 0.0, 1.0f);
    this->MakeSlotAvailable(&this->color_interpolation_param_);

    min_radius_param_ << new param::FloatParam(0.1f, 0.f, 2.f, 0.01f);
    this->MakeSlotAvailable(&this->min_radius_param_);

    param::EnumParam* gcm = new param::EnumParam(0);
    gcm->SetTypePair(0, "GlyphGlobal");
    gcm->SetTypePair(1, "PerAxis");
    this->color_mode_param_ << gcm;
    this->MakeSlotAvailable(&this->color_mode_param_);

    superquadric_exponent_param_ << new param::FloatParam(1.0f, -100.0f, 100.0f, 0.1f);
    this->MakeSlotAvailable(&this->superquadric_exponent_param_);

    gizmo_arrow_thickness_ << new param::FloatParam(0.1f, 0.0f, 2.0f, 0.01f);
    this->MakeSlotAvailable(&this->gizmo_arrow_thickness_);
}


GlyphRenderer::~GlyphRenderer(void) {
    this->Release();
}


bool GlyphRenderer::create(void) {
    // profiling
#ifdef MEGAMOL_USE_PROFILING
    perf_manager_ = const_cast<frontend_resources::PerformanceManager*>(
        &frontend_resources.get<frontend_resources::PerformanceManager>());

    frontend_resources::PerformanceManager::basic_timer_config render_timer;
    render_timer.name = "render";
    render_timer.api = frontend_resources::PerformanceManager::query_api::OPENGL;
    timers_ = perf_manager_->add_timers(this, {render_timer});
#endif

    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();

    // create shader programs
    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

    try {
        // TODO: use std::filesystem::path?
        box_prgm_ = core::utility::make_glowl_shader(
            "box", shader_options, "moldyn_gl/glyph_renderer/box.vert.glsl", "moldyn_gl/glyph_renderer/box.frag.glsl");

        ellipsoid_prgm_ = core::utility::make_glowl_shader("ellipsoid", shader_options,
            "moldyn_gl/glyph_renderer/ellipsoid.vert.glsl", "moldyn_gl/glyph_renderer/ellipsoid.frag.glsl");

        arrow_prgm_ = core::utility::make_glowl_shader("arrow", shader_options,
            "moldyn_gl/glyph_renderer/arrow.vert.glsl", "moldyn_gl/glyph_renderer/arrow.frag.glsl");

        superquadric_prgm_ = core::utility::make_glowl_shader("superquadric", shader_options,
            "moldyn_gl/glyph_renderer/superquadric.vert.glsl", "moldyn_gl/glyph_renderer/superquadric.frag.glsl");

        gizmo_arrowglyph_prgm_ = core::utility::make_glowl_shader("gizmo_arrowglyph", shader_options,
            "moldyn_gl/glyph_renderer/gizmo_arrowglyph.vert.glsl",
            "moldyn_gl/glyph_renderer/gizmo_arrowglyph.frag.glsl");
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to compile glyph shader: %s. [%s, %s, line %d]\n", std::string(e.what()).c_str(), __FILE__,
            __FUNCTION__, __LINE__);

        return false;
    }

    glEnable(GL_TEXTURE_1D);
    glGenTextures(1, &this->grey_tf_);
    unsigned char tex[6] = {0, 0, 0, 255, 255, 255};
    glBindTexture(GL_TEXTURE_1D, this->grey_tf_);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glBindTexture(GL_TEXTURE_1D, 0);
    glDisable(GL_TEXTURE_1D);

    return true;
}

bool GlyphRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {

    auto* rhs_epdc = this->get_data_slot_.CallAs<geocalls::EllipsoidalParticleDataCall>();
    if ((rhs_epdc != NULL) && ((*rhs_epdc)(1))) {
        call.SetTimeFramesCount(rhs_epdc->FrameCount());
        call.AccessBoundingBoxes() = rhs_epdc->AccessBoundingBoxes();
    } else {
        call.SetTimeFramesCount(1);
        call.AccessBoundingBoxes().Clear();
    }

    return true;
}

void GlyphRenderer::release(void) {
    glDeleteTextures(1, &this->grey_tf_);

#ifdef MEGAMOL_USE_PROFILING
    perf_manager_->remove_timers(timers_);
#endif
}

bool megamol::moldyn_gl::rendering::GlyphRenderer::validateData(geocalls::EllipsoidalParticleDataCall* rhs_edc) {

    if (this->last_hash_ != rhs_edc->DataHash() || this->last_frame_id_ != rhs_edc->FrameID()) {
        this->position_buffers_.reserve(rhs_edc->GetParticleListCount());
        this->radius_buffers_.reserve(rhs_edc->GetParticleListCount());
        this->direction_buffers_.reserve(rhs_edc->GetParticleListCount());
        this->color_buffers_.reserve(rhs_edc->GetParticleListCount());

        for (uint32_t x = 0; x < rhs_edc->GetParticleListCount(); ++x) {
            auto& l = rhs_edc->AccessParticles(x);
            this->position_buffers_.emplace_back(utility::SSBOBufferArray("position_buffer" + std::to_string(x)));
            this->radius_buffers_.emplace_back(utility::SSBOBufferArray("radius_buffer" + std::to_string(x)));
            this->direction_buffers_.emplace_back(utility::SSBOBufferArray("direction_buffer" + std::to_string(x)));
            this->color_buffers_.emplace_back(utility::SSBOBufferArray("color_buffer" + std::to_string(x)));

            this->direction_buffers_[x].SetData(l.GetQuatData(), l.GetQuatDataStride(), 4 * sizeof(float), l.GetCount(),
                [](void* dst, const void* src) { memcpy(dst, src, sizeof(float) * 4); });
            const auto num_items_per_chunk = this->direction_buffers_[x].GetMaxNumItemsPerChunk();

            switch (l.GetVertexDataType()) {
            case geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ:
            case geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR:
                // radius is skipped automatically by memcpy width
                this->position_buffers_[x].SetDataWithItems(l.GetVertexData(), l.GetVertexDataStride(),
                    3 * sizeof(float), l.GetCount(), num_items_per_chunk,
                    [](void* dst, const void* src) { memcpy(dst, src, sizeof(float) * 3); });
                break;
            case geocalls::SimpleSphericalParticles::VERTDATA_DOUBLE_XYZ:
                // narrow double to float
                this->position_buffers_[x].SetDataWithItems(l.GetVertexData(), l.GetVertexDataStride(),
                    3 * sizeof(float), l.GetCount(), num_items_per_chunk, [](void* dst, const void* src) {
                        auto* d = static_cast<float*>(dst);
                        const auto* s = static_cast<const double*>(src);
                        for (auto i = 0; i < 3; ++i) {
                            *(d + i) = static_cast<float>(*(s + i));
                        }
                    });
                break;
            case geocalls::SimpleSphericalParticles::VERTDATA_SHORT_XYZ:
            case geocalls::SimpleSphericalParticles::VERTDATA_NONE:
            default:
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "GlyphRenderer: no support for vertex data types SHORT_XYZ or NONE");
                return false;
            }

            switch (l.GetColourDataType()) {
            case geocalls::SimpleSphericalParticles::COLDATA_NONE:
                break;
            case geocalls::SimpleSphericalParticles::COLDATA_UINT8_RGB:
                // we could just pad this, but such datasets need to disappear...
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "GlyphRenderer: COLDATA_UINT8_RGB is deprecated and unsupported");
                return false;
            case geocalls::SimpleSphericalParticles::COLDATA_UINT8_RGBA:
                // extend to floats
                this->color_buffers_[x].SetDataWithItems(l.GetColourData(), l.GetColourDataStride(), 4 * sizeof(float),
                    l.GetCount(), num_items_per_chunk, [](void* dst, const void* src) {
                        auto* d = static_cast<float*>(dst);
                        const auto* s = static_cast<const unsigned char*>(src);
                        for (auto i = 0; i < 4; ++i) {
                            *(d + i) = static_cast<float>(*(s + i)) / 255.0f;
                        }
                    });
                break;
            case geocalls::SimpleSphericalParticles::COLDATA_FLOAT_RGB:
                this->color_buffers_[x].SetDataWithItems(l.GetColourData(), l.GetColourDataStride(), 4 * sizeof(float),
                    l.GetCount(), num_items_per_chunk, [](void* dst, const void* src) {
                        memcpy(dst, src, sizeof(float) * 3);
                        auto* d = static_cast<float*>(dst);
                        *(d + 3) = 1.0f;
                    });
                break;
            case geocalls::SimpleSphericalParticles::COLDATA_FLOAT_RGBA:
                // this is paranoid and could be avoided for cases where data is NOT interleaved
                this->color_buffers_[x].SetDataWithItems(l.GetColourData(), l.GetColourDataStride(), 4 * sizeof(float),
                    l.GetCount(), num_items_per_chunk,
                    [](void* dst, const void* src) { memcpy(dst, src, sizeof(float) * 4); });
                break;
            case geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I:
                this->color_buffers_[x].SetDataWithItems(l.GetColourData(), l.GetColourDataStride(), 4 * sizeof(float),
                    l.GetCount(), num_items_per_chunk, [](void* dst, const void* src) {
                        auto* d = static_cast<float*>(dst);
                        const auto* s = static_cast<const float*>(src);
                        *d = *s;
                        // rest is garbage
                    });
                break;
            case geocalls::SimpleSphericalParticles::COLDATA_USHORT_RGBA:
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "GlyphRenderer: COLDATA_USHORT_RGBA is unsupported");
                return false;
            case geocalls::SimpleSphericalParticles::COLDATA_DOUBLE_I:
                this->color_buffers_[x].SetDataWithItems(l.GetColourData(), l.GetColourDataStride(), 4 * sizeof(float),
                    l.GetCount(), num_items_per_chunk, [](void* dst, const void* src) {
                        auto* d = static_cast<float*>(dst);
                        const auto* s = static_cast<const double*>(src);
                        *d = static_cast<float>(*s);
                        // rest is garbage
                    });
                break;
            default:;
            }

            this->radius_buffers_[x].SetDataWithItems(l.GetRadiiData(), l.GetRadiiDataStride(), sizeof(float) * 3,
                l.GetCount(), num_items_per_chunk,
                [](void* dst, const void* src) { memcpy(dst, src, sizeof(float) * 3); });
        }
        this->last_hash_ = rhs_edc->DataHash();
        this->last_frame_id_ = rhs_edc->FrameID();
    }
    return true;
}

bool GlyphRenderer::Render(mmstd_gl::CallRender3DGL& call) {
    auto* rhs_epdc = this->get_data_slot_.CallAs<geocalls::EllipsoidalParticleDataCall>();
    if (rhs_epdc == nullptr)
        return false;

    rhs_epdc->SetFrameID(static_cast<int>(call.Time()));
    if (!(*rhs_epdc)(1))
        return false;

    rhs_epdc->SetFrameID(static_cast<int>(call.Time()));
    if (!(*rhs_epdc)(0))
        return false;

    if (!this->validateData(rhs_epdc))
        return false;

    auto* rhs_tfc = this->get_tf_slot_.CallAs<mmstd_gl::CallGetTransferFunctionGL>();
    auto* rhs_flagsc = this->read_flags_slot_.CallAs<mmstd_gl::FlagCallRead_GL>();
    bool use_flags = (rhs_flagsc != nullptr);

    bool use_clip = false;
    auto rhs_clipc = this->get_clip_plane_slot_.CallAs<view::CallClipPlane>();
    if (rhs_clipc != nullptr && (*rhs_clipc)(0)) {
        use_clip = true;
    }

    bool use_per_axis_color = false;
    switch (this->color_mode_param_.Param<core::param::EnumParam>()->Value()) {
    case 0:
        use_per_axis_color = false;
        break;
    case 1:
        use_per_axis_color = true;
        break;
    default:
        break;
    }

    view::Camera cam = call.GetCamera();
    auto view_temp = cam.getViewMatrix();
    auto proj_temp = cam.getProjectionMatrix();
    auto cam_pos = cam.get<view::Camera::Pose>().position;

    // todo...
    //glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    int w = call.GetFramebuffer()->getWidth();
    int h = call.GetFramebuffer()->getHeight();
    glm::vec4 viewport_stuff;
    viewport_stuff[0] = 0.0f;
    viewport_stuff[1] = 0.0f;
    viewport_stuff[2] = static_cast<float>(w);
    viewport_stuff[3] = static_cast<float>(h);
    if (viewport_stuff[2] < 1.0f)
        viewport_stuff[2] = 1.0f;
    if (viewport_stuff[3] < 1.0f)
        viewport_stuff[3] = 1.0f;
    viewport_stuff[2] = 2.0f / viewport_stuff[2];
    viewport_stuff[3] = 2.0f / viewport_stuff[3];

    glm::mat4 mv_matrix = view_temp, p_matrix = proj_temp;
    glm::mat4 mvp_matrix = p_matrix * mv_matrix;
    glm::mat4 mvp_matrix_i = glm::inverse(mvp_matrix);
    glm::mat4 mv_matrix_i = glm::inverse(mv_matrix);

    std::shared_ptr<glowl::GLSLProgram> shader;
    switch (this->glyph_param_.Param<core::param::EnumParam>()->Value()) {
    case Glyph::BOX:
        shader = this->box_prgm_;
        orientation_param_.Param<core::param::EnumParam>()->SetGUIVisible(false);
        superquadric_exponent_param_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        gizmo_arrow_thickness_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        break;
    case Glyph::ELLIPSOID:
        shader = this->ellipsoid_prgm_;
        orientation_param_.Param<core::param::EnumParam>()->SetGUIVisible(false);
        superquadric_exponent_param_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        gizmo_arrow_thickness_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        break;
    case Glyph::ARROW:
        shader = this->arrow_prgm_;
        orientation_param_.Param<core::param::EnumParam>()->SetGUIVisible(true);
        superquadric_exponent_param_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        gizmo_arrow_thickness_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        break;
    case Glyph::SUPERQUADRIC:
        shader = this->superquadric_prgm_;
        orientation_param_.Param<core::param::EnumParam>()->SetGUIVisible(false);
        superquadric_exponent_param_.Param<core::param::FloatParam>()->SetGUIVisible(true);
        gizmo_arrow_thickness_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        break;
    case Glyph::GIZMO_ARROWGLYPH:
        shader = this->gizmo_arrowglyph_prgm_;
        orientation_param_.Param<core::param::EnumParam>()->SetGUIVisible(false);
        superquadric_exponent_param_.Param<core::param::FloatParam>()->SetGUIVisible(false);
        gizmo_arrow_thickness_.Param<core::param::FloatParam>()->SetGUIVisible(true);
        break;
    default:;
        shader = this->ellipsoid_prgm_;
    }
    shader->use();

    glUniform4fv(shader->getUniformLocation("view_attr"), 1, glm::value_ptr(viewport_stuff));
    glUniformMatrix4fv(shader->getUniformLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp_matrix));
    glUniformMatrix4fv(shader->getUniformLocation("mv_i"), 1, GL_FALSE, glm::value_ptr(mv_matrix_i));
    glUniformMatrix4fv(shader->getUniformLocation("mvp_t"), 1, GL_TRUE, glm::value_ptr(mvp_matrix));
    glUniformMatrix4fv(shader->getUniformLocation("mvp_i"), 1, GL_FALSE, glm::value_ptr(mvp_matrix_i));
    glUniform4fv(shader->getUniformLocation("cam"), 1, glm::value_ptr(cam_pos));
    glUniform1f(shader->getUniformLocation("scaling"), this->scale_param_.Param<param::FloatParam>()->Value());
    glUniform1f(
        shader->getUniformLocation("radius_scaling"), this->radius_scale_param_.Param<param::FloatParam>()->Value());
    glUniform1i(shader->getUniformLocation("orientation"), this->orientation_param_.Param<param::EnumParam>()->Value());
    glUniform1f(shader->getUniformLocation("min_radius"), this->min_radius_param_.Param<param::FloatParam>()->Value());
    glUniform1f(shader->getUniformLocation("color_interpolation"),
        this->color_interpolation_param_.Param<param::FloatParam>()->Value());
    glUniform1f(
        shader->getUniformLocation("length_filter"), this->length_filter_param_.Param<param::FloatParam>()->Value());
    glUniform1f(
        shader->getUniformLocation("exponent"), this->superquadric_exponent_param_.Param<param::FloatParam>()->Value());


    uint32_t num_total_glyphs = 0;
    uint32_t curr_glyph_offset = 0;
    for (unsigned int i = 0; i < rhs_epdc->GetParticleListCount(); i++) {
        num_total_glyphs += rhs_epdc->AccessParticles(i).GetCount();
    }

    unsigned int fal = 0;
    if (use_flags || use_clip) {
        glEnable(GL_CLIP_DISTANCE0);
    }
    if (use_flags) {
        (*rhs_flagsc)(mmstd_gl::FlagCallRead_GL::CallGetData);
        auto flags = rhs_flagsc->getData();
        //if (flags->flags->getByteSize() / sizeof(core::FlagStorage::FlagVectorType) < num_total_glyphs) {
        //    megamol::core::utility::log::Log::DefaultLog.WriteError("Not enough flags in storage for proper selection!");
        //    return false;
        //}
        flags->validateFlagCount(num_total_glyphs);

        // TODO HAZARD BUG this is not in sync with the buffer arrays for all other attributes and a design flaw of the
        // flag storage!!!!
        flags->flags->bindBase(GL_SHADER_STORAGE_BUFFER, 4);
        //glBindBufferRange(
        //    GL_SHADER_STORAGE_BUFFER, 4, this->flags_buffer.GetHandle(0), 0, num_total_glyphs * sizeof(GLuint));
    }
    glUniform4f(shader->getUniformLocation("flag_selected_col"), 1.f, 0.f, 0.f, 1.f);
    glUniform4f(shader->getUniformLocation("flag_softselected_col"), 1.f, 1.f, 0.f, 1.f);

    if (use_clip) {
        auto clip_point_coords = rhs_clipc->GetPlane().Point();
        auto clip_normal_coords = rhs_clipc->GetPlane().Normal();
        glm::vec3 pt(clip_point_coords.X(), clip_point_coords.Y(), clip_point_coords.Z());
        glm::vec3 nr(clip_normal_coords.X(), clip_normal_coords.Y(), clip_normal_coords.Z());

        std::array<float, 4> clip_data = {rhs_clipc->GetPlane().Normal().X(), rhs_clipc->GetPlane().Normal().Y(),
            rhs_clipc->GetPlane().Normal().Z(), -glm::dot(pt, nr)};

        glUniform4fv(shader->getUniformLocation("clip_data"), 1, clip_data.data());
        auto c = rhs_clipc->GetColour();
        /*glUniform4f(shader->ParameterLocation("clip_color"), static_cast<float>(c[0]) / 255.f,
            static_cast<float>(c[1]) / 255.f, static_cast<float>(c[2]) / 255.f, static_cast<float>(c[3]) / 255.f);*/
    }

    for (unsigned int i = 0; i < rhs_epdc->GetParticleListCount(); i++) {

        auto& el_parts = rhs_epdc->AccessParticles(i);

        if (el_parts.GetCount() == 0 || el_parts.GetQuatData() == nullptr || el_parts.GetRadiiData() == nullptr) {
            curr_glyph_offset += el_parts.GetCount();
            continue;
        }

        uint32_t options = 0;

        bool bind_color = true;
        switch (el_parts.GetColourDataType()) {
        case geocalls::EllipsoidalParticleDataCall::Particles::COLDATA_NONE: {
            options = GlyphOptions::USE_GLOBAL;
            const auto gc = el_parts.GetGlobalColour();
            const std::array<float, 4> gcf = {static_cast<float>(gc[0]) / 255.0f, static_cast<float>(gc[1]) / 255.0f,
                static_cast<float>(gc[2]) / 255.0f, static_cast<float>(gc[3]) / 255.0f};
            glUniform4fv(shader->getUniformLocation("global_color"), 1, gcf.data());
            bind_color = false;
        } break;
        case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
            curr_glyph_offset += el_parts.GetCount();
            continue;
            break;
        case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
        case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
        case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
            // these should have been converted to vec4 colors
            options = 0;
            break;
        case geocalls::SimpleSphericalParticles::COLDATA_DOUBLE_I:
        case geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I:
            // these should have been converted to vec4 colors with only a red channel for I
            options = GlyphOptions::USE_TRANSFER_FUNCTION;
            glActiveTexture(GL_TEXTURE0);
            if (rhs_tfc == nullptr) {
                glBindTexture(GL_TEXTURE_1D, this->grey_tf_);
                glUniform2f(shader->getUniformLocation("tf_range"), el_parts.GetMinColourIndexValue(),
                    el_parts.GetMaxColourIndexValue());
            } else if ((*rhs_tfc)(0)) {
                glBindTexture(GL_TEXTURE_1D, rhs_tfc->OpenGLTexture());
                glUniform2fv(shader->getUniformLocation("tf_range"), 1, rhs_tfc->Range().data());
                //glUniform2f(shader->ParameterLocation("tf_range"), elParts.GetMinColourIndexValue(),
                //    elParts.GetMaxColourIndexValue());
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "GlyphRenderer: could not retrieve transfer function!");
                return false;
            }
            glUniform1i(shader->getUniformLocation("tf_texture"), 0);
            break;
        case geocalls::SimpleSphericalParticles::COLDATA_USHORT_RGBA:
            // we should never get this far
            curr_glyph_offset += el_parts.GetCount();
            continue;
            break;
        default:
            curr_glyph_offset += el_parts.GetCount();
            continue;
            break;
        }
        if (use_flags)
            options = options | GlyphOptions::USE_FLAGS;
        if (use_clip)
            options = options | GlyphOptions::USE_CLIP;
        if (use_per_axis_color)
            options = options | GlyphOptions::USE_PER_AXIS;
        glUniform1ui(shader->getUniformLocation("options"), options);

        switch (el_parts.GetVertexDataType()) {
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
            curr_glyph_offset += el_parts.GetCount();
            continue;
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            // anything to do...?
            break;
        case geocalls::EllipsoidalParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
            curr_glyph_offset += el_parts.GetCount();
            continue;
            break;
        default:
            continue;
        }

        auto& the_pos = position_buffers_[i];
        auto& the_quat = direction_buffers_[i];
        auto& the_rad = radius_buffers_[i];
        auto& the_col = color_buffers_[i];

        const auto num_chunks = the_pos.GetNumChunks();
        for (GLuint x = 0; x < num_chunks; ++x) {
            const auto actual_items = the_pos.GetNumItems(x);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, the_pos.GetHandle(x));
            glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, the_pos.GetHandle(x), 0, actual_items * sizeof(float) * 3);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, the_quat.GetHandle(x));
            glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, the_quat.GetHandle(x), 0, actual_items * sizeof(float) * 4);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, the_rad.GetHandle(x));
            glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, the_rad.GetHandle(x), 0, actual_items * sizeof(float) * 3);

            if (bind_color) {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, the_col.GetHandle(x));
                glBindBufferRange(
                    GL_SHADER_STORAGE_BUFFER, 3, the_col.GetHandle(x), 0, actual_items * sizeof(float) * 4);
            }

            if (use_flags) {
                glUniform1ui(shader->getUniformLocation("flag_offset"), curr_glyph_offset);
            }

#ifdef MEGAMOL_USE_PROFILING
            perf_manager_->start_timer(timers_[0], this->GetCoreInstance()->GetFrameID());
#endif

            switch (this->glyph_param_.Param<core::param::EnumParam>()->Value()) {
            case Glyph::BOX:
                // https://stackoverflow.com/questions/28375338/cube-using-single-gl-triangle-strip
                //glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 14, static_cast<GLsizei>(actualItems));
                // but just drawing the front-facing triangles, that's better
                glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(actual_items) * 3);
                break;
            case Glyph::ELLIPSOID:
                glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(actual_items) * 3);
                break;
            case Glyph::ARROW:
                glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(actual_items) * 3);
                break;
            case Glyph::SUPERQUADRIC:
                glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(actual_items) * 3);
                break;
            case Glyph::GIZMO_ARROWGLYPH:
                glUniform1f(shader->getUniformLocation("arrow_thickness"),
                    gizmo_arrow_thickness_.Param<core::param::FloatParam>()->Value());
                glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(actual_items) * 3);
                break;
            default:;
            }

#ifdef MEGAMOL_USE_PROFILING
            perf_manager_->stop_timer(timers_[0]);
#endif

            curr_glyph_offset += el_parts.GetCount();
        }
    }
    rhs_epdc->Unlock();
    glUseProgram(0);

    // todo if you implement selection, write flags back :)
    //
    // todo clean up state
    if (use_clip || use_flags) {
        glDisable(GL_CLIP_DISTANCE0);
    }
    glDisable(GL_DEPTH_TEST);

    return true;
}
