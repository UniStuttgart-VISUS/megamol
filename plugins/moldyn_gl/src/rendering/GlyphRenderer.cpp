/*
 * GlyphRenderer.cpp
 *
 * Copyright (C) 2022 by VISUS (Universitaet Stuttgart)
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
#include "mmcore/view/CallClipPlane.h"
#include "mmcore_gl/flags/FlagCallsGL.h"
#include "mmcore_gl/utility/ShaderSourceFactory.h"
#include "mmcore_gl/view/CallGetTransferFunctionGL.h"
#include "stdafx.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/String.h"
#include "vislib/assert.h"
#include "vislib/math/Quaternion.h"
#include "vislib_gl/graphics/gl/ShaderSource.h"
#include <cstring>
#include <iostream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::moldyn_gl;
using namespace megamol::moldyn_gl::rendering;

//const uint32_t max_ssbo_size = 2 * 1024 * 1024 * 1024;

GlyphRenderer::GlyphRenderer(void)
        : core_gl::view::Renderer3DModuleGL()
        , m_get_data_slot("getData", "The slot to fetch the data")
        , m_get_tf_slot("getTF", "The slot for the transfer function")
        , m_get_clip_plane_slot("getClipPlane", "The slot for the clip plane")
        , m_read_flags_slot("readFlags", "The slot for reading the selection flags")
        , m_glyph_param("glyph", "Which glyph to render")
        , m_scale_param("scaling", "TODO: scales the box??")
        , m_radius_scale_param("radius_scaling", "scales the glyph radii")
        , m_orientation_param("Orientation", "Selects along which axis the arrows are aligned")
        , m_length_filter_param("lengthFilter", "Filters the arrows by length")
        , m_color_interpolation_param(
              "colorInterpolation", "Interpolate between directional coloring (0) and glyph color (1)")
        , m_min_radius_param("minRadius", "Sets the minimum radius length. Applied to each axis.")
        , m_color_mode_param("colorMode", "Switch between global glyph and per axis color")
        , m_superquadric_exponent_param("exponent", "Sets the exponent used in the implicit superquadric equation") {

    this->m_get_data_slot.SetCompatibleCall<geocalls::EllipsoidalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->m_get_data_slot);

    this->m_get_tf_slot.SetCompatibleCall<core_gl::view::CallGetTransferFunctionGLDescription>();
    this->MakeSlotAvailable(&this->m_get_tf_slot);

    this->m_get_clip_plane_slot.SetCompatibleCall<view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->m_get_clip_plane_slot);

    this->m_read_flags_slot.SetCompatibleCall<core_gl::FlagCallRead_GLDescription>();
    this->MakeSlotAvailable(&this->m_read_flags_slot);

    param::EnumParam* gp = new param::EnumParam(0);
    gp->SetTypePair(Glyph::BOX, "Box");
    gp->SetTypePair(Glyph::ELLIPSOID, "Ellipsoid");
    gp->SetTypePair(Glyph::ARROW, "Arrow");
    gp->SetTypePair(Glyph::SUPERQUADRIC, "Superquadric");
    gp->SetTypePair(Glyph::GIZMO_ARROWGLYPH, "Gizmo");
    gp->SetTypePair(Glyph::GIZMO_LINE, "Line");
    this->m_glyph_param << gp;
    this->MakeSlotAvailable(&this->m_glyph_param);

    m_scale_param << new param::FloatParam(1.0f, 0.0f, 100.0f);
    this->MakeSlotAvailable(&this->m_scale_param);

    m_radius_scale_param << new param::FloatParam(1.0f, 0.0f, 100.0f, 0.1f);
    this->MakeSlotAvailable(&this->m_radius_scale_param);

    // currently only needed for arrow
    param::EnumParam* op = new param::EnumParam(3);
    op->SetTypePair(0, "x");
    op->SetTypePair(1, "y");
    op->SetTypePair(2, "z");
    op->SetTypePair(3, "largest radius");
    this->m_orientation_param << op;
    this->MakeSlotAvailable(&this->m_orientation_param);

    m_length_filter_param << new param::FloatParam(0.0f, 0.0f);
    this->MakeSlotAvailable(&this->m_length_filter_param);

    m_color_interpolation_param << new param::FloatParam(1.0f, 0.0, 1.0f);
    this->MakeSlotAvailable(&this->m_color_interpolation_param);

    m_min_radius_param << new param::FloatParam(0.1f, 0.f, 2.f, 0.01f);
    this->MakeSlotAvailable(&this->m_min_radius_param);

    param::EnumParam* gcm = new param::EnumParam(0);
    gcm->SetTypePair(0, "GlyphGlobal");
    gcm->SetTypePair(1, "PerAxis");
    this->m_color_mode_param << gcm;
    this->MakeSlotAvailable(&this->m_color_mode_param);

    m_superquadric_exponent_param << new param::FloatParam(1.0f, -100.0f, 100.0f, 0.1f);
    this->MakeSlotAvailable(&this->m_superquadric_exponent_param);
}


GlyphRenderer::~GlyphRenderer(void) {
    this->Release();
}


bool GlyphRenderer::create(void) {
    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    if (!ogl_ctx.areExtAvailable(vislib_gl::graphics::gl::GLSLShader::RequiredExtensions()))
        return false;

    bool ret_val = true;
    ret_val = ret_val && this->makeShader("glyph::box_vertex", "glyph::box_fragment", this->m_box_shader);
    ret_val = ret_val && this->makeShader("glyph::ellipsoid_vertex", "glyph::ellipsoid_fragment", this->m_ellipsoid_shader);
    ret_val = ret_val && this->makeShader("glyph::arrow_vertex", "glyph::arrow_fragment", this->m_arrow_shader);
    ret_val = ret_val && this->makeShader("glyph::superquadric_vertex", "glyph::superquadric_fragment", this->m_superquadric_shader);
    ret_val = ret_val && this->makeShader("glyph::gizmo_arrowglyph_vertex", "glyph::gizmo_arrowglyph_fragment", this->m_gizmo_arrowglyph_shader);
    ret_val = ret_val && this->makeShader("glyph::gizmo_line_vertex", "glyph::gizmo_line_fragment", this->m_gizmo_line_shader);

    glEnable(GL_TEXTURE_1D);
    glGenTextures(1, &this->m_grey_tf);
    unsigned char tex[6] = {0, 0, 0, 255, 255, 255};
    glBindTexture(GL_TEXTURE_1D, this->m_grey_tf);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glBindTexture(GL_TEXTURE_1D, 0);
    glDisable(GL_TEXTURE_1D);

    return ret_val;
}

bool GlyphRenderer::makeShader(
    std::string vertex_name, std::string fragment_name, vislib_gl::graphics::gl::GLSLShader& shader) {
    using namespace megamol::core::utility::log;
    using namespace vislib_gl::graphics::gl;

    ShaderSource vert_src;
    ShaderSource frag_src;
    auto ssf = std::make_shared<core_gl::utility::ShaderSourceFactory>(instance()->Configuration().ShaderDirectories());
    if (!ssf->MakeShaderSource(vertex_name.c_str(), vert_src)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "GlyphRenderer: unable to load vertex shader source: %s", vertex_name.c_str());
        return false;
    }
    if (!ssf->MakeShaderSource(fragment_name.c_str(), frag_src)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "GlyphRenderer: unable to load fragment shader source: %s", fragment_name.c_str());
        return false;
    }
    try {
        if (!shader.Create(vert_src.Code(), vert_src.Count(), frag_src.Code(), frag_src.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
                "GlyphRenderer: unable to compile shader: unknown error\n");
            return false;
        }
    } catch (AbstractOpenGLShader::CompileException& ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "GlyphRenderer: unable to compile shader (@%s): %s\n",
            vislib_gl::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "GlyphRenderer: unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "GlyphRenderer: unable to compile shader: unknown exception\n");
        return false;
    }
    return true;
}

bool GlyphRenderer::GetExtents(core_gl::view::CallRender3DGL& call) {

    auto* rhs_epdc = this->m_get_data_slot.CallAs<geocalls::EllipsoidalParticleDataCall>();
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
    // TODO: maybe use smart_ptr instead
    // TODO: replace old shaderfactory with new sf
    this->m_box_shader.Release();
    this->m_ellipsoid_shader.Release();
    this->m_arrow_shader.Release();
    this->m_superquadric_shader.Release();
    this->m_gizmo_arrowglyph_shader.Release();
    this->m_gizmo_line_shader.Release();
    glDeleteTextures(1, &this->m_grey_tf);
}

bool megamol::moldyn_gl::rendering::GlyphRenderer::validateData(geocalls::EllipsoidalParticleDataCall* rhs_edc) {

    if (this->m_last_hash != rhs_edc->DataHash() || this->m_last_frame_id != rhs_edc->FrameID()) {
        this->m_position_buffers.reserve(rhs_edc->GetParticleListCount());
        this->m_radius_buffers.reserve(rhs_edc->GetParticleListCount());
        this->m_direction_buffers.reserve(rhs_edc->GetParticleListCount());
        this->m_color_buffers.reserve(rhs_edc->GetParticleListCount());
        
        for (uint32_t x = 0; x < rhs_edc->GetParticleListCount(); ++x) {
            auto& l = rhs_edc->AccessParticles(x);
            this->m_position_buffers.emplace_back(utility::SSBOBufferArray("position_buffer" + std::to_string(x)));
            this->m_radius_buffers.emplace_back(utility::SSBOBufferArray("radius_buffer" + std::to_string(x)));
            this->m_direction_buffers.emplace_back(utility::SSBOBufferArray("direction_buffer" + std::to_string(x)));
            this->m_color_buffers.emplace_back(utility::SSBOBufferArray("color_buffer" + std::to_string(x)));

            this->m_direction_buffers[x].SetData(l.GetQuatData(), l.GetQuatDataStride(), 4 * sizeof(float), l.GetCount(),
                [](void* dst, const void* src) { memcpy(dst, src, sizeof(float) * 4); });
            const auto num_items_per_chunk = this->m_direction_buffers[x].GetMaxNumItemsPerChunk();

            switch (l.GetVertexDataType()) {
            case geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ:
            case geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR:
                // radius is skipped automatically by memcpy width
                this->m_position_buffers[x].SetDataWithItems(l.GetVertexData(), l.GetVertexDataStride(),
                    3 * sizeof(float), l.GetCount(), num_items_per_chunk,
                    [](void* dst, const void* src) { memcpy(dst, src, sizeof(float) * 3); });
                break;
            case geocalls::SimpleSphericalParticles::VERTDATA_DOUBLE_XYZ:
                // narrow double to float
                this->m_position_buffers[x].SetDataWithItems(l.GetVertexData(), l.GetVertexDataStride(),
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
                this->m_color_buffers[x].SetDataWithItems(l.GetColourData(), l.GetColourDataStride(), 4 * sizeof(float),
                    l.GetCount(), num_items_per_chunk, [](void* dst, const void* src) {
                        auto* d = static_cast<float*>(dst);
                        const auto* s = static_cast<const unsigned char*>(src);
                        for (auto i = 0; i < 4; ++i) {
                            *(d + i) = static_cast<float>(*(s + i)) / 255.0f;
                        }
                    });
                break;
            case geocalls::SimpleSphericalParticles::COLDATA_FLOAT_RGB:
                this->m_color_buffers[x].SetDataWithItems(l.GetColourData(), l.GetColourDataStride(), 4 * sizeof(float),
                    l.GetCount(), num_items_per_chunk, [](void* dst, const void* src) {
                        memcpy(dst, src, sizeof(float) * 3);
                        auto* d = static_cast<float*>(dst);
                        *(d + 3) = 1.0f;
                    });
                break;
            case geocalls::SimpleSphericalParticles::COLDATA_FLOAT_RGBA:
                // this is paranoid and could be avoided for cases where data is NOT interleaved
                this->m_color_buffers[x].SetDataWithItems(l.GetColourData(), l.GetColourDataStride(), 4 * sizeof(float),
                    l.GetCount(), num_items_per_chunk,
                    [](void* dst, const void* src) { memcpy(dst, src, sizeof(float) * 4); });
                break;
            case geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I:
                this->m_color_buffers[x].SetDataWithItems(l.GetColourData(), l.GetColourDataStride(), 4 * sizeof(float),
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
                this->m_color_buffers[x].SetDataWithItems(l.GetColourData(), l.GetColourDataStride(), 4 * sizeof(float),
                    l.GetCount(), num_items_per_chunk, [](void* dst, const void* src) {
                        auto* d = static_cast<float*>(dst);
                        const auto* s = static_cast<const double*>(src);
                        *d = static_cast<float>(*s);
                        // rest is garbage
                    });
                break;
            default:;
            }

            this->m_radius_buffers[x].SetDataWithItems(l.GetRadiiData(), l.GetRadiiDataStride(), sizeof(float) * 3,
                l.GetCount(), num_items_per_chunk,
                [](void* dst, const void* src) { memcpy(dst, src, sizeof(float) * 3); });
        }
        this->m_last_hash = rhs_edc->DataHash();
        this->m_last_frame_id = rhs_edc->FrameID();
    }
    return true;
}

bool GlyphRenderer::Render(core_gl::view::CallRender3DGL& call) {
    auto* rhs_epdc = this->m_get_data_slot.CallAs<geocalls::EllipsoidalParticleDataCall>();
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

    auto* rhs_tfc = this->m_get_tf_slot.CallAs<core_gl::view::CallGetTransferFunctionGL>();
    auto* rhs_flagsc = this->m_read_flags_slot.CallAs<core_gl::FlagCallRead_GL>();
    bool use_flags = (rhs_flagsc != nullptr);

    bool use_clip = false;
    auto rhs_clipc = this->m_get_clip_plane_slot.CallAs<view::CallClipPlane>();
    if (rhs_clipc != nullptr && (*rhs_clipc)(0)) {
        use_clip = true;
    }

    bool use_per_axis_color = false;
    switch (this->m_color_mode_param.Param<core::param::EnumParam>()->Value()) {
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

    vislib_gl::graphics::gl::GLSLShader* shader;
    switch (this->m_glyph_param.Param<core::param::EnumParam>()->Value()) {
    case Glyph::BOX:
        shader = &this->m_box_shader;
        m_orientation_param.Param<core::param::EnumParam>()->SetGUIVisible(false);
        m_superquadric_exponent_param.Param<core::param::FloatParam>()->SetGUIVisible(false);
        break;
    case Glyph::ELLIPSOID:
        shader = &this->m_ellipsoid_shader;
        m_orientation_param.Param<core::param::EnumParam>()->SetGUIVisible(false);
        m_superquadric_exponent_param.Param<core::param::FloatParam>()->SetGUIVisible(false);
        break;
    case Glyph::ARROW:
        shader = &this->m_arrow_shader;
        m_orientation_param.Param<core::param::EnumParam>()->SetGUIVisible(true);
        m_superquadric_exponent_param.Param<core::param::FloatParam>()->SetGUIVisible(false);
        break;
    case Glyph::SUPERQUADRIC:
        shader = &this->m_superquadric_shader;
        m_orientation_param.Param<core::param::EnumParam>()->SetGUIVisible(false);
        m_superquadric_exponent_param.Param<core::param::FloatParam>()->SetGUIVisible(true);
        break;
    case Glyph::GIZMO_ARROWGLYPH:
        shader = &this->m_gizmo_arrowglyph_shader;
        m_orientation_param.Param<core::param::EnumParam>()->SetGUIVisible(false);
        m_superquadric_exponent_param.Param<core::param::FloatParam>()->SetGUIVisible(false);
        break;
    case Glyph::GIZMO_LINE:
        shader = &this->m_gizmo_line_shader;
        m_orientation_param.Param<core::param::EnumParam>()->SetGUIVisible(false);
        m_superquadric_exponent_param.Param<core::param::FloatParam>()->SetGUIVisible(false);
        break;
    default:;
        shader = &this->m_ellipsoid_shader;
    }
    shader->Enable();

    glUniform4fv(shader->ParameterLocation("view_attr"), 1, glm::value_ptr(viewport_stuff));
    glUniformMatrix4fv(shader->ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp_matrix));
    glUniformMatrix4fv(shader->ParameterLocation("mv_i"), 1, GL_FALSE, glm::value_ptr(mv_matrix_i));
    glUniformMatrix4fv(shader->ParameterLocation("mvp_t"), 1, GL_TRUE, glm::value_ptr(mvp_matrix));
    glUniformMatrix4fv(shader->ParameterLocation("mvp_i"), 1, GL_FALSE, glm::value_ptr(mvp_matrix_i));
    glUniform4fv(shader->ParameterLocation("cam"), 1, glm::value_ptr(cam_pos));
    glUniform1f(shader->ParameterLocation("scaling"), this->m_scale_param.Param<param::FloatParam>()->Value());
    glUniform1f(shader->ParameterLocation("radius_scaling"), this->m_radius_scale_param.Param<param::FloatParam>()->Value());
    glUniform1i(shader->ParameterLocation("orientation"), this->m_orientation_param.Param<param::EnumParam>()->Value());
    glUniform1f(shader->ParameterLocation("min_radius"), this->m_min_radius_param.Param<param::FloatParam>()->Value());
    glUniform1f(shader->ParameterLocation("color_interpolation"),
        this->m_color_interpolation_param.Param<param::FloatParam>()->Value());
    glUniform1f(shader->ParameterLocation("length_filter"), this->m_length_filter_param.Param<param::FloatParam>()->Value());
    glUniform1f(
        shader->ParameterLocation("exponent"), this->m_superquadric_exponent_param.Param<param::FloatParam>()->Value());


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
        (*rhs_flagsc)(core_gl::FlagCallRead_GL::CallGetData);
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
    glUniform4f(shader->ParameterLocation("flag_selected_col"), 1.f, 0.f, 0.f, 1.f);
    glUniform4f(shader->ParameterLocation("flag_softselected_col"), 1.f, 1.f, 0.f, 1.f);

    if (use_clip) {
        auto clip_point_coords = rhs_clipc->GetPlane().Point();
        auto clip_normal_coords = rhs_clipc->GetPlane().Normal();
        glm::vec3 pt(clip_point_coords.X(), clip_point_coords.Y(), clip_point_coords.Z());
        glm::vec3 nr(clip_normal_coords.X(), clip_normal_coords.Y(), clip_normal_coords.Z());

        std::array<float, 4> clip_data = {rhs_clipc->GetPlane().Normal().X(), rhs_clipc->GetPlane().Normal().Y(),
            rhs_clipc->GetPlane().Normal().Z(), -glm::dot(pt, nr)};

        glUniform4fv(shader->ParameterLocation("clip_data"), 1, clip_data.data());
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
            glUniform4fv(shader->ParameterLocation("global_color"), 1, gcf.data());
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
                glBindTexture(GL_TEXTURE_1D, this->m_grey_tf);
                glUniform2f(shader->ParameterLocation("tf_range"), el_parts.GetMinColourIndexValue(),
                    el_parts.GetMaxColourIndexValue());
            } else if ((*rhs_tfc)(0)) {
                glBindTexture(GL_TEXTURE_1D, rhs_tfc->OpenGLTexture());
                glUniform2fv(shader->ParameterLocation("tf_range"), 1, rhs_tfc->Range().data());
                //glUniform2f(shader->ParameterLocation("tf_range"), elParts.GetMinColourIndexValue(),
                //    elParts.GetMaxColourIndexValue());
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "GlyphRenderer: could not retrieve transfer function!");
                return false;
            }
            glUniform1i(shader->ParameterLocation("tf_texture"), 0);
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
        glUniform1ui(shader->ParameterLocation("options"), options);

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

        auto& the_pos = m_position_buffers[i];
        auto& the_quat = m_direction_buffers[i];
        auto& the_rad = m_radius_buffers[i];
        auto& the_col = m_color_buffers[i];

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
                glUniform1ui(shader->ParameterLocation("flag_offset"), curr_glyph_offset);
            }

            switch (this->m_glyph_param.Param<core::param::EnumParam>()->Value()) {
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
                glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(actual_items) * 3);
                break;
            case Glyph::GIZMO_LINE:
                glDrawArraysInstanced(GL_LINES, 0, 2, static_cast<GLsizei>(actual_items) * 3);
                break;
            default:;
            }
            curr_glyph_offset += el_parts.GetCount();
        }
    }
    rhs_epdc->Unlock();
    shader->Disable();

    // todo if you implement selection, write flags back :)
    //
    // todo clean up state
    if (use_clip || use_flags) {
        glDisable(GL_CLIP_DISTANCE0);
    }
    glDisable(GL_DEPTH_TEST);

    return true;
}
