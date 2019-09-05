/*
 * EllipsoidRenderer.cpp
 *
 * Copyright (C) 2008-2015 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GlyphRenderer.h"
#include <cstring>
#include <iostream>
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "inttypes.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/FlagCall.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/String.h"
#include "vislib/assert.h"
#include "vislib/math/Quaternion.h"
#include "vislib/sys/Log.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::stdplugin;
using namespace megamol::stdplugin::moldyn;
using namespace megamol::stdplugin::moldyn::rendering;

const uint32_t max_ssbo_size = 2 * 1024 * 1024 * 1024;

GlyphRenderer::GlyphRenderer(void)
    : Renderer3DModule_2()
    , getDataSlot("getData", "The slot to fetch the data")
    , glyphParam("glyph", "which glyph to render")
    , colorInterpolationParam("colorInterpolation", "interpolate between directional coloring (0) and glyph color (1)") {

    this->getDataSlot.SetCompatibleCall<core::moldyn::EllipsoidalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    param::EnumParam* gp = new param::EnumParam(0);
    gp->SetTypePair(Glyph::BOX, "Box");
    gp->SetTypePair(Glyph::ELLIPSOID, "Ellipsoid");
    gp->SetTypePair(Glyph::ARROW, "Arrow");
    gp->SetTypePair(Glyph::SUPERQUADRIC, "Superquadric");
    this->glyphParam  << gp;
    this->MakeSlotAvailable(&this->glyphParam);

    colorInterpolationParam << new param::FloatParam(1.0f, 0.0, 1.0f);
    this->MakeSlotAvailable(&this->colorInterpolationParam);
}


GlyphRenderer::~GlyphRenderer(void) { this->Release(); }


bool GlyphRenderer::create(void) {

    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) return false;

    bool retVal = true;
    retVal = retVal && this->makeShader("glyph::ellipsoid_vertex", "glyph::ellipsoid_fragment", this->ellipsoidShader);
    retVal = retVal && this->makeShader("glyph::box_vertex", "glyph::box_fragment", this->boxShader);

    return retVal;
}

bool GlyphRenderer::makeShader(std::string vertexName, std::string fragmentName, vislib::graphics::gl::GLSLShader& shader) {
    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    ShaderSource vertSrc;
    ShaderSource fragSrc;

    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(vertexName.c_str(), vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "GlyphRenderer: unable to load vertex shader source: %s", vertexName.c_str());
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(fragmentName.c_str(), fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "GlyphRenderer: unable to load fragment shader source: %s", fragmentName.c_str());
        return false;
    }
    try {
        if (!shader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "GlyphRenderer: unable to compile shader: unknown error\n");
            return false;
        }
    } catch (AbstractOpenGLShader::CompileException& ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "GlyphRenderer: unable to compile shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception& e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "GlyphRenderer: unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "GlyphRenderer: unable to compile shader: unknown exception\n");
        return false;
    }
    return true;
}

bool GlyphRenderer::GetExtents(core::view::CallRender3D_2& call) {

    auto* epdc = this->getDataSlot.CallAs<core::moldyn::EllipsoidalParticleDataCall>();
    if ((epdc != NULL) && ((*epdc)(1))) {
        call.SetTimeFramesCount(epdc->FrameCount());
        call.AccessBoundingBoxes() = epdc->AccessBoundingBoxes();
    } else {
        call.SetTimeFramesCount(1);
        call.AccessBoundingBoxes().Clear();
    }

    return true;
}
void GlyphRenderer::release(void) { this->ellipsoidShader.Release(); }

bool megamol::stdplugin::moldyn::rendering::GlyphRenderer::validateData(
    core::moldyn::EllipsoidalParticleDataCall* edc) {

    if (this->lastHash != edc->DataHash() || this->lastFrameID != edc->FrameID()) {
        this->position_buffers.reserve(edc->GetParticleListCount());
        this->radius_buffers.reserve(edc->GetParticleListCount());
        this->direction_buffers.reserve(edc->GetParticleListCount());
        this->color_buffers.reserve(edc->GetParticleListCount());

        for (uint32_t x = 0; x < edc->GetParticleListCount(); ++x) {
            auto& l = edc->AccessParticles(x);
            this->position_buffers.emplace_back(utility::SSBOBufferArray("position_buffer" + std::to_string(x)));
            this->radius_buffers.emplace_back(utility::SSBOBufferArray("radius_buffer" + std::to_string(x)));
            this->direction_buffers.emplace_back(utility::SSBOBufferArray("direction_buffer" + std::to_string(x)));
            this->color_buffers.emplace_back(utility::SSBOBufferArray("color_buffer" + std::to_string(x)));

            this->direction_buffers[x].SetDataWithSize(l.GetQuatData(), l.GetQuatDataStride(), 4 * sizeof(float),
                l.GetCount(), max_ssbo_size, [](void* dst, const void* src) { memcpy(dst, src, sizeof(float) * 4); });
            const auto num_items_per_chunk = this->direction_buffers[x].GetMaxNumItemsPerChunk();

            switch (l.GetVertexDataType()) {
            case core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ:
            case core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR:
                // radius is skipped automatically by memcpy width
                this->position_buffers[x].SetDataWithItems(l.GetVertexData(), l.GetVertexDataStride(),
                    3 * sizeof(float), l.GetCount(), num_items_per_chunk,
                    [](void* dst, const void* src) { memcpy(dst, src, sizeof(float) * 3); });
                break;
            case core::moldyn::SimpleSphericalParticles::VERTDATA_DOUBLE_XYZ:
                // narrow double to float
                this->position_buffers[x].SetDataWithItems(l.GetVertexData(), l.GetVertexDataStride(),
                    3 * sizeof(float), l.GetCount(), num_items_per_chunk, [](void* dst, const void* src) {
                        auto* d = static_cast<float*>(dst);
                        const auto* s = static_cast<const double*>(src);
                        for (auto i = 0; i < 3; ++i) {
                            *(d + i) = static_cast<float>(*(s + i));
                        }
                    });
                break;
            case core::moldyn::SimpleSphericalParticles::VERTDATA_SHORT_XYZ:
            case core::moldyn::SimpleSphericalParticles::VERTDATA_NONE:
            default:
                vislib::sys::Log::DefaultLog.WriteError(
                    "GlyphRenderer: no support for vertex data types SHORT_XYZ or NONE");
                continue;
            }

            switch (l.GetColourDataType()) {
            case core::moldyn::SimpleSphericalParticles::COLDATA_NONE:
                break;
            case core::moldyn::SimpleSphericalParticles::COLDATA_UINT8_RGB:
                // we could just pad this, but such datasets need to disappear...
                vislib::sys::Log::DefaultLog.WriteError(
                    "GlyphRenderer: COLDATA_UINT8_RGB is deprecated and unsupported");
                continue;
                break;
            case core::moldyn::SimpleSphericalParticles::COLDATA_UINT8_RGBA:
                // extend to floats
                this->color_buffers[x].SetDataWithItems(l.GetColourData(), l.GetColourDataStride(), 4 * sizeof(float),
                    l.GetCount(), num_items_per_chunk, [](void* dst, const void* src) {
                        auto* d = static_cast<float*>(dst);
                        const auto* s = static_cast<const unsigned char*>(src);
                        for (auto i = 0; i < 4; ++i) {
                            *(d + i) = static_cast<float>(*(s + i)) / 255.0f;
                        }
                    });
                break;
            case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGB:
                this->color_buffers[x].SetDataWithItems(l.GetColourData(), l.GetColourDataStride(), 4 * sizeof(float),
                    l.GetCount(), num_items_per_chunk, [](void* dst, const void* src) {
                        memcpy(dst, src, sizeof(float) * 3);
                        auto* d = static_cast<float*>(dst);
                        *(d + 3) = 1.0f;
                    });
                break;
            case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGBA:
                // this is paranoid and could be avoided for cases where data is NOT interleaved
                this->color_buffers[x].SetDataWithItems(l.GetColourData(), l.GetColourDataStride(), 4 * sizeof(float),
                    l.GetCount(), num_items_per_chunk,
                    [](void* dst, const void* src) { memcpy(dst, src, sizeof(float) * 4); });
                break;
            case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I:
                this->color_buffers[x].SetDataWithItems(l.GetColourData(), l.GetColourDataStride(), 4 * sizeof(float),
                    l.GetCount(), num_items_per_chunk, [](void* dst, const void* src) {
                        auto* d = static_cast<float*>(dst);
                        const auto* s = static_cast<const float*>(src);
                        *d = *s;
                        // rest is garbage
                    });
                break;
            case core::moldyn::SimpleSphericalParticles::COLDATA_USHORT_RGBA:
                vislib::sys::Log::DefaultLog.WriteError("GlyphRenderer: COLDATA_USHORT_RGBA is unsupported");
                continue;
                break;
            case core::moldyn::SimpleSphericalParticles::COLDATA_DOUBLE_I:
                this->color_buffers[x].SetDataWithItems(l.GetColourData(), l.GetColourDataStride(), 4 * sizeof(float),
                    l.GetCount(), num_items_per_chunk, [](void* dst, const void* src) {
                        auto* d = static_cast<float*>(dst);
                        const auto* s = static_cast<const double*>(src);
                        *d = static_cast<float>(*s);
                        // rest is garbage
                    });
                break;
            default:;
            }

            this->radius_buffers[x].SetDataWithItems(l.GetRadiiData(), l.GetRadiiDataStride(), sizeof(float) * 3,
                l.GetCount(), num_items_per_chunk,
                [](void* dst, const void* src) { memcpy(dst, src, sizeof(float) * 3); });
        }
    }
    return true;
}

bool GlyphRenderer::Render(core::view::CallRender3D_2& call) {
    auto* epdc = this->getDataSlot.CallAs<core::moldyn::EllipsoidalParticleDataCall>();
    if (epdc == nullptr) return false;

    epdc->SetFrameID(static_cast<int>(call.Time()));
    if (!(*epdc)(1)) return false;

    epdc->SetFrameID(static_cast<int>(call.Time()));
    if (!(*epdc)(0)) return false;

    if (!this->validateData(epdc)) return false;

    view::Camera_2 cam;
    call.GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;

    // Generate complete snapshot and calculate matrices
    cam.calc_matrices(snapshot, viewTemp, projTemp, thecam::snapshot_content::all);

    glm::vec4 CamPos = snapshot.position;
    auto CamView = snapshot.view_vector;
    auto CamRight = snapshot.right_vector;
    auto CamUp = snapshot.up_vector;
    auto CamNearClip = snapshot.frustum_near;
    auto Eye = cam.eye();
    bool rightEye = (Eye == core::thecam::Eye::right);

    // todo...
     glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
    // glDisable(GL_POINT_SPRITE_ARB);
    // glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    // glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    // glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

    // glEnableClientState(GL_VERTEX_ARRAY);
    // glEnableClientState(GL_COLOR_ARRAY);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    //float viewportStuff[4] = {cameraInfo->TileRect().Left(), cameraInfo->TileRect().Bottom(),
    //    cameraInfo->TileRect().Width(), cameraInfo->TileRect().Height()};
    float viewportStuff[4] = {
        cam.image_tile().left(), cam.image_tile().bottom(), cam.image_tile().width(), cam.image_tile().height()};
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glm::mat4 mv_matrix = viewTemp, p_matrix = projTemp;
    glm::mat4 mvp_matrix = p_matrix * mv_matrix;
    glm::mat4 mvp_matrix_i = glm::inverse(mvp_matrix);
    glm::mat4 mv_matrix_i = glm::inverse(mv_matrix);

    this->GetLights();
    glm::vec4 light = {0.0f, 0.0f, 0.0f, 0.0f};
    if (this->lightMap.size() != 1) {
        vislib::sys::Log::DefaultLog.WriteWarn("GlyphRenderer: Only one single directional light source is supported by this renderer");
    } else {
        const auto lightPos = this->lightMap.begin()->second.dl_direction;
        if (lightPos.size() == 3) {
            light[0] = lightPos[0];
            light[1] = lightPos[1];
            light[2] = lightPos[2];
        }
    }

    vislib::graphics::gl::GLSLShader* shader;
    switch (this->glyphParam.Param<core::param::EnumParam>()->Value()) {
    case Glyph::BOX:
        shader = &this->boxShader;
        break;
    case Glyph::ELLIPSOID:
        shader = &this->ellipsoidShader;
        break;
    case Glyph::ARROW:
        shader = &this->ellipsoidShader;
        break;
    case Glyph::SUPERQUADRIC:
        shader = &this->ellipsoidShader;
        break;
    default:;
        shader = &this->ellipsoidShader;
    }
    shader->Enable();

    glUniform4fv(shader->ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniformMatrix4fv(shader->ParameterLocation("MV_I"), 1, GL_FALSE, glm::value_ptr(mv_matrix_i));
    glUniformMatrix4fv(shader->ParameterLocation("MV_T"), 1, GL_TRUE, glm::value_ptr(mv_matrix));
    glUniformMatrix4fv(shader->ParameterLocation("MVP"), 1, GL_FALSE, glm::value_ptr(mvp_matrix));
    glUniformMatrix4fv(shader->ParameterLocation("MVP_T"), 1, GL_TRUE, glm::value_ptr(mvp_matrix));
    glUniformMatrix4fv(shader->ParameterLocation("MVP_I"), 1, GL_FALSE, glm::value_ptr(mvp_matrix_i));
    glUniform4fv(shader->ParameterLocation("light"), 1, glm::value_ptr(light));
    glUniform4fv(shader->ParameterLocation("cam"), 1, glm::value_ptr(CamPos));

    glUniform1f(shader->ParameterLocation("colorInterpolation"),
        this->colorInterpolationParam.Param<param::FloatParam>()->Value());

    //glUniform3fvARB(shader->ParameterLocation("camIn"), 1, glm::value_ptr(CamView));
    //glUniform3fvARB(shader->ParameterLocation("camRight"), 1, glm::value_ptr(CamRight));
    //glUniform3fvARB(shader->ParameterLocation("camUp"), 1, glm::value_ptr(CamUp));
    
    for (unsigned int i = 0; i < epdc->GetParticleListCount(); i++) {

        auto& elParts = epdc->AccessParticles(i);

        if (elParts.GetCount() == 0 || elParts.GetQuatData() == nullptr || elParts.GetRadiiData() == nullptr) continue;

        bool bindColor = true;
        switch (elParts.GetColourDataType()) {
        case core::moldyn::EllipsoidalParticleDataCall::Particles::COLDATA_NONE: {
            glUniform1i(shader->ParameterLocation("useGlobalColor"), 1);
            glUniform1i(shader->ParameterLocation("intensityOnly"), 0);
            const auto gc = elParts.GetGlobalColour();
            const std::array<float, 4> gcf = {static_cast<float>(gc[0]) / 255.0f, static_cast<float>(gc[1]) / 255.0f,
                static_cast<float>(gc[2]) / 255.0f, static_cast<float>(gc[3]) / 255.0f};
            glUniform4fv(shader->ParameterLocation("globalColor"), 1, gcf.data());
            bindColor = false;
        } break;
        case core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
            continue;
            break;
        case core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
        case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
        case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
            // these should have been converted to vec4 colors
            glUniform1i(shader->ParameterLocation("useGlobalColor"), 0);
            glUniform1i(shader->ParameterLocation("intensityOnly"), 0);
            break;
        case core::moldyn::SimpleSphericalParticles::COLDATA_DOUBLE_I:
        case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I:
            // these should have been converted to vec4 colors with only a red channel for I
            glUniform1i(shader->ParameterLocation("useGlobalColor"), 0);
            glUniform1i(shader->ParameterLocation("intensityOnly"), 1);
            break;
        case core::moldyn::SimpleSphericalParticles::COLDATA_USHORT_RGBA:
            // we should never get this far
            continue;
            break;
        default:
            continue;
            break;
        }

        switch (elParts.GetVertexDataType()) {
        case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE:
            continue;
        case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            // anything to do...?
            break;
        case core::moldyn::EllipsoidalParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
            continue;
            break;
        default:
            continue;
        }

        auto& the_pos = position_buffers[i];
        auto& the_quat = direction_buffers[i];
        auto& the_rad = radius_buffers[i];
        auto& the_col = color_buffers[i];
        // TODO flags

        const auto numChunks = the_pos.GetNumChunks();
        for (GLuint x = 0; x < numChunks; ++x) {
            const auto actualItems = the_pos.GetNumItems(x);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, the_pos.GetHandle(x));
            glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, the_pos.GetHandle(x), 0, actualItems * sizeof(float) * 3);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, the_quat.GetHandle(x));
            glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, the_quat.GetHandle(x), 0, actualItems * sizeof(float) * 4);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, the_rad.GetHandle(x));
            glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, the_rad.GetHandle(x), 0, actualItems * sizeof(float) * 3);

            if (bindColor) {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, the_col.GetHandle(x));
                glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 3, the_col.GetHandle(x), 0, actualItems * sizeof(float) * 4);
            }

            // TODO flags
            // TODO flags offset
            switch (this->glyphParam.Param<core::param::EnumParam>()->Value()) {
            case Glyph::BOX:
                // https://stackoverflow.com/questions/28375338/cube-using-single-gl-triangle-strip
                glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 14, static_cast<GLsizei>(actualItems));
                break;
            case Glyph::ELLIPSOID:
                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(actualItems));
                break;
            case Glyph::ARROW:
                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(actualItems));
                break;
            case Glyph::SUPERQUADRIC:
                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(actualItems));
                break;
            default:;
            }
            //the_pos.SignalCompletion();
            //the_quat.SignalCompletion();
            //the_rad.SignalCompletion();
            //if (bindColor) the_col.SignalCompletion();
        }
        glDisable(GL_TEXTURE_1D);
    }
    epdc->Unlock();
    shader->Disable();

    // todo clean up state
    glDisable(GL_DEPTH_TEST);

    return true;
}