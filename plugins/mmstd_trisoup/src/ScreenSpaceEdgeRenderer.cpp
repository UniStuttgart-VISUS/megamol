/*
 * ScreenSpaceEdgeRenderer.cpp
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ScreenSpaceEdgeRenderer.h"
#include "mmcore/CoreInstance.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"

using namespace megamol;
using namespace megamol::trisoup;

using namespace megamol::core;

/*
 * ScreenSpaceEdgeRenderer::ScreenSpaceEdgeRenderer
 */
ScreenSpaceEdgeRenderer::ScreenSpaceEdgeRenderer(void) : Renderer3DModule(),
        rendererSlot("renderer", "Connects to the renderer actually rendering the image"),
        colorSlot("color", "The triangle color (if not colors are read from file)") {

    this->rendererSlot.SetCompatibleCall<view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->rendererSlot);

    this->colorSlot.SetParameter(new param::StringParam("silver"));
    this->MakeSlotAvailable(&this->colorSlot);
}


/*
 * ScreenSpaceEdgeRenderer::~ScreenSpaceEdgeRenderer
 */
ScreenSpaceEdgeRenderer::~ScreenSpaceEdgeRenderer(void) {
    this->Release();
}


/*
 * ScreenSpaceEdgeRenderer::create
 */
bool ScreenSpaceEdgeRenderer::create(void) {
    ASSERT(IsAvailable());

    vislib::graphics::gl::ShaderSource vert, frag;

    if (!instance()->ShaderSourceFactory().MakeShaderSource("ScreenSpaceEdge::vertex", vert)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("ScreenSpaceEdge::fragment", frag)) {
        return false;
    }

    //printf("\nVertex Shader:\n%s\n\nFragment Shader:\n%s\n",
    //    vert.WholeCode().PeekBuffer(),
    //    frag.WholeCode().PeekBuffer());

    try {
        if (!this->shader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile ScreenSpaceEdge shader: Unknown error\n");
            return false;
        }

    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile ScreenSpaceEdge shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()), ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile ScreenSpaceEdge shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile ScreenSpaceEdge shader: Unknown exception\n");
        return false;
    }

    return true;
}


/*
 * ScreenSpaceEdgeRenderer::GetExtents
 */
bool ScreenSpaceEdgeRenderer::GetExtents(Call& call) {
    view::CallRender3D *inCall = dynamic_cast<view::CallRender3D*>(&call);
    if (inCall == NULL) return false;

    view::CallRender3D *outCall = this->rendererSlot.CallAs<view::CallRender3D>();
    if (outCall == NULL) return false;

    *outCall = *inCall;

    if (!(*outCall)(1)) return false;

    *inCall = *outCall;

    return true;
}


/*
 * ScreenSpaceEdgeRenderer::release
 */
void ScreenSpaceEdgeRenderer::release(void) {
    if (vislib::graphics::gl::GLSLShader::IsValidHandle(this->shader.ProgramHandle())) this->shader.Release();
    if (this->fbo.IsValid()) this->fbo.Release();
}


/*
 * ScreenSpaceEdgeRenderer::Render
 */
bool ScreenSpaceEdgeRenderer::Render(Call& call) {
    view::CallRender3D *inCall = dynamic_cast<view::CallRender3D*>(&call);
    if (inCall == NULL) return false;

    view::CallRender3D *outCall = this->rendererSlot.CallAs<view::CallRender3D>();
    if (outCall == NULL) return false;

    //const vislib::math::Rectangle<int>& vp = inCall->GetViewport();
    inCall->DisableOutputBuffer();
    const vislib::math::Rectangle<int>& vp = inCall->GetCameraParameters()->TileRect();

    if (!this->fbo.IsValid()
            || (this->fbo.GetWidth() != static_cast<unsigned int>(vp.Width()))
            || (this->fbo.GetHeight() != static_cast<unsigned int>(vp.Height()))) {
        if (this->fbo.IsValid()) this->fbo.Release();
        this->fbo.Create(
            static_cast<unsigned int>(vp.Width()), static_cast<unsigned int>(vp.Height()),
            GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
            vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
    }

    fbo.Enable();
    ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    *outCall = *inCall;
    outCall->SetOutputBuffer(&fbo);

    bool renderValid = (*outCall)(core::view::AbstractCallRender::FnRender);

    fbo.Disable();

    inCall->EnableOutputBuffer();

    ::glPushAttrib(GL_TEXTURE_BIT | GL_TRANSFORM_BIT);

    float r, g, b;
    this->colorSlot.ResetDirty();
    utility::ColourParser::FromString(this->colorSlot.Param<param::StringParam>()->Value(), r, g, b);

    ::glDisable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    shader.Enable();
    glActiveTexture(GL_TEXTURE0 + 0);
    fbo.BindColourTexture(0);
    glActiveTexture(GL_TEXTURE0 + 1);
    fbo.BindDepthTexture();
    shader.SetParameter("inColorTex", 0);
    shader.SetParameter("inDepthTex", 1);
    shader.SetParameter("viewportMax", vp.Width() - 1, vp.Height() - 1);
    shader.SetParameter("color", r, g, b);

    ::glMatrixMode(GL_PROJECTION);
    ::glPushMatrix();
    ::glLoadIdentity();

    ::glMatrixMode(GL_MODELVIEW);
    ::glPushMatrix();
    ::glLoadIdentity();

    ::glBegin(GL_QUADS);
    ::glVertex3d(-1.0, -1.0, 0.5);
    ::glVertex3d(1.0, -1.0, 0.5);
    ::glVertex3d(1.0, 1.0, 0.5);
    ::glVertex3d(-1.0, 1.0, 0.5);
    ::glEnd();

    // GL_MODELVIEW
    ::glPopMatrix();

    ::glMatrixMode(GL_PROJECTION);
    ::glPopMatrix();

    shader.Disable();

    ::glPopAttrib();

    return true;
}
