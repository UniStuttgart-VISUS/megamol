/*
 * ColStereoDisplay.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES 1
#include "glh/glh_extensions.h"
#include "ColStereoDisplay.h"
#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */
#include "param/BoolParam.h"
#include "special/ClusterSignRenderer.h"
#include "view/CallRenderView.h"
#include "vislib/Log.h"
#if defined(DEBUG) || defined(_DEBUG)
#include "vislib/Trace.h"
#endif

using namespace megamol::core;


/*
 * special::ColStereoDisplay::ColStereoDisplay
 */
special::ColStereoDisplay::ColStereoDisplay(void) : AbstractStereoDisplay(),
        fbo(), compShader(),
        flipEyes("flipEyes", "Flip the eye configuration for the columns") {

    this->flipEyes << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->flipEyes);
}


/*
 * special::ColStereoDisplay::~ColStereoDisplay
 */
special::ColStereoDisplay::~ColStereoDisplay(void) {
    this->Release();
}


/*
 * special::ColStereoDisplay::Render
 */
void special::ColStereoDisplay::Render(void) {

    view::CallRenderView *crv = this->getCallRenderView();
    if (crv != NULL) {
        unsigned int rhw = this->width() / 2; // right half width

        if ((this->fbo.GetWidth() != this->width())
                || (this->fbo.GetHeight() != this->height())) {
            this->fbo.Release();
            this->fbo.Create(this->width(), this->height());
        }

        bool flip = this->flipEyes.Param<param::BoolParam>()->Value();

#if defined(DEBUG) || defined(_DEBUG)
        unsigned int oldTraceLevel = vislib::Trace::GetInstance().GetLevel();
        vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_ERROR);
#endif
        this->fbo.Enable();
#if defined(DEBUG) || defined(_DEBUG)
        vislib::Trace::GetInstance().SetLevel(oldTraceLevel);
#endif

        ::glViewport(0, 0, this->width(), this->height());
        ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ::glViewport(0, 0, this->width() - rhw, this->height());
        ::glScissor(0, 0, this->width() - rhw, this->height());
        crv->ResetAll();
        crv->SetProjection(vislib::graphics::CameraParameters::STEREO_OFF_AXIS,
            flip ? vislib::graphics::CameraParameters::RIGHT_EYE : vislib::graphics::CameraParameters::LEFT_EYE);
        crv->SetTile(static_cast<float>(this->width()), static_cast<float>(this->height()),
            0.0f, 0.0f, static_cast<float>(this->width()), static_cast<float>(this->height()));
        crv->SetViewportSize(this->width(), this->height());
        ::glEnable(GL_SCISSOR_TEST);
        (*crv)();

        ::glViewport(this->width() - rhw, 0, rhw, this->height());
        ::glScissor(this->width() - rhw, 0, rhw, this->height());
        crv->ResetAll();
        crv->SetProjection(vislib::graphics::CameraParameters::STEREO_OFF_AXIS,
            flip ? vislib::graphics::CameraParameters::LEFT_EYE : vislib::graphics::CameraParameters::RIGHT_EYE);
        crv->SetTile(static_cast<float>(this->width()), static_cast<float>(this->height()),
            0.0f, 0.0f, static_cast<float>(this->width()), static_cast<float>(this->height()));
        crv->SetViewportSize(this->width(), this->height());
        ::glEnable(GL_SCISSOR_TEST);
        (*crv)();
        ::glDisable(GL_SCISSOR_TEST);

        this->fbo.Disable();

        ::glViewport(0, 0, this->width(), this->height());

        ::glEnable(GL_TEXTURE_2D);
        this->fbo.BindColourTexture();
        ::glDisable(GL_LIGHTING);
        ::glDisable(GL_DEPTH_TEST);
        ::glDisable(GL_BLEND);

        ::glMatrixMode(GL_PROJECTION);
        ::glLoadIdentity();
        ::glMatrixMode(GL_MODELVIEW);
        ::glLoadIdentity();

        this->compShader.Enable();
        this->compShader.SetParameter("img", 0);
        this->compShader.SetParameter("size",
            static_cast<float>(this->width()),
            static_cast<float>(this->height()),
            static_cast<float>(this->width() - rhw));

        ::glColor3ub(255, 255, 255);
        ::glBegin(GL_QUADS);
        ::glVertex2i(-1, -1);
        ::glVertex2i( 1, -1);
        ::glVertex2i( 1,  1);
        ::glVertex2i(-1,  1);
        ::glEnd();

        this->compShader.Disable();

        ::glDisable(GL_TEXTURE_2D);

    } else {
        ::glViewport(0, 0, this->width(), this->height());
        special::ClusterSignRenderer::RenderBroken(
            this->width(), this->height());

    }

}


/*
 * special::ColStereoDisplay::create
 */
bool special::ColStereoDisplay::create(void) {
    using vislib::sys::Log;
    if (!AbstractStereoDisplay::create()) return false;

    if (!this->fbo.AreExtensionsAvailable()
            || !this->compShader.AreExtensionsAvailable()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Failed to create ColStereoDisplay: Required gl extensions are not available.");
        return false;
    }
    if (!this->fbo.InitialiseExtensions()
            || !this->compShader.InitialiseExtensions()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Failed to create ColStereoDisplay: Fail to initialize gl extensions.");
        return false;
    }

    if (!this->fbo.Create(1, 1)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Failed to create ColStereoDisplay: Failed to create framebuffer object.");
        return false;
    }

    // we just do not need the factory here
    const char *fragShader = 
        "uniform sampler2D img;" "\n"
        "uniform vec3 size;" "\n"
        "" "\n"
        "void main() { " "\n"
        "    vec2 texCoord;" "\n"
        "    texCoord.y = gl_FragCoord.y / size.y;" "\n"
        "    if ((((int)gl_FragCoord.x) % 2) == 1) {" "\n"
        "        texCoord.x = (size.z + (gl_FragCoord.x - 1.0) * 0.5) / size.x;" "\n"
        "        gl_FragColor = texture2D(img, texCoord);" "\n"
        "    } else {" "\n"
        "        texCoord.x = gl_FragCoord.x * 0.5 / size.x;" "\n"
        "        gl_FragColor = texture2D(img, texCoord);" "\n"
        "    }" "\n"
        "}";

    if (!this->compShader.Create(vislib::graphics::gl::GLSLShader
            ::FTRANSFORM_VERTEX_SHADER_SRC, fragShader)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Failed to create ColStereoDisplay: Failed to create shader.");
        return false;
    }

    return true;
}


/*
 * special::ColStereoDisplay::release
 */
void special::ColStereoDisplay::release(void) {
    this->fbo.Release();
    this->compShader.Release();
}
