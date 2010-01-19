/*
 * TitleSceneView.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "TitleSceneView.h"
#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */
#include <GL/gl.h>
#include <GL/glu.h>
#include "vislib/mathfunctions.h"
#include "vislib/sysfunctions.h"
#include "vislib/OpenGLVISLogo.h"
#include "MegaMolLogo.h"


using namespace megamol::core;


/*
 * special::TitleSceneView::visLogo
 */
vislib::graphics::AbstractVISLogo *special::TitleSceneView::visLogo = NULL;


/*
 * special::TitleSceneView::megamolLogo
 */
vislib::graphics::AbstractVISLogo *special::TitleSceneView::megamolLogo = NULL;


/*
 * special::TitleSceneView::usageCount
 */
unsigned int special::TitleSceneView::usageCount = 0;


/*
 * special::TitleSceneView::fancyShader
 */
vislib::graphics::gl::GLSLShader *special::TitleSceneView::fancyShader = NULL;


/*
 * special::TitleSceneView::TitleSceneView
 */
special::TitleSceneView::TitleSceneView(void)
        : view::AbstractView(), Module(), cam() {
    // We do not make the slot 'AbstractView::getRenderViewSlot()' available,
    // since we are a top level view and do not want to be rendered inside
    // another one
}


/*
 * special::TitleSceneView::~TitleSceneView
 */
special::TitleSceneView::~TitleSceneView(void) {
    this->Release();
}


/*
 * special::TitleSceneView::Render
 */
void special::TitleSceneView::Render(void) {
    ::glViewport(0, 0,
        static_cast<GLsizei>(this->cam.Parameters()->TileRect().Width()),
        static_cast<GLsizei>(this->cam.Parameters()->TileRect().Height()));
    glClearColor(0.0f, 0.0f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    this->cam.glMultProjectionMatrix();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    float lp[4] = {2.0f, 2.0f, 2.0f, 0.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lp);

    float la[4] = {0.1f, 0.1f, 0.1f, 1.0f};
    glLightfv(GL_LIGHT0, GL_AMBIENT, la);

    float ld[4] = {0.9f, 0.9f, 0.9f, 1.0f};
    glLightfv(GL_LIGHT0, GL_DIFFUSE, ld);

    glScalef(0.4f, 0.4f, 0.4f);
    glTranslatef(1.7f, 0.0f, 0.0f);

    if (visLogo) {
        if (fancyShader != NULL) {
            fancyShader->Enable();

            glEnable(GL_CULL_FACE);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_CLIP_PLANE0);
            int oldDepthFunc;
            glGetIntegerv(GL_DEPTH_FUNC, &oldDepthFunc);
            glDepthFunc(GL_LESS);
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_POLYGON_OFFSET_FILL);

            const double thickness = 0.3;
            double cp[4] = { 0.0, -1.0, 0.0, 0.7 };

            float angle = this->getVISAngle();
            float offset = 0.0f;

            for (cp[3] = -0.5; cp[3] < 0.5; cp[3] += thickness) {

                glClipPlane(GL_CLIP_PLANE0, cp);
                glPolygonOffset(offset, 0.0f);
                offset += 0.01f;

                glPushMatrix();
                glTranslatef(1.2f, 0.0f, 0.0f);
                glRotatef(angle, 0.0f, 0.0f, -1.0f);
                glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
                visLogo->Draw();
                glPopMatrix();
            }

            glDisable(GL_POLYGON_OFFSET_FILL);
            glDepthFunc(oldDepthFunc);
            glDisable(GL_CLIP_PLANE1);
            glDisable(GL_BLEND);
            glDisable(GL_CULL_FACE);

            fancyShader->Disable();
        } else {
            glPushMatrix();
            glTranslatef(1.2f, 0.0f, 0.0f);
            glRotatef(this->getVISAngle(), 0.0f, 0.0f, -1.0f);
            glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
            visLogo->Draw();
            glPopMatrix();
        }
    }

    if (megamolLogo != NULL) {
        glDisable(GL_LIGHTING);
        glScalef(0.6f, 0.6f, 0.6f);
        glRotatef(180.0f, 0.0f, 1.0f, 0.0f);
        megamolLogo->Draw();
    }
}


/*
 * special::TitleSceneView::ResetView
 */
void special::TitleSceneView::ResetView(void) {
    this->cam.Parameters()->SetClip(0.1f, 100.0f);
    this->cam.Parameters()->SetApertureAngle(30.0f);
    this->cam.Parameters()->SetProjection(
        vislib::graphics::CameraParameters::MONO_PERSPECTIVE);
    this->cam.Parameters()->SetStereoParameters(0.3f,
        vislib::graphics::CameraParameters::LEFT_EYE, 3.0f);
    this->cam.Parameters()->Limits()->LimitClippingDistances(0.01f, 0.1f);
}


/*
 * special::TitleSceneView::Resize
 */
void special::TitleSceneView::Resize(unsigned int width, unsigned int height) {
    this->cam.Parameters()->SetVirtualViewSize(
        static_cast<vislib::graphics::ImageSpaceType>(width), 
        static_cast<vislib::graphics::ImageSpaceType>(height));
}


/*
 * special::TitleSceneView::UpdateFreeze
 */
void special::TitleSceneView::UpdateFreeze(bool freeze) {
    // TODO: Implement something useful here
}


/*
 * special::TitleSceneView::onRenderView
 */
bool special::TitleSceneView::onRenderView(Call& call) {
    throw vislib::UnsupportedOperationException(
        "TitleSceneView::onRenderView", __FILE__, __LINE__);
    return false;
}


/*
 * special::TitleSceneView::create
 */
bool special::TitleSceneView::create(void) {
    usageCount++;

    if (visLogo == NULL) {
        visLogo = new vislib::graphics::gl::OpenGLVISLogo();
        try {
            visLogo->Create();
        } catch(...) {
            SAFE_DELETE(visLogo);
        }
    }

    if (megamolLogo == NULL) {
        megamolLogo = new MegaMolLogo();
        try {
            megamolLogo->Create();
        } catch(...) {
            SAFE_DELETE(megamolLogo);
        }
    }

    if (fancyShader == NULL) {
        if (vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()) {
            try {
                if (!vislib::graphics::gl::GLSLShader
                        ::InitialiseExtensions()) {
                    throw 0;
                }
                fancyShader = new vislib::graphics::gl::GLSLShader();
                if (!fancyShader->Create(
// vertex shader
"varying vec3 normal;\n"
"varying float depth;\n"
"void main(void) {\n"
"    normal = gl_NormalMatrix * gl_Normal;\n"
"    gl_FrontColor = gl_Color;\n"
"    gl_Position = ftransform();\n"
"    depth = clamp(0.5 + 0.05 * dot(gl_ModelViewMatrixInverse[3].xyz, gl_Vertex.xyz), 0.0, 1.0);\n"
"}\n"
,
// fragment shader
"varying vec3 normal;\n"
"varying float depth;\n"
"void main(void) {\n"
"    vec3 norm = normalize(normal);\n"
"    float nDl = dot(norm, normalize(gl_LightSource[0].position.xyz));\n"
"    float dif = nDl * ((sign(nDl) + 1.0) * 0.75 - 0.5);\n"
"    float spec = pow(max(dot(norm, normalize(gl_LightSource[0].halfVector.xyz)), 0.0), 20.0);\n"
"    nDl = pow(max(nDl, 0.0), 10.0);\n"
"    vec3 col = gl_Color.rgb;\n"
"    col *= (0.1 + 0.9 * dif);\n"
"    col += depth * abs(norm.yyy) * gl_Color.rgb;\n"
"    col += depth * spec * vec3(1.0);\n"
"    gl_FragColor = vec4(col, clamp(0.5 + 0.25 * dif + spec, 0.0, 1.0));\n"
"}\n"
)) {
                    throw 0;
                }
            } catch(vislib::Exception e) {
#if defined(_WIN32) && defined(DEBUG)
                const char *txt = e.GetMsgA();
#endif /* defined(_WIN32) && defined(DEBUG) */
                SAFE_DELETE(fancyShader);
            } catch(...) {
                SAFE_DELETE(fancyShader);
            }
        }
    }

    return true;
}


/*
 * special::TitleSceneView::release
 */
void special::TitleSceneView::release(void) {
    if (usageCount > 0) usageCount--;
    if (usageCount == 0) {
        if (visLogo != NULL) {
            visLogo->Release();
            SAFE_DELETE(visLogo);
        }
        if (megamolLogo != NULL) {
            megamolLogo->Release();
            SAFE_DELETE(megamolLogo);
        }
        if (fancyShader != NULL) {
            fancyShader->Release();
            SAFE_DELETE(fancyShader);
        }
    }
}


/*
 * special::TitleSceneView::getVISAngle
 */
float special::TitleSceneView::getVISAngle(void) {
    return vislib::sys::GetTicksOfDay() % 10000 * 0.036f;
}
