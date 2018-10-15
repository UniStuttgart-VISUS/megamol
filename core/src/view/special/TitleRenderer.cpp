/*
 * TitleRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#define _USE_MATH_DEFINES
#include "mmcore/view/special/TitleRenderer.h"
#include <cmath>
//#include <ctime>
#include "mmcore/view/special/MegaMolLogo.h"
#include "mmcore/CoreInstance.h"
#include "vislib/assert.h"
#include "vislib/graphics/graphicstypes.h"
#include "vislib/math/Matrix.h"
#include "vislib/graphics/gl/OpenGLVISLogo.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/math/Vector.h"

using namespace megamol::core;


/*
 * view::special::TitleRenderer::AbstractIcon::AbstractIcon
 */
view::special::TitleRenderer::AbstractIcon::AbstractIcon(void)
        : vislib::graphics::AbstractVISLogo() {
    // Intentionally empty
}


/*
 * view::special::TitleRenderer::AbstractIcon::~AbstractIcon
 */
view::special::TitleRenderer::AbstractIcon::~AbstractIcon(void) {
    // Intentionally empty
}


/*
 * view::special::TitleRenderer::AbstractIcon::cylLen
 */
const float view::special::TitleRenderer::AbstractIcon::cylLen = 1.8f;


/*
 * view::special::TitleRenderer::AbstractIcon::cylRad
 */
const float view::special::TitleRenderer::AbstractIcon::cylRad = 0.1f;


/*
 * view::special::TitleRenderer::AbstractIcon::s1Rad
 */
const float view::special::TitleRenderer::AbstractIcon::s1Rad = 0.5f;


/*
 * view::special::TitleRenderer::AbstractIcon::s2Rad
 */
const float view::special::TitleRenderer::AbstractIcon::s2Rad = 0.3f;


/*
 * view::special::TitleRenderer::AbstractIcon::sDist
 */
const float view::special::TitleRenderer::AbstractIcon::sDist = 0.7f;


/*
 * view::special::TitleRenderer::FallbackIcon::FallbackIcon
 */
view::special::TitleRenderer::FallbackIcon::FallbackIcon(void)
        : AbstractIcon(), quadric(NULL) {
    // Intentionally empty
}


/*
 * view::special::TitleRenderer::FallbackIcon::~FallbackIcon
 */
view::special::TitleRenderer::FallbackIcon::~FallbackIcon(void) {
    this->Release();
    ASSERT(this->quadric == NULL);
}


/*
 * view::special::TitleRenderer::FallbackIcon::Create
 */
void view::special::TitleRenderer::FallbackIcon::Create(void) {
    ASSERT(this->quadric == NULL);
    this->quadric = ::gluNewQuadric();
}


/*
 * view::special::TitleRenderer::FallbackIcon::Draw
 */
void view::special::TitleRenderer::FallbackIcon::Draw(void) {
    const unsigned int cylSlices = 24;
    const unsigned int sphereSlices = 48;
    const unsigned int sphereStacks = 48;

    ::glPushMatrix();
    ::glScalef(2.0f / cylLen, 2.0f / cylLen, 2.0f / cylLen);

    ::glRotatef(-90.0f, 0.0f, 1.0f, 0.0f);
    ::glColor3ub(255, 0, 0);
    ::gluCylinder(this->quadric, cylRad, cylRad, 0.5 * cylLen, cylSlices, 1);
    ::glTranslatef(0.0f, 0.0f, 0.5f * cylLen);
    ::gluDisk(this->quadric, 0.0, cylRad, cylSlices, 1);

    ::glRotatef(180.0f, 0.0f, 1.0f, 0.0f);
    ::glColor3ub(0, 255, 0);
    ::glTranslatef(0.0f, 0.0f, 0.5f * cylLen);
    ::gluCylinder(this->quadric, cylRad, cylRad, 0.5 * cylLen, cylSlices, 1);
    ::glTranslatef(0.0f, 0.0f, 0.5f * cylLen);
    ::gluDisk(this->quadric, 0.0, cylRad, cylSlices, 1);

    ::glTranslatef(0.0f, 0.0f, -0.5f * cylLen);
    ::glColor3ub(0x80, 0xC0, 0xFF);
    ::glTranslatef(0.0f, 0.0f, (s1Rad - s2Rad) * -0.5f);
    ::glTranslatef(0.0f, 0.0f, 0.5f * sDist);
    ::gluSphere(this->quadric, s1Rad, sphereSlices, sphereStacks);
    ::glTranslatef(0.0f, 0.0f, -sDist);
    ::gluSphere(this->quadric, s2Rad, sphereSlices, sphereStacks);

    ::glPopMatrix();
}


/*
 * view::special::TitleRenderer::FallbackIcon::Release
 */
void view::special::TitleRenderer::FallbackIcon::Release(void) {
    if (this->quadric != NULL) {
        ::gluDeleteQuadric(this->quadric);
        this->quadric = NULL;
    }
}


/*
 * view::special::TitleRenderer::GPURaycastIcon::GPURaycastIcon
 */
view::special::TitleRenderer::GPURaycastIcon::GPURaycastIcon(CoreInstance *core)
        : AbstractIcon(), error(false), core(core), shader(NULL) {
    // Intentionally empty
}


/*
 * view::special::TitleRenderer::GPURaycastIcon::~GPURaycastIcon
 */
view::special::TitleRenderer::GPURaycastIcon::~GPURaycastIcon(void) {
    this->Release();
    this->core = NULL; // DO NOT DELETE
    ASSERT(this->shader == NULL);
}


/*
 * view::special::TitleRenderer::GPURaycastIcon::Create
 */
void view::special::TitleRenderer::GPURaycastIcon::Create(void) {
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        this->error = true;
        return;
    }

    vislib::graphics::gl::ShaderSource vertexShader;
    vislib::graphics::gl::ShaderSource fragmentShader;

    if (!this->core->ShaderSourceFactory().MakeShaderSource("titlelogo::vertex", vertexShader)) {
        this->error = true;
        return;
    }
    if (!this->core->ShaderSourceFactory().MakeShaderSource("titlelogo::fragment", fragmentShader)) {
        this->error = true;
        return;
    }

    try {
        this->shader = new vislib::graphics::gl::GLSLShader();
        if (!this->shader->Compile(vertexShader.Code(), vertexShader.Count(), fragmentShader.Code(), fragmentShader.Count())) {
            this->error = true;
            delete this->shader;
            this->shader = NULL;
            return;
        }
        if (this->shader->BindAttribute(8, "inParams") != GL_NO_ERROR) {
            this->error = true;
            delete this->shader;
            this->shader = NULL;
            return;
        }
        if (this->shader->BindAttribute(9, "quatC") != GL_NO_ERROR) {
            this->error = true;
            delete this->shader;
            this->shader = NULL;
            return;
        }
        if (this->shader->BindAttribute(10, "inPos") != GL_NO_ERROR) {
            this->error = true;
            delete this->shader;
            this->shader = NULL;
            return;
        }
        if (!this->shader->Link()) {
            this->error = true;
            delete this->shader;
            this->shader = NULL;
            return;
        }
    } catch (vislib::Exception ex) {
        //const char *msg = ex.GetMsgA();
        this->error = true;
        delete this->shader;
        this->shader = NULL;
    } catch (...) {
        this->error = true;
        delete this->shader;
        this->shader = NULL;
    }
}


/*
 * view::special::TitleRenderer::GPURaycastIcon::Draw
 */
void view::special::TitleRenderer::GPURaycastIcon::Draw(void) {
    if (!this->shader) return;

    ::glPushMatrix();
    ::glScalef(2.0f / cylLen, 2.0f / cylLen, 2.0f / cylLen);

    const float xExt = 0.5f * cylLen;
    const float yzExt = vislib::math::Max(s1Rad, s2Rad);

    double proj[16];
    double modview[16];
    int viewport[4];
    ::glGetDoublev(GL_PROJECTION_MATRIX, proj);
    ::glGetDoublev(GL_MODELVIEW_MATRIX, modview);
    ::glGetIntegerv(GL_VIEWPORT, viewport);

    vislib::math::Vector<double, 3> os[8];
    vislib::math::Vector<double, 3> ws[4];
    os[0].Set(xExt, yzExt , yzExt);
    os[1].Set(-xExt, yzExt , yzExt);
    os[2].Set(xExt, -yzExt, yzExt);
    os[3].Set(-xExt, -yzExt, yzExt);
    os[4].Set(xExt, yzExt, -yzExt);
    os[5].Set(-xExt, yzExt, -yzExt);
    os[6].Set(xExt, -yzExt, -yzExt);
    os[7].Set(-xExt, -yzExt, -yzExt);

    for (unsigned int i = 0; i < 8; i++) {
        ::gluProject(os[i].X(), os[i].Y(), os[i].Z(), modview, proj, viewport,
            ws[0].PeekComponents(), ws[0].PeekComponents() + 1, ws[0].PeekComponents() + 2);
        if (i == 0) {
            ws[1] = ws[0];
            ws[2] = ws[0];
        } else {
            ws[1].Set(vislib::math::Min(ws[1].X(), ws[0].X()),
                vislib::math::Min(ws[1].Y(), ws[0].Y()),
                vislib::math::Min(ws[1].Z(), ws[0].Z()));
            ws[2].Set(vislib::math::Max(ws[2].X(), ws[0].X()),
                vislib::math::Max(ws[2].Y(), ws[0].Y()),
                vislib::math::Max(ws[2].Z(), ws[0].Z()));
        }
    }

    ws[0] = ws[1];
    ws[2].SetZ(ws[0].Z());
    ws[3] = ws[2];
    ws[1].SetY(ws[2].Y());
    ws[3].SetY(ws[0].Y());

    for (unsigned int i = 0; i < 4; i++) {
        ::gluUnProject(ws[i].X(), ws[i].Y(), ws[i].Z(), modview, proj, viewport,
            os[i].PeekComponents(), os[i].PeekComponents() + 1, os[i].PeekComponents() + 2);
    }

    this->shader->Enable();
    this->shader->SetParameter("viewAttr", static_cast<float>(viewport[0]), static_cast<float>(viewport[1]),
        2.0f / static_cast<float>(viewport[2]), 2.0f / static_cast<float>(viewport[3]));

    ::glBegin(GL_QUADS);
    ::glColor3ub(0x80, 0xC0, 0xFF);
    ::glVertexAttrib4fARB(8, s1Rad, sDist, cylRad, cylLen);
    ::glVertexAttrib4fARB(9, 0.0f, 0.0f, 0.0f, 1.0f);
    ::glVertexAttrib3fARB(10, 0.5f * (s2Rad - s1Rad), 0.0f, 0.0f);
    for (int i = 3; i >= 0; i--) {
        ::glVertex4d(os[i].X(), os[i].Y(), os[i].Z(), s2Rad);
    }
    ::glEnd();

    this->shader->Disable();

    ::glPopMatrix();
}


/*
 * view::special::TitleRenderer::GPURaycastIcon::Release
 */
void view::special::TitleRenderer::GPURaycastIcon::Release(void) {
    // DO NOT DELETE this->core
    if (this->shader) {
        this->shader->Release();
        delete this->shader;
        this->shader = NULL;
    }
}


/*
 * view::special::TitleRenderer::TitleRenderer
 */
view::special::TitleRenderer::TitleRenderer()
        : view::AbstractRenderingView::AbstractTitleRenderer(), title(NULL),
        titleWidth(0.0f), icon(NULL),
#ifdef ICON_DEBUGGING
        i2(NULL),
#endif /* ICON_DEBUGGING */
        camera() {
    // intentionally empty
}


/*
 * view::special::TitleRenderer::~TitleRenderer
 */
view::special::TitleRenderer::~TitleRenderer() {
    this->Release();
    ASSERT(this->title == NULL);
    ASSERT(this->icon == NULL);
}


/*
 * view::special::TitleRenderer::Create
 */
bool view::special::TitleRenderer::Create(void) {
    this->title = new MegaMolLogo();
    this->title->Create();
    this->titleWidth = dynamic_cast<MegaMolLogo*>(this->title)->MaxX();

    this->icon = NULL; // needs to be evaluated lazy

    const float distance = 10.0f;
    const float stereoDisp = 1.0f;

    this->camera.Parameters()->SetView(
        vislib::graphics::SceneSpacePoint3D(0.0f, 0.0f, -distance),
        vislib::graphics::SceneSpacePoint3D(0.0f, 0.0f, 0.0f),
        vislib::graphics::SceneSpaceVector3D(0.0f, 1.0f, 0.0f));
    this->camera.Parameters()->SetFocalDistance(distance);
    this->camera.Parameters()->SetClip(distance - 2.0f, distance + 2.0f);
    this->camera.Parameters()->SetStereoDisparity(stereoDisp);

    return true;
}


/*
 * view::special::TitleRenderer::Render
 */
void view::special::TitleRenderer::Render(
        float tileX, float tileY, float tileW, float tileH,
        float virtW, float virtH, bool stereo, bool leftEye, double instTime,
        CoreInstance *core) {

    double angle;
    angle = 15.0 * instTime;

    if (this->icon == NULL) {
        this->icon = new GPURaycastIcon(core);
        this->icon->Create();
        if (dynamic_cast<GPURaycastIcon*>(this->icon)->HasError()) {
            delete this->icon;
            this->icon = new FallbackIcon();
            this->icon->Create();
        }
#ifdef ICON_DEBUGGING
        this->i2 = new FallbackIcon();
        this->i2->Create();
#endif /* ICON_DEBUGGING */
    }
    if (!this->title || !this->icon) return;
#ifdef ICON_DEBUGGING
    if (!this->i2) return;
    {
        vislib::graphics::AbstractVISLogo *ai = this->icon;
        this->icon = i2;
        this->i2 = ai;
    }
#endif /* ICON_DEBUGGING */
    const float titleScale = 0.5f;
    const float titleGap = 0.0f;
    const float viewBorderScale = 1.25f;

    this->camera.Parameters()->SetVirtualViewSize(virtW, virtH);
    this->camera.Parameters()->SetTileRect(vislib::graphics::ImageSpaceRectangle(tileX, tileY, tileX + tileW, tileY + tileH));
    this->camera.Parameters()->SetProjection(stereo ? vislib::graphics::CameraParameters::STEREO_OFF_AXIS : vislib::graphics::CameraParameters::MONO_PERSPECTIVE);
    this->camera.Parameters()->SetEye(leftEye ? vislib::graphics::CameraParameters::LEFT_EYE : vislib::graphics::CameraParameters::RIGHT_EYE);

    this->zoomCamera(viewBorderScale * 0.5f * (2.0f + titleGap + (this->titleWidth * titleScale)), viewBorderScale, -viewBorderScale);

    this->camera.glSetMatrices();

    glDisable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    ::glPushMatrix();
    ::glTranslatef(
        this->camera.Parameters()->EyePosition().X(),
        this->camera.Parameters()->EyePosition().Y(),
        this->camera.Parameters()->EyePosition().Z());
    ::glRotated(angle * 0.1, 0.0, -1.0, 0.0);
    const unsigned int steps = 150;

    ::glColor4ub(0, 0, 0, 192);
    ::glBegin(GL_TRIANGLE_FAN);
    ::glVertex3d(0.0, -10.0, 0.0);
    for (unsigned int i = 0; i <= steps; i++) {
        double a = M_PI * static_cast<double>(2 * i) / static_cast<double>(steps);
        ::glVertex3d(cos(a) * 10.0, -10.0, sin(a) * 10.0);
    }
    ::glEnd();
    ::glBegin(GL_TRIANGLE_STRIP);
    for (unsigned int i = 0; i <= steps; i++) {
        double a = M_PI * static_cast<double>(2 * i) / static_cast<double>(steps);
        ::glVertex3d(cos(a) * 10.0, -10.0, sin(a) * 10.0);
        ::glVertex3d(cos(a) * 10.0, -6.0, sin(a) * 10.0);
    }
    ::glEnd();
    ::glBegin(GL_QUADS);
    for (unsigned int i = 0; i <= steps; i++) {
        const double more = 0.06;
        double a11 = M_PI * static_cast<double>(2 * i) / static_cast<double>(steps);
        double a12 = M_PI * static_cast<double>(2 * i + 2) / static_cast<double>(steps);
        double a21 = M_PI * (more + static_cast<double>(2 * i) / static_cast<double>(steps));
        double a22 = M_PI * (more + static_cast<double>(2 * i + 2) / static_cast<double>(steps));
        double a31 = M_PI * (2.0 * more + static_cast<double>(2 * i) / static_cast<double>(steps));
        double a32 = M_PI * (2.0 * more + static_cast<double>(2 * i + 2) / static_cast<double>(steps));
        ::glColor4ub(0, 0, 0, 192);
        ::glVertex3d(cos(a11) * 10.0, -6.0, sin(a11) * 10.0);
        ::glVertex3d(cos(a12) * 10.0, -6.0, sin(a12) * 10.0);
        if (i % 2) ::glColor4ub(150, 150, 0, 192);
            else ::glColor4ub(24, 24, 24, 192);
        ::glVertex3d(cos(a22) * 10.0, -3.5, sin(a22) * 10.0);
        ::glVertex3d(cos(a21) * 10.0, -3.5, sin(a21) * 10.0);
        ::glVertex3d(cos(a21) * 10.0, -3.5, sin(a21) * 10.0);
        ::glVertex3d(cos(a22) * 10.0, -3.5, sin(a22) * 10.0);
        if (i % 2) ::glColor4ub(150, 150, 0, 0);
            else ::glColor4ub(24, 24, 24, 0);
        ::glVertex3d(cos(a32) * 10.0, -1.0, sin(a32) * 10.0);
        ::glVertex3d(cos(a31) * 10.0, -1.0, sin(a31) * 10.0);
    }
    ::glEnd();

    ::glPopMatrix();
    ::glPushMatrix();

    double d = (1.0 - (instTime * 0.25 - static_cast<double>(static_cast<int>(instTime * 0.25)))) * 4.0 * 12.0;
    ::glTranslated(0.0, -1.5, 2.0);
    ::glScaled(0.125, 0.05, 1.0);
    ::glTranslated(d, 0.0, 0.0);

    for (unsigned int j = 0; j < 5; j++) {
        for (unsigned int i = 0; i < 2; i++) {
            ::glBegin(GL_TRIANGLE_FAN);
            ::glColor4ub(255, 255, 255, 192);
            ::glVertex2d(0.0, 0.0);
            ::glColor4ub(255, 255, 255, 0);
            ::glVertex2d(2.0, 0.0);
            ::glVertex2d(1.3, 0.8);
            ::glVertex2d(0.0, 1.0);
            ::glVertex2d(-1.3, 0.8);
            ::glVertex2d(-2.0, 0.0);
            ::glVertex2d(-1.3, -0.8);
            ::glVertex2d(0.0, -1.0);
            ::glVertex2d(1.3, -0.8);
            ::glVertex2d(2.0, 0.0);
            ::glEnd();
            ::glTranslated(-2.0 * d, 0.0, 0.0);
        }
        ::glTranslated(4.0 * d, -1.0, -1.0);
    }

    ::glPopMatrix();

    ::glClear(GL_DEPTH_BUFFER_BIT);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    const float lp[4] = {1.0f, 1.0f, -1.0f, 0.0f};
    ::glLightfv(GL_LIGHT0, GL_POSITION, lp);
    const float la[4] = {0.1f, 0.1f, 0.1f, 1.0f};
    ::glLightfv(GL_LIGHT0, GL_AMBIENT, la);
    const float ld[4] = {0.9f, 0.9f, 0.9f, 1.0f};
    ::glLightfv(GL_LIGHT0, GL_DIFFUSE, ld);

    ::glTranslatef(((this->titleWidth * titleScale) + titleGap) * 0.5f, 0.0f, 0.0f);

// not synchronized at all
//#ifdef _WIN32
//    //SYSTEMTIME now; // not sufficiently synchronized :-(
//    //GetSystemTime(&now);
//    //angle = static_cast<double>(now.wSecond);
//    //angle += static_cast<double>(now.wMinute) * 60.0;
//    //angle += static_cast<double>(now.wHour) * 60.0 * 60.0;
//    //angle += static_cast<double>(now.wMilliseconds) * 0.001;
//    //angle *= 30.0; // 20 deg per second
//#else /* _WIN32 */
//    angle = 0.0;
//#endif /* _WIN32 */

    ::glPushMatrix();
    ::glRotated(angle, 0.0, -1.0, 0.0);
    ::glRotatef(45.0f, 0.0f, 0.0f, -1.0f);

    ::glEnable(GL_CULL_FACE);
    ::glEnable(GL_LIGHTING);
    ::glEnable(GL_DEPTH_TEST);
    this->icon->Draw();

    ::glPopMatrix();

    ::glDisable(GL_CULL_FACE);
    ::glDisable(GL_LIGHTING);
    ::glDisable(GL_DEPTH_TEST);
    ::glTranslatef(-(1.0f + titleGap), -0.05f, 0.0f);
    ::glScalef(titleScale, titleScale, titleScale);
    ::glRotatef(180.0f, 0.0f, 1.0f, 0.0f);
    this->title->Draw();

}


/*
 * view::special::TitleRenderer::Release
 */
void view::special::TitleRenderer::Release(void) {
    if (this->title) {
        this->title->Release();
        delete this->title;
        this->title = NULL;
    }
    if (this->icon) {
        this->icon->Release();
        delete this->icon;
        this->icon = NULL;
    }
#ifdef ICON_DEBUGGING
    if (this->i2) {
        this->i2->Release();
        delete this->i2;
        this->i2 = NULL;
    }
#endif /* ICON_DEBUGGING */
}


/*
 * view::special::TitleRenderer::zoomCamera
 */
void view::special::TitleRenderer::zoomCamera(float x, float y, float z) {
    vislib::math::Vector<float, 3> d(
        x - this->camera.Parameters()->Position().X(),
        y - this->camera.Parameters()->Position().Y(),
        z - this->camera.Parameters()->Position().Z());
    float f = vislib::math::Abs(this->camera.Parameters()->Front().Dot(d));
    float r = vislib::math::Abs(this->camera.Parameters()->Right().Dot(d));
    float u = vislib::math::Abs(this->camera.Parameters()->Up().Dot(d));

    if (vislib::math::IsEqual(f, 0.0f)) return;

    float haay = ::atan(u / f);
    float haax = ::atan((r * this->camera.Parameters()->VirtualViewSize().Height() / this->camera.Parameters()->VirtualViewSize().Width()) / f);

    this->camera.Parameters()->SetApertureAngle(vislib::math::AngleRad2Deg(vislib::math::Max(haay, haax) * 2.0f));

}
