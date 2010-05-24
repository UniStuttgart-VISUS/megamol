/*
 * TitleRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "TitleRenderer.h"
#include <cmath>
#include "MegaMolLogo.h"
#include "vislib/assert.h"
#include "vislib/graphicstypes.h"
#include "vislib/OpenGLVISLogo.h"

using namespace megamol::core;


/*
 * view::special::TitleRenderer::FallbackIcon::FallbackIcon
 */
view::special::TitleRenderer::FallbackIcon::FallbackIcon(void)
        : vislib::graphics::AbstractVISLogo(), quadric(NULL) {
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
    const float cylLen = 1.8f;
    const float cylRad = 0.1f;
    const float s1Rad = 0.5f;
    const float s2Rad = 0.3f;
    const float sDist = 0.7f;

    const unsigned int cylSlices = 24;
    const unsigned int sphereSlices = 48;
    const unsigned int sphereStacks = 48;

    ::glPushMatrix();

    ::glScalef(2.0f / cylLen, 2.0f / cylLen, 2.0f / cylLen);

    ::glRotatef(-90.0f, 0.0f, 1.0f, 0.0f);
    ::glColor3ub(255, 0, 0);
    ::gluCylinder(this->quadric, cylRad, cylRad, 0.5 * cylLen, cylSlices, 1);
    ::glTranslatef(0.0f, 0.0f, 0.5 * cylLen);
    ::gluDisk(this->quadric, 0.0, cylRad, cylSlices, 1);

    ::glRotatef(180.0f, 0.0f, 1.0f, 0.0f);
    ::glColor3ub(0, 255, 0);
    ::glTranslatef(0.0f, 0.0f, 0.5 * cylLen);
    ::gluCylinder(this->quadric, cylRad, cylRad, 0.5 * cylLen, cylSlices, 1);
    ::glTranslatef(0.0f, 0.0f, 0.5 * cylLen);
    ::gluDisk(this->quadric, 0.0, cylRad, cylSlices, 1);

    ::glTranslatef(0.0f, 0.0f, -0.5 * cylLen);
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
 * view::special::TitleRenderer::TitleRenderer
 */
view::special::TitleRenderer::TitleRenderer()
        : view::AbstractRenderingView::AbstractTitleRenderer(), title(NULL),
        titleWidth(0.0f), icon(NULL), camera() {
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

    // TODO: Implement
    this->icon = new FallbackIcon();
    this->icon->Create();

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
        float virtW, float virtH, bool stereo, bool leftEye,
        double time) {
    if (!this->title || !this->icon) return;

    const float titleScale = 0.5f;
    const float titleGap = 0.0f;
    const float viewBorderScale = 1.25f;

    this->camera.Parameters()->SetVirtualViewSize(virtW, virtH);
    this->camera.Parameters()->SetTileRect(vislib::graphics::ImageSpaceRectangle(tileX, tileY, tileX + tileW, tileY + tileH));
    this->camera.Parameters()->SetProjection(stereo ? vislib::graphics::CameraParameters::STEREO_OFF_AXIS : vislib::graphics::CameraParameters::MONO_PERSPECTIVE);
    this->camera.Parameters()->SetEye(leftEye ? vislib::graphics::CameraParameters::LEFT_EYE : vislib::graphics::CameraParameters::RIGHT_EYE);

    this->zoomCamera(viewBorderScale * 0.5f * (2.0f + titleGap + (this->titleWidth * titleScale)), viewBorderScale, -viewBorderScale);

    this->camera.glSetMatrices();

    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    const float lp[4] = {1.0f, 1.0f, -1.0f, 0.0f};
    ::glLightfv(GL_LIGHT0, GL_POSITION, lp);
    const float la[4] = {0.1f, 0.1f, 0.1f, 1.0f};
    ::glLightfv(GL_LIGHT0, GL_AMBIENT, la);
    const float ld[4] = {0.9f, 0.9f, 0.9f, 1.0f};
    ::glLightfv(GL_LIGHT0, GL_DIFFUSE, ld);

    ::glTranslatef(((this->titleWidth * titleScale) + titleGap) * 0.5f, 0.0f, 0.0f);

    ::glPushMatrix();
    ::glRotated(15.0 * time, 0.0, 1.0, 0.0);
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
