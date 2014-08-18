/*
 * CamTestApp.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */
#define _USE_MATH_DEFINES
#include "CamTestApp.h"

#include "vislibGlutInclude.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <cstdio>

#include "vislib/CameraParamsTileRectOverride.h"
#include "vislib/graphicstypes.h"
#include "vislib/PerformanceCounter.h"
#include "vislib/Rectangle.h"
#include <cmath>

/*
 * CamTestApp::Lens::Lens
 */
CamTestApp::Lens::Lens(void) {
    this->w = 0.25f;
    this->h = 0.25f;
    this->x = float(::rand() % 1000) / 1000.0f * (1.0f - this->w);
    this->y = float(::rand() % 1000) / 1000.0f * (1.0f - this->h);
    this->ax = float(::rand() % 2000) / 2000.0f * float(M_PI);
    this->ay = sin(this->ax);
    this->ax = cos(this->ax);
}


/*
 * CamTestApp::Lens::~Lens
 */
CamTestApp::Lens::~Lens(void) {
}


/*
 * CamTestApp::Lens::SetCameraParameters
 */
void CamTestApp::Lens::SetCameraParameters(
        const vislib::SmartPtr<vislib::graphics::CameraParameters>& params) {
    this->camera.SetParameters(
        new vislib::graphics::CameraParamsTileRectOverride(params));
}


/*
 * CamTestApp::Lens::Update
 */
void CamTestApp::Lens::Update(float sec) {
    bool pong = false;

    this->x += ax * sec;
    this->y += ay * sec;

    if (((this->x < 0.0f) && (this->ax < 0.0f)) || ((this->x > 1.0f - this->w) && (this->ax > 0.0f))) {
        this->ax = -this->ax;
        pong = true;
    }
    if (((this->y < 0.0f) && (this->ay < 0.0f)) || ((this->y > 1.0f - this->h) && (this->ay > 0.0f))) {
        this->ay = -this->ay;
        pong = true;
    }

    if (pong) {
        double d = atan2(this->ay, this->ax);
        d += (double(::rand() % 2001 - 1000) / 1000.0f) * 0.5f;
        this->ax = float(cos(d));
        this->ay = float(sin(d));
    }

    vislib::graphics::CameraParamsTileRectOverride *params = 
        this->camera.Parameters().DynamicCast<vislib::graphics::CameraParamsTileRectOverride>();
    if (params) {
        params->SetTileRect(vislib::math::Rectangle<vislib::graphics::ImageSpaceType>(
            x * params->VirtualViewSize().Width(), 
            y * params->VirtualViewSize().Height(), 
            (x + w) * params->VirtualViewSize().Width(), 
            (y + h) * params->VirtualViewSize().Height()));
    }
}


/*
 * CamTestApp::Lens::BeginDraw
 */
void CamTestApp::Lens::BeginDraw(unsigned int ww, unsigned int wh, bool ortho) {
    glViewport(0, 0, ww, wh);

 //   this->camera.SetTileRectangle(
 //       vislib::math::Rectangle<vislib::graphics::ImageSpaceType>
 //       (this->x * this->camera.GetVirtualWidth(), this->y * this->camera.GetVirtualHeight(), 
 //       (this->x + this->w) * this->camera.GetVirtualWidth(), (this->y + this->h) * this->camera.GetVirtualHeight()));

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glTranslatef(-1.0f, -1.0f, 0.0f);
    glScalef(2.0f, 2.0f, 1.0f);
    glTranslatef(this->x, this->y, 0.0f);

    glDisable(GL_DEPTH_TEST);
    glColor3ub(255, 255, 0);
    glBegin(GL_LINE_LOOP);
        glVertex2f(0.0f, 0.0f);
        glVertex2f(this->w, 0.0f);
        glVertex2f(this->w, this->h);
        glVertex2f(0.0f, this->h);
    glEnd();
    glEnable(GL_DEPTH_TEST);

    glViewport(int(this->x * ww), int(this->y * wh), int(this->w * ww), int(this->h * wh));

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    this->camera.glMultProjectionMatrix();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    this->camera.glMultViewMatrix();
}


/*
 * CamTestApp::Lens::EndDraw
 */
void CamTestApp::Lens::EndDraw(void) {
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
}


/*
 * CamTestApp::CamTestApp
 */
CamTestApp::CamTestApp(void) : AbstractGlutApp() {
    this->lensCount = 7;
    this->lenses = new Lens[this->lensCount];
    this->walkSpeed = 0.25f;
    this->rotSpeed = 0.5f;

    this->ortho = false;
    this->nativeFull = true;

    vislib::SmartPtr<vislib::graphics::CameraParameters> params
        = this->camera.Parameters();
    for (unsigned int I = 0; I < this->lensCount; I++) {
        this->lenses[I].SetCameraParameters(params);
    }

    params->SetView(
        vislib::math::Point<float, 3>(0.0, -2.5, 0.0),
        vislib::math::Point<float, 3>(0.0, 0.0, 0.0),
        vislib::math::Vector<float, 3>(0.0, 0.0, 1.0));
    params->SetClip(1.0f, 5.0f);
    params->SetFocalDistance(2.5f);
    params->SetApertureAngle(40.0f);
    params->SetVirtualViewSize(10.0f, 10.0f);
    params->SetProjection(this->ortho 
        ? vislib::graphics::CameraParameters::MONO_ORTHOGRAPHIC
        : vislib::graphics::CameraParameters::MONO_PERSPECTIVE);
}


/*
 * CamTestApp::~CamTestApp
 */
CamTestApp::~CamTestApp(void) {
    delete[] this->lenses;
}


/*
 * CamTestApp::GLInit
 */
int CamTestApp::GLInit(void) {
    this->logo.Create();
    glEnable(GL_DEPTH_TEST);
    this->lastTime = vislib::sys::PerformanceCounter::Query();
    return 0;
}


/*
 * CamTestApp::GLDeinit
 */
void CamTestApp::GLDeinit(void) {
    this->logo.Release();
}


/*
 * CamTestApp::OnResize
 */
void CamTestApp::OnResize(unsigned int w, unsigned int h) {
    AbstractGlutApp::OnResize(w, h);
    this->camera.Parameters()->SetVirtualViewSize(
        float(w) / 500.0f, float(h) / 500.0f);
}


/*
 * CamTestApp::OnKeyPress
 */
bool CamTestApp::OnKeyPress(unsigned char key, int x, int y) {
    switch(key) {
        case 'o':
            this->ortho = !this->ortho;
            this->camera.Parameters()->SetProjection(this->ortho 
                ? vislib::graphics::CameraParameters::MONO_ORTHOGRAPHIC
                : vislib::graphics::CameraParameters::MONO_PERSPECTIVE);
            printf("Orthographic projection is %s\n", this->ortho ? "on" : "off");
            return true;
        case 'n':
            this->nativeFull = !this->nativeFull;
            printf("Native calls for full image are %s\n", this->nativeFull ? "on" : "off");
            return true;
        default: return false;            
    }
}


/*
 * CamTestApp::OnMouseEvent
 */
void CamTestApp::OnMouseEvent(int button, int state, int x, int y) {
}


/*
 * CamTestApp::OnMouseMove
 */
void CamTestApp::OnMouseMove(int x, int y) {
}


/*
 * CamTestApp::OnSpecialKey
 */
void CamTestApp::OnSpecialKey(int key, int x, int y) {
}


/*
 * CamTestApp::Render
 */
void CamTestApp::Render(void) {
    UINT64 time = vislib::sys::PerformanceCounter::Query();
    float sec = this->walkSpeed * float(time - this->lastTime) / 1000.0f;

    this->lastTime = time;
    this->angle = static_cast<float>(static_cast<int>(static_cast<float>(time) * this->rotSpeed) % 3600) * 0.1f;

    glViewport(0, 0, this->GetWidth(), this->GetHeight());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if (this->nativeFull) {
        if (this->ortho) {
            double w, h;
            w = 0.5 * this->camera.Parameters()->VirtualViewSize().Width();
            h = 0.5 * this->camera.Parameters()->VirtualViewSize().Height();
            glOrtho(-w, w, -h, h,
                this->camera.Parameters()->NearClip(), this->camera.Parameters()->FarClip());
        } else {
            gluPerspective(this->camera.Parameters()->ApertureAngle(), this->GetAspectRatio(),
                this->camera.Parameters()->NearClip(), this->camera.Parameters()->FarClip());
        }
    } else {
        this->camera.glMultProjectionMatrix();
    }

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    if (this->nativeFull) {
        const vislib::math::Point<double, 3> &pos = this->camera.Parameters()->Position();
        const vislib::math::Point<double, 3> &lat = this->camera.Parameters()->LookAt();
        const vislib::math::Vector<double, 3> &up = this->camera.Parameters()->Up();
        gluLookAt(pos.X(), pos.Y(), pos.Z(), lat.X(), lat.Y(), lat.Z(), up.X(), up.Y(), up.Z());

    } else {
        this->camera.glMultViewMatrix();
    }

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    this->logo.Draw();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glClear(GL_DEPTH_BUFFER_BIT);

    for (unsigned int i = 0; i < this->lensCount; i++) {
        this->lenses[i].Update(sec/*, this->camera*/);
        this->lenses[i].BeginDraw(this->GetWidth(), this->GetHeight(), this->ortho);
        this->logo.Draw();
        this->lenses[i].EndDraw();
    }

    glFlush();

    glutSwapBuffers();
    glutPostRedisplay();
}
