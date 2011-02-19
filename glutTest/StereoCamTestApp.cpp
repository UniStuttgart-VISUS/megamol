/*
 * StereoCamTestApp.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "StereoCamTestApp.h"

#include "vislib/CameraParamsEyeOverride.h"
#include "vislib/CameraParamsTileRectOverride.h"
#include "vislib/PerformanceCounter.h"

#include "vislibGlutInclude.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <cstdio>


// red-green-glasses
#define COLOR_MASK_LEFT_RED     GL_TRUE
#define COLOR_MASK_LEFT_GREEN   GL_FALSE
#define COLOR_MASK_LEFT_BLUE    GL_FALSE
#define COLOR_MASK_RIGHT_RED    GL_FALSE
#define COLOR_MASK_RIGHT_GREEN  GL_TRUE
#define COLOR_MASK_RIGHT_BLUE   GL_FALSE

//// red-blue-glasses
//#define COLOR_MASK_LEFT_RED     GL_TRUE
//#define COLOR_MASK_LEFT_GREEN   GL_FALSE
//#define COLOR_MASK_LEFT_BLUE    GL_FALSE
//#define COLOR_MASK_RIGHT_RED    GL_FALSE
//#define COLOR_MASK_RIGHT_GREEN  GL_TRUE
//#define COLOR_MASK_RIGHT_BLUE   GL_TRUE


StereoCamTestApp::StereoCamTestApp(void) : AbstractGlutApp() {

    this->parameters = this->cameraLeft.Parameters();
    this->parameters->SetView(
        vislib::math::Point<float, 3>(0.0, -10.0, 0.0),
        vislib::math::Point<float, 3>(0.0, 0.0, 0.0),
        vislib::math::Vector<float, 3>(0.0, 0.0, 1.0));
    this->parameters->SetClip(1.0f, 15.0f);
    this->parameters->SetApertureAngle(20.0f);
    this->parameters->SetVirtualViewSize(
        vislib::math::Dimension<vislib::graphics::ImageSpaceType, 2>(1.0f, 1.0f));
    this->parameters->SetStereoParameters(0.125f, 
        vislib::graphics::CameraParameters::LEFT_EYE, 10.0f);

    vislib::graphics::CameraParamsEyeOverride *rightEyeParams 
        = new vislib::graphics::CameraParamsEyeOverride(this->parameters);
    rightEyeParams->SetEye(vislib::graphics::CameraParameters::RIGHT_EYE);

#ifdef TILE_RIGHT_EYE
    vislib::SmartPtr<vislib::graphics::CameraParameters> reParams = rightEyeParams;
    for (int idx = 0; idx < TILE_RIGHT_EYE * TILE_RIGHT_EYE; idx++) {
        vislib::graphics::CameraParamsTileRectOverride *cptro
            = new vislib::graphics::CameraParamsTileRectOverride(reParams);
        cptro->SetTileRect(vislib::math::Rectangle<vislib::graphics::ImageSpaceType>
            (float((idx % TILE_RIGHT_EYE)) / float(TILE_RIGHT_EYE),
            float((idx / TILE_RIGHT_EYE)) / float(TILE_RIGHT_EYE),
            float((idx % TILE_RIGHT_EYE) + 1) / float(TILE_RIGHT_EYE),
            float((idx / TILE_RIGHT_EYE) + 1) / float(TILE_RIGHT_EYE)));
        this->cameraRight[idx].SetParameters(cptro);
    }

#else /* TILE_RIGHT_EYE */
    this->cameraRight.SetParameters(rightEyeParams);

#endif /* TILE_RIGHT_EYE */

    printf("Stereo Projection set to STEREO_OFF_AXIS\n");
    this->parameters->SetProjection(vislib::graphics::CameraParameters::STEREO_OFF_AXIS);
    float sd = 0.5f;
    printf("Stereo Disparity set to %f\n", sd);
    this->parameters->SetStereoDisparity(sd);

    //printf("Activate auto focus\n");
    //this->parameters->SetFocalDistance(0.0f);
    //this->parameters->SetAutoFocusOffset(-1.0f);

}

StereoCamTestApp::~StereoCamTestApp(void) {
}

int StereoCamTestApp::GLInit(void) {
    this->logo.Create();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    float lp[4] = {0.0f, -2.0f, 2.0f, 0.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lp);

    float la[4] = {0.1f, 0.1f, 0.1f, 1.0f};
    glLightfv(GL_LIGHT0, GL_AMBIENT, la);

    float ld[4] = {0.9f, 0.9f, 0.9f, 1.0f};
    glLightfv(GL_LIGHT0, GL_DIFFUSE, ld);

    glEnable(GL_COLOR_MATERIAL);

    this->lastTime = vislib::sys::PerformanceCounter::Query();
    return 0;
}

void StereoCamTestApp::GLDeinit(void) {
    glDisable(GL_LIGHTING);
    glDisable(GL_LIGHT0);
    glDisable(GL_COLOR_MATERIAL);
    this->logo.Release();
}

void StereoCamTestApp::OnResize(unsigned int w, unsigned int h) {
    AbstractGlutApp::OnResize(w, h);
    this->parameters->SetVirtualViewSize(float(w), float(h));
#ifdef TILE_RIGHT_EYE
    float fw = float(w) / float(TILE_RIGHT_EYE);
    float fh = float(h) / float(TILE_RIGHT_EYE);
    for (int idx = 0; idx < TILE_RIGHT_EYE * TILE_RIGHT_EYE; idx++) {
        vislib::graphics::CameraParamsTileRectOverride *cptro
            = this->cameraRight[idx].Parameters()
            .DynamicCast<vislib::graphics::CameraParamsTileRectOverride>();
        cptro->SetTileRect(vislib::math::Rectangle<vislib::graphics::ImageSpaceType>
            (float((idx % TILE_RIGHT_EYE)) * fw,
            float((idx / TILE_RIGHT_EYE)) * fh,
            float((idx % TILE_RIGHT_EYE) + 1) * fw,
            float((idx / TILE_RIGHT_EYE) + 1) * fh));
    }
#endif /* TILE_RIGHT_EYE */
}

void StereoCamTestApp::Render(void) {
    UINT64 time = vislib::sys::PerformanceCounter::Query();
    this->lastTime = time;
    this->angle = static_cast<float>(static_cast<int>(static_cast<float>(time) * 0.5f) % 3600) * 0.1f;

    glViewport(0, 0, this->GetWidth(), this->GetHeight());
    glColor4ub(255, 255, 255, 255);

    // render left eye image
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColorMask(COLOR_MASK_LEFT_RED, COLOR_MASK_LEFT_GREEN, COLOR_MASK_LEFT_BLUE, GL_TRUE);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    this->cameraLeft.glMultProjectionMatrix();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    this->cameraLeft.glMultViewMatrix();

    this->RenderTestBox();

    // render right eye image
    glClear(GL_DEPTH_BUFFER_BIT);
    glColorMask(COLOR_MASK_RIGHT_RED, COLOR_MASK_RIGHT_GREEN, COLOR_MASK_RIGHT_BLUE, GL_TRUE);
#ifdef TILE_RIGHT_EYE
    float w = float(this->GetWidth()) / float(TILE_RIGHT_EYE);
    float h = float(this->GetHeight()) / float(TILE_RIGHT_EYE);
    for (int idx = 0; idx < TILE_RIGHT_EYE * TILE_RIGHT_EYE; idx++) {
        glViewport(
            static_cast<GLint>(float((idx % TILE_RIGHT_EYE)) * w),
            static_cast<GLint>(float((idx / TILE_RIGHT_EYE)) * h),
            static_cast<GLsizei>(w), 
            static_cast<GLsizei>(h));
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        this->cameraRight[idx].glMultProjectionMatrix();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        this->cameraRight[idx].glMultViewMatrix();

        this->RenderTestBox();
    }
#else /* TILE_RIGHT_EYE */
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    this->cameraRight.glMultProjectionMatrix();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    this->cameraRight.glMultViewMatrix();

    this->RenderTestBox();
#endif /* TILE_RIGHT_EYE */

    // done rendering
    glViewport(0, 0, this->GetWidth(), this->GetHeight());
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay(); // because we do animation stuff
}

bool StereoCamTestApp::OnKeyPress(unsigned char key, int x, int y) {
    switch(key) {
        case '1':
            printf("Stereo Projection set to STEREO_PARALLEL\n");
            this->parameters->SetProjection(vislib::graphics::CameraParameters::STEREO_PARALLEL);
            break;
        case '2':
            printf("Stereo Projection set to STEREO_OFF_AXIS\n");
            this->parameters->SetProjection(vislib::graphics::CameraParameters::STEREO_OFF_AXIS);
            break;
        case '3':
            printf("Stereo Projection set to STEREO_TOE_IN\n");
            this->parameters->SetProjection(vislib::graphics::CameraParameters::STEREO_TOE_IN);
            break;
        case 'a': {
            vislib::graphics::SceneSpaceType sd = this->parameters->StereoDisparity();
            sd /= 1.2f;
            this->parameters->SetStereoDisparity(sd);
            printf("Stereo Disparity set to %f\n", sd);
        } break;
        case 'y': {
            vislib::graphics::SceneSpaceType sd = this->parameters->StereoDisparity();
            sd *= 1.2f;
            this->parameters->SetStereoDisparity(sd);
            printf("Stereo Disparity set to %f\n", sd);
        } break;
        case 's': {
            vislib::graphics::SceneSpaceType sd = this->parameters->FocalDistance();
            sd -= 0.1f;
            this->parameters->SetFocalDistance(sd);
            printf("Focal distance set to %f\n", sd);
        } break;
        case 'x': {
            vislib::graphics::SceneSpaceType sd = this->parameters->FocalDistance();
            sd += 0.1f;
            this->parameters->SetFocalDistance(sd);
            printf("Focal distance set to %f\n", sd);
        } break;

        default:
            return AbstractGlutApp::OnKeyPress(key, x, y);
    }
    glutPostRedisplay();
    return true;
}

void StereoCamTestApp::RenderTestBox(void) {
    glDisable(GL_LIGHTING);
    glBegin(GL_LINE_LOOP);
        glVertex3i(-1, -1, -1);
        glVertex3i( 1, -1, -1);
        glVertex3i( 1,  1, -1);
        glVertex3i(-1,  1, -1);
    glEnd();
    glBegin(GL_LINE_LOOP);
        glVertex3i(-1, -1,  1);
        glVertex3i( 1, -1,  1);
        glVertex3i( 1,  1,  1);
        glVertex3i(-1,  1,  1);
    glEnd();
    glBegin(GL_LINES);
        glVertex3i(-1, -1, -1);
        glVertex3i(-1, -1,  1);
        glVertex3i( 1, -1, -1);
        glVertex3i( 1, -1,  1);
        glVertex3i(-1,  1, -1);
        glVertex3i(-1,  1,  1);
        glVertex3i( 1,  1, -1);
        glVertex3i( 1,  1,  1);
    glEnd();

    glBegin(GL_POINTS);
    for (float y = -1.0f; y < 1.0f; y += 0.2f)
        glVertex3f(0.0f, y, -1.0f);
    glEnd();

    glEnable(GL_LIGHTING);

    glPushMatrix();
    glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
    glRotatef(this->angle, 0.0f, -1.0f, 0.0f);

    this->logo.Draw();

    glPopMatrix();
}
