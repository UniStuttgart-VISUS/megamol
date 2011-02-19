/*
 * CamEulerRotatorTestApp.cpp
 *
 * Copyright (C) 2006-2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */
#include "CamEulerRotatorTestApp.h"

#include "vislibGlutInclude.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <cstdio>

#include "vislib/graphicstypes.h"
#include "vislib/ObservableCameraParams.h"
#include "vislib/Rectangle.h"


/*
 * CamEulerRotatorTestApp::CamEulerRotatorTestApp
 */
CamEulerRotatorTestApp::CamEulerRotatorTestApp(void) : AbstractGlutApp(),
        camera(new vislib::graphics::ObservableCameraParams()) {
    using namespace vislib::graphics;

#ifdef REGISTER_TEST_OBSERVER
    this->camera.Parameters().DynamicCast<ObservableCameraParams>()
        ->AddCameraParameterObserver(&this->testObserver);
#endif /* REGISTER_TEST_OBSERVER */
    this->camera.Parameters()->SetClip(0.1f, 7.0f);
    this->camera.Parameters()->SetFocalDistance(2.5f);
    this->camera.Parameters()->SetApertureAngle(50.0f);
    this->camera.Parameters()->SetView(
        vislib::math::Point<double, 3>(0.0, -2.5, 0.0),
        vislib::math::Point<double, 3>(0.0, 0.0, 0.0),
        vislib::math::Vector<double, 3>(0.0, 0.0, 1.0));
    this->camera.Parameters()->SetVirtualViewSize(10.0f, 10.0f);
    this->camera.Parameters()->SetProjection(
        vislib::graphics::CameraParameters::MONO_PERSPECTIVE);

    this->modkeys.SetModifierCount(3);
    this->modkeys.RegisterObserver(&this->cursor);

    this->cursor.SetButtonCount(3);
    this->cursor.SetInputModifiers(&this->modkeys);
    this->cursor.SetCameraParams(this->camera.Parameters());
    
    this->rotator.SetCameraParams(this->camera.Parameters());
    this->rotator.SetTestButton(0); // left button
    this->rotator.SetModifierTestCount(0);
    this->rotator.SetAltModifier(
        vislib::graphics::InputModifiers::MODIFIER_SHIFT);
    this->cursor.RegisterCursorEvent(&this->rotator);

    this->mover.SetCameraParams(this->camera.Parameters());
    this->mover.SetTestButton(2); // middle button
    this->mover.SetModifierTestCount(0);
    this->cursor.RegisterCursorEvent(&this->mover);

    this->camera.Parameters()->SetView(
        vislib::math::Point<double, 3>(0.0, -2.5, 0.0),
        vislib::math::Point<double, 3>(0.0, 0.0, 0.0),
        vislib::math::Vector<double, 3>(0.0, 0.0, 1.0));

}


/*
 * CamEulerRotatorTestApp::~CamEulerRotatorTestApp
 */
CamEulerRotatorTestApp::~CamEulerRotatorTestApp(void) {
}


/*
 * CamEulerRotatorTestApp::GLInit
 */
int CamEulerRotatorTestApp::GLInit(void) {
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    this->camera.glMultViewMatrix();

    float lp[4] = {-2.0f, -2.0f, 2.0f, 0.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lp);

    float la[4] = {0.1f, 0.1f, 0.1f, 1.0f};
    glLightfv(GL_LIGHT0, GL_AMBIENT, la);

    float ld[4] = {0.9f, 0.9f, 0.9f, 1.0f};
    glLightfv(GL_LIGHT0, GL_DIFFUSE, ld);

    glEnable(GL_COLOR_MATERIAL);

    this->logo.Create();

    this->rotator.ResetOrientation();
    this->camera.Parameters()->SetView(
        vislib::math::Point<double, 3>(0.0, -2.5, 0.0),
        vislib::math::Point<double, 3>(0.0, 0.0, 0.0),
        vislib::math::Vector<double, 3>(0.0, 0.0, 1.0));

    printf("\n");
    printf("Keys used by CamEulerRotatorTestApp:\n");
    printf("========================================\n");
    printf("\tSHIFT\tCamera Rolls\n");
    printf("\tHome\tReset Euler Rotation to Base Orientation\n");
    printf("\tEnter\tSet Base Orientation to current orientation\n");
    printf("\tr\tDe-/Activates update of Base Orientation on roll\n");
    printf("\n");

    return 0;
}


/*
 * CamEulerRotatorTestApp::GLDeinit
 */
void CamEulerRotatorTestApp::GLDeinit(void) {
    glDisable(GL_LIGHT0);
    glDisable(GL_LIGHTING);
    this->logo.Release();
}


/*
 * CamEulerRotatorTestApp::OnResize
 */
void CamEulerRotatorTestApp::OnResize(unsigned int w, unsigned int h) {
    AbstractGlutApp::OnResize(w, h);
    this->camera.Parameters()->SetVirtualViewSize(
        static_cast<vislib::graphics::ImageSpaceType>(w),
        static_cast<vislib::graphics::ImageSpaceType>(h));
}


/*
 * CamEulerRotatorTestApp::OnKeyPress
 */
bool CamEulerRotatorTestApp::OnKeyPress(unsigned char key, int x, int y) {
    bool retval = true;
    switch(key) {
        case 'r':
            this->rotator.SetBaseOrientationOnRoll(!this->rotator.GetSetBaseOrientationOnRoll());
            if (this->rotator.GetSetBaseOrientationOnRoll()) {
                printf("Base Orientation will be updated when camera rolls\n");
            } else {
                printf("Base Orientation will remain unchanged when camera rolls\n");
            }
            break;
        case 13:
            this->rotator.UpdateBaseOrientation();
            printf("Base Orientation updated\n");
            break;
        default: retval = false; break;
    }
    this->cursor.SetPosition(static_cast<vislib::graphics::ImageSpaceType>(x), 
        static_cast<vislib::graphics::ImageSpaceType>(y), true);
    return retval;
}


/*
 * CamEulerRotatorTestApp::OnMouseEvent
 */
void CamEulerRotatorTestApp::OnMouseEvent(int button, int state, int x, int y) {
    unsigned int btn = 0;
    int modifiers = glutGetModifiers();

    switch (button) {
        case GLUT_LEFT_BUTTON: btn = 0; break;
        case GLUT_RIGHT_BUTTON: btn = 1; break;
        case GLUT_MIDDLE_BUTTON: btn = 2; break;
    }

    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_SHIFT, (modifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT);
    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_CTRL, (modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL);
    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_ALT, (modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT);
    this->cursor.SetPosition(static_cast<vislib::graphics::ImageSpaceType>(x), 
        static_cast<vislib::graphics::ImageSpaceType>(y), true);
    this->cursor.SetButtonState(btn, (state == GLUT_DOWN));
}


/*
 * CamEulerRotatorTestApp::OnMouseMove
 */
void CamEulerRotatorTestApp::OnMouseMove(int x, int y) {
    this->cursor.SetPosition(static_cast<vislib::graphics::ImageSpaceType>(x), 
        static_cast<vislib::graphics::ImageSpaceType>(y), true);
}


/*
 * CamEulerRotatorTestApp::OnSpecialKey
 */
void CamEulerRotatorTestApp::OnSpecialKey(int key, int x, int y) {
    int modifiers = glutGetModifiers();

    switch (key) {
        case GLUT_KEY_HOME:
            this->rotator.ResetOrientation();
            printf("Orientation resetted to Base Orientation\n");
            break;
        default: break;
    }

    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_SHIFT, (modifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT);
    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_CTRL, (modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL);
    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_ALT, (modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT);
    this->cursor.SetPosition(static_cast<vislib::graphics::ImageSpaceType>(x), 
        static_cast<vislib::graphics::ImageSpaceType>(y), true);
}


/*
 * CamEulerRotatorTestApp::Render
 */
void CamEulerRotatorTestApp::Render(void) {

    glViewport(0, 0, this->GetWidth(), this->GetHeight());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    this->camera.glMultProjectionMatrix();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    this->camera.glMultViewMatrix();

    this->RenderLogo();

    glFlush();

    glutSwapBuffers();
    glutPostRedisplay();
}


/*
 * CamEulerRotatorTestApp::RenderLogo
 */
void CamEulerRotatorTestApp::RenderLogo(void) {
    this->logo.Draw();

    // Draw Axis Markers:
    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);

        glColor3ub(255, 0, 0);
        glVertex3f(2.0f, -0.2f, 0.0f); glVertex3f(2.0f, 0.2f, 0.0f);
        glVertex3f(2.0f, 0.0f, -0.2f); glVertex3f(2.0f, 0.0f, 0.2f);

        glColor3ub(0, 255, 0);
        glVertex3f(-0.2f, 2.0f, 0.0f); glVertex3f(0.2f, 2.0f, 0.0f);
        glVertex3f(0.0f, 2.0f, -0.2f); glVertex3f(0.0f, 2.0f, 0.2f);

        glColor3ub(0, 0, 255);
        glVertex3f(-0.2f, 0.0f, 2.0f); glVertex3f(0.2f, 0.0f, 2.0f);
        glVertex3f(0.0f, -0.2f, 2.0f); glVertex3f(0.0f, 0.2f, 2.0f);
    glEnd();
    glColor3ub(255, 0, 0);
    glBegin(GL_LINE_LOOP);
        glVertex3f(-2.0f, -0.2f, 0.0f); glVertex3f(-2.0f, 0.0f, -0.2f); glVertex3f(-2.0f, 0.2f, 0.0f); glVertex3f(-2.0f, 0.0f, 0.2f);
    glEnd();
    glColor3ub(0, 255, 0);
    glBegin(GL_LINE_LOOP);
        glVertex3f(-0.2f, -2.0f, 0.0f); glVertex3f(0.0f, -2.0f, -0.2f); glVertex3f(0.2f, -2.0f, 0.0f); glVertex3f(0.0f, -2.0f, 0.2f);
    glEnd();
    glColor3ub(0, 0, 255);
    glBegin(GL_LINE_LOOP);
        glVertex3f(-0.2f, 0.0f, -2.0f); glVertex3f(0.0f, -0.2f, -2.0f); glVertex3f(0.2f, 0.0f, -2.0f); glVertex3f(0.0f, 0.2f, -2.0f);
    glEnd();

    glEnable(GL_LIGHTING);
}
