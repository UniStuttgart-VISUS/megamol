/*
 * GLSLShaderTest.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */
#define _USE_MATH_DEFINES
#include "GLSLShaderTest.h"

#include "vislibGlutInclude.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <cstdio>

#include "vislib/graphicstypes.h"
#include "vislib/Rectangle.h"
#include "vislib/sysfunctions.h"


/*
 * GLSLShaderTest::GLSLShaderTest
 */
GLSLShaderTest::GLSLShaderTest(void) : AbstractGlutApp(), schade() {

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

    this->rotator.SetCameraParams(this->camera.Parameters());
    this->rotator.SetTestButton(0); // left button
    this->rotator.SetModifierTestCount(0);
    this->rotator.SetAltModifier(
        vislib::graphics::InputModifiers::MODIFIER_SHIFT);

    this->cursor.SetButtonCount(3);
    this->cursor.SetInputModifiers(&this->modkeys);
    this->cursor.SetCameraParams(this->camera.Parameters());
    this->cursor.RegisterCursorEvent(&this->rotator);
}


/*
 * GLSLShaderTest::~GLSLShaderTest
 */
GLSLShaderTest::~GLSLShaderTest(void) {
}


/*
 * GLSLShaderTest::GLInit
 */
int GLSLShaderTest::GLInit(void) {

    if (!::vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return -12;
    }

    this->logo.Create();
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

    if (!this->schade.Create(
            vislib::graphics::gl::GLSLShader::FTRANSFORM_VERTEX_SHADER_SRC,
"\n"
"uniform vec2 v;\n"
"\n"
"void main(void) {\n"
"  float alpha = v.x * v.x;\n"
"  float beta = 1.0 - alpha;\n"
"  float scale = (1.0 - gl_FragCoord.z) * 40.0;\n"
"  alpha *= scale;\n"
"  beta *= scale;\n"
"  gl_FragColor = alpha * vec4(0.25, 0.5, 1.0, 1.0) + beta * vec4(0.5, 1.0, 0.25, 1.0);\n"
"}\n")) {
        return -13;
    }

    this->camera.Parameters()->SetView(
        vislib::math::Point<double, 3>(0.0, -2.5, 0.0),
        vislib::math::Point<double, 3>(0.0, 0.0, 0.0),
        vislib::math::Vector<double, 3>(0.0, 0.0, 1.0));

    return 0;
}


/*
 * GLSLShaderTest::GLDeinit
 */
void GLSLShaderTest::GLDeinit(void) {
    this->schade.Release();
    this->logo.Release();
}


/*
 * GLSLShaderTest::OnResize
 */
void GLSLShaderTest::OnResize(unsigned int w, unsigned int h) {
    AbstractGlutApp::OnResize(w, h);
    this->camera.Parameters()->SetVirtualViewSize(
        static_cast<vislib::graphics::ImageSpaceType>(w),
        static_cast<vislib::graphics::ImageSpaceType>(h));
}


/*
 * GLSLShaderTest::OnKeyPress
 */
bool GLSLShaderTest::OnKeyPress(unsigned char key, int x, int y) {
    this->cursor.SetPosition(static_cast<vislib::graphics::ImageSpaceType>(x), 
        static_cast<vislib::graphics::ImageSpaceType>(y), true);
    return false; // retval;
}


/*
 * GLSLShaderTest::OnMouseEvent
 */
void GLSLShaderTest::OnMouseEvent(int button, int state, int x, int y) {
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
 * GLSLShaderTest::OnMouseMove
 */
void GLSLShaderTest::OnMouseMove(int x, int y) {
    this->cursor.SetPosition(static_cast<vislib::graphics::ImageSpaceType>(x), 
        static_cast<vislib::graphics::ImageSpaceType>(y), true);
}


/*
 * GLSLShaderTest::OnSpecialKey
 */
void GLSLShaderTest::OnSpecialKey(int key, int x, int y) {
    int modifiers = glutGetModifiers();

    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_SHIFT, (modifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT);
    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_CTRL, (modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL);
    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_ALT, (modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT);
    this->cursor.SetPosition(static_cast<vislib::graphics::ImageSpaceType>(x), 
        static_cast<vislib::graphics::ImageSpaceType>(y), true);
}


/*
 * GLSLShaderTest::Render
 */
void GLSLShaderTest::Render(void) {

    glViewport(0, 0, this->GetWidth(), this->GetHeight());
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    this->camera.glMultProjectionMatrix();

    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
    this->camera.glMultViewMatrix();

    this->schade.Enable();

    float angle = float(M_PI) * float(vislib::sys::GetTicksOfDay() % 2000) / 1000.0f;
    float value[2] = { sin(angle), cos(angle) };

//    this->schade.SetParameter("v", value[0], value[1]);
    this->schade.SetParameterArray2("v", 1, value);

    this->logo.Draw();

    this->schade.Disable();

	glFlush();

	glutSwapBuffers();
    glutPostRedisplay();
}
