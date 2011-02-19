/*
 * GLSLGeomShaderTest.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */
#define _USE_MATH_DEFINES
#include "GLSLGeomShaderTest.h"

#include "vislibGlutInclude.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <cstdio>

#include "vislib/graphicstypes.h"
#include "vislib/Rectangle.h"
#include "vislib/sysfunctions.h"


/*
 * GLSLGeomShaderTest::GLSLGeomShaderTest
 */
GLSLGeomShaderTest::GLSLGeomShaderTest(void) : AbstractGlutApp(), schade() {

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
 * GLSLGeomShaderTest::~GLSLGeomShaderTest
 */
GLSLGeomShaderTest::~GLSLGeomShaderTest(void) {
}


/*
 * GLSLGeomShaderTest::GLInit
 */
int GLSLGeomShaderTest::GLInit(void) {

    if (!::vislib::graphics::gl::GLSLGeometryShader::InitialiseExtensions()) {
        return -12;
    }

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

    if (!this->schade.Compile(
/* Vertex Shader*/
"#version 120\n"
"#extension GL_EXT_gpu_shader4:enable\n"
"\n"
"void main(void) {\n"
"    gl_Position = ftransform();\n"
"}\n",
/* Geometry Shader */
"#version 120\n"
"#extension GL_EXT_gpu_shader4:enable\n"
"#extension GL_EXT_geometry_shader4 : enable\n"
"\n"
"#define DELTA 0.5\n"
"#define DOUBLEDELTA 1.0\n"
"\n"
"void main(void) {\n"
"    gl_Position = gl_PositionIn[0];\n"
"    gl_Position.y -= DELTA;\n"
"    EmitVertex();\n"
"\n"
"    gl_Position.y += DOUBLEDELTA;\n"
"    gl_Position.x -= DELTA;\n"
"    EmitVertex();\n"
"\n"
"    gl_Position.x += DOUBLEDELTA;\n"
"    EmitVertex();\n"
"\n"
"    EndPrimitive();\n"
"}\n",
/* Fragment Shader */
"#version 120\n"
"void main(void) {\n"
"  gl_FragColor = vec4(0.25, 0.5, 1.0, 1.0);\n"
"}\n")) {
        return -13;
    }

    //glProgramParameteriEXT(this->schade, GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS);
    //glProgramParameteriEXT(this->schade, GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
    //glProgramParameteriEXT(this->schade, GL_GEOMETRY_VERTICES_OUT_EXT, 3);
    this->schade.SetProgramParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS);
    this->schade.SetProgramParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
    this->schade.SetProgramParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 3);

    if (!this->schade.Link()) {
        return -14;
    }

    this->camera.Parameters()->SetView(
        vislib::math::Point<double, 3>(0.0, -2.5, 0.0),
        vislib::math::Point<double, 3>(0.0, 0.0, 0.0),
        vislib::math::Vector<double, 3>(0.0, 0.0, 1.0));

    return 0;
}


/*
 * GLSLGeomShaderTest::GLDeinit
 */
void GLSLGeomShaderTest::GLDeinit(void) {
    this->schade.Release();
}


/*
 * GLSLGeomShaderTest::OnResize
 */
void GLSLGeomShaderTest::OnResize(unsigned int w, unsigned int h) {
    AbstractGlutApp::OnResize(w, h);
    this->camera.Parameters()->SetVirtualViewSize(
        static_cast<vislib::graphics::ImageSpaceType>(w),
        static_cast<vislib::graphics::ImageSpaceType>(h));
}


/*
 * GLSLGeomShaderTest::OnKeyPress
 */
bool GLSLGeomShaderTest::OnKeyPress(unsigned char key, int x, int y) {
    this->cursor.SetPosition(static_cast<vislib::graphics::ImageSpaceType>(x), 
        static_cast<vislib::graphics::ImageSpaceType>(y), true);
    return false; // retval;
}


/*
 * GLSLGeomShaderTest::OnMouseEvent
 */
void GLSLGeomShaderTest::OnMouseEvent(int button, int state, int x, int y) {
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
 * GLSLGeomShaderTest::OnMouseMove
 */
void GLSLGeomShaderTest::OnMouseMove(int x, int y) {
    this->cursor.SetPosition(static_cast<vislib::graphics::ImageSpaceType>(x), 
        static_cast<vislib::graphics::ImageSpaceType>(y), true);
}


/*
 * GLSLGeomShaderTest::OnSpecialKey
 */
void GLSLGeomShaderTest::OnSpecialKey(int key, int x, int y) {
    int modifiers = glutGetModifiers();

    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_SHIFT, (modifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT);
    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_CTRL, (modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL);
    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_ALT, (modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT);
    this->cursor.SetPosition(static_cast<vislib::graphics::ImageSpaceType>(x), 
        static_cast<vislib::graphics::ImageSpaceType>(y), true);
}


/*
 * GLSLGeomShaderTest::Render
 */
void GLSLGeomShaderTest::Render(void) {

    glViewport(0, 0, this->GetWidth(), this->GetHeight());
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    this->camera.glMultProjectionMatrix();

    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
    this->camera.glMultViewMatrix();

    this->schade.Enable();

    glBegin(GL_POINTS);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f( 1.0f, 0.0f, 0.0f);
    glVertex3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f,  1.0f, 0.0f);
    glVertex3f(0.0f, -1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f,  1.0f);
    glVertex3f(0.0f, 0.0f, -1.0f);
    glEnd();

    this->schade.Disable();

    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);
        glVertex3i( 1,  1,  1); glVertex3i(-1,  1,  1);
        glVertex3i( 1, -1,  1); glVertex3i(-1, -1,  1);
        glVertex3i( 1,  1, -1); glVertex3i(-1,  1, -1);
        glVertex3i( 1, -1, -1); glVertex3i(-1, -1, -1);

        glVertex3i( 1,  1,  1); glVertex3i( 1, -1,  1);
        glVertex3i(-1,  1,  1); glVertex3i(-1, -1,  1);
        glVertex3i( 1,  1, -1); glVertex3i( 1, -1, -1);
        glVertex3i(-1,  1, -1); glVertex3i(-1, -1, -1);

        glVertex3i( 1,  1,  1); glVertex3i( 1,  1, -1);
        glVertex3i(-1,  1,  1); glVertex3i(-1,  1, -1);
        glVertex3i( 1, -1,  1); glVertex3i( 1, -1, -1);
        glVertex3i(-1, -1,  1); glVertex3i(-1, -1, -1);
    glEnd();
    glEnable(GL_LIGHTING);
                      
	glFlush();

	glutSwapBuffers();
    glutPostRedisplay();
}
