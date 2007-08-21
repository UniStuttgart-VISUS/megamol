/*
 * GLSLGeomShaderTest.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */
#include "GLSLGeomShaderTest.h"

#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <cstdio>

#include "vislib/graphicstypes.h"
#include "vislib/Rectangle.h"
#include "vislib/sysfunctions.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include "vislogo.h"
#include <cstdlib>


/*
 * GLSLGeomShaderTest::GLSLGeomShaderTest
 */
GLSLGeomShaderTest::GLSLGeomShaderTest(void) : AbstractGlutApp(), schade() {

    this->beholder.SetView(
        vislib::math::Point<double, 3>(0.0, -2.5, 0.0),
        vislib::math::Point<double, 3>(0.0, 0.0, 0.0),
        vislib::math::Vector<double, 3>(0.0, 0.0, 1.0));

    this->camera.SetBeholder(&this->beholder);
    this->camera.SetNearClipDistance(0.1f);
    this->camera.SetFarClipDistance(7.0f);
    this->camera.SetFocalDistance(2.5f);
    this->camera.SetApertureAngle(50.0f);
    this->camera.SetVirtualWidth(10.0f);
    this->camera.SetVirtualHeight(10.0f);
    this->camera.SetProjectionType(vislib::graphics::Camera::MONO_PERSPECTIVE);

    this->modkeys.SetModifierCount(3);
    this->modkeys.RegisterObserver(&this->cursor);

    this->rotator2.SetBeholder(&this->beholder);
    this->rotator2.SetTestButton(0); // left button
    this->rotator2.SetModifierTestCount(0);
    this->rotator2.SetAltModifier(vislib::graphics::InputModifiers::MODIFIER_CTRL);

    this->cursor.SetButtonCount(3);
    this->cursor.SetInputModifiers(&this->modkeys);
    this->cursor.SetCamera(&this->camera);
    this->cursor.RegisterCursorEvent(&this->rotator2);
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

    VisLogoDoStuff();
    VisLogoTwistLogo();
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

    this->beholder.SetView(
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
    this->camera.SetVirtualWidth(static_cast<vislib::graphics::ImageSpaceType>(w));
    this->camera.SetVirtualHeight(static_cast<vislib::graphics::ImageSpaceType>(h));
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

////    glUniform1fARB(this->schade.ParameterLocation("v"), 1, viewportStuff);
//    float angle = M_PI * float(vislib::sys::GetTicksOfDay() % 2000) / 1000.0f;
//    this->schade.SetParameter("v", sinf(angle), cos(angle));
//
//    unsigned int vCount = VisLogoCountVertices();
//	unsigned int p;
//
//    glBegin(GL_QUAD_STRIP);
//
//    for (unsigned int i = 0; i < 20; i++) {
//		for (unsigned int j = 0; j < vCount / 20; j++) {
//			p = (i + j * 20) % vCount;
//			glColor3dv(VisLogoVertexColor(p)->f);
//			glNormal3dv(VisLogoVertexNormal(p)->f);
//			glVertex3dv(VisLogoVertex(p)->f);
//
//            p = ((i + 1) % 20 + j * 20) % vCount;
//			glColor3dv(VisLogoVertexColor(p)->f);
//			glNormal3dv(VisLogoVertexNormal(p)->f);
//			glVertex3dv(VisLogoVertex(p)->f);
//		}
//	}
//
//    p = 0; // closing strip
//	glColor3dv(VisLogoVertexColor(p)->f);
//	glNormal3dv(VisLogoVertexNormal(p)->f);
//	glVertex3dv(VisLogoVertex(p)->f);
//
//    p = 1;
//	glColor3dv(VisLogoVertexColor(p)->f);
//	glNormal3dv(VisLogoVertexNormal(p)->f);
//	glVertex3dv(VisLogoVertex(p)->f);

    //glEnd();

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
