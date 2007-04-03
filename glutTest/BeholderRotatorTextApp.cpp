/*
 * BeholderRotatorTextApp.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */
#include "BeholderRotatorTextApp.h"

#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <cstdio>

#include "vislib/graphicstypes.h"
#include "vislib/Rectangle.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include "vislogo.h"
#include <cstdlib>


/*
 * BeholderRotatorTextApp::BeholderRotatorTextApp
 */
BeholderRotatorTextApp::BeholderRotatorTextApp(void) : AbstractGlutApp() {

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

    this->cursor.SetButtonCount(3);
    this->cursor.SetInputModifiers(&this->modkeys);
    this->cursor.SetCamera(&this->camera);
    
    this->rotator1.SetBeholder(&this->beholder);
    this->rotator1.SetTestButton(0); // left button
    this->rotator1.SetModifierTestCount(0);
    this->rotator1.SetAltModifier(vislib::graphics::InputModifiers::MODIFIER_CTRL);

    this->rotator2.SetBeholder(&this->beholder);
    this->rotator2.SetTestButton(0); // left button
    this->rotator2.SetModifierTestCount(0);
    this->rotator2.SetAltModifier(vislib::graphics::InputModifiers::MODIFIER_CTRL);

    this->SetupRotator2();
}


/*
 * BeholderRotatorTextApp::~BeholderRotatorTextApp
 */
BeholderRotatorTextApp::~BeholderRotatorTextApp(void) {
}


/*
 * BeholderRotatorTextApp::GLInit
 */
int BeholderRotatorTextApp::GLInit(void) {
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

    return 0;
}


/*
 * BeholderRotatorTextApp::OnResize
 */
void BeholderRotatorTextApp::OnResize(unsigned int w, unsigned int h) {
    AbstractGlutApp::OnResize(w, h);
    this->camera.SetVirtualWidth(w);
    this->camera.SetVirtualHeight(h);
}


/*
 * BeholderRotatorTextApp::OnKeyPress
 */
bool BeholderRotatorTextApp::OnKeyPress(unsigned char key, int x, int y) {
    bool retval = true;
    switch(key) {
        case '1': this->SetupRotator1(); break;
        case '2': this->SetupRotator2(); break;
        default: retval = false; break;
    }
    this->cursor.SetPosition(x, y, true);
    return retval;
}


/*
 * BeholderRotatorTextApp::OnMouseEvent
 */
void BeholderRotatorTextApp::OnMouseEvent(int button, int state, int x, int y) {
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
    this->cursor.SetPosition(x, y, true);
    this->cursor.SetButtonState(btn, (state == GLUT_DOWN));
}


/*
 * BeholderRotatorTextApp::OnMouseMove
 */
void BeholderRotatorTextApp::OnMouseMove(int x, int y) {
    this->cursor.SetPosition(x, y, true);
}


/*
 * BeholderRotatorTextApp::OnSpecialKey
 */
void BeholderRotatorTextApp::OnSpecialKey(int key, int x, int y) {
    int modifiers = glutGetModifiers();

    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_SHIFT, (modifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT);
    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_CTRL, (modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL);
    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_ALT, (modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT);
    this->cursor.SetPosition(x, y, true);
}


/*
 * BeholderRotatorTextApp::Render
 */
void BeholderRotatorTextApp::Render(void) {

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
 * BeholderRotatorTextApp::RenderLogo
 */
void BeholderRotatorTextApp::RenderLogo(void) {
    unsigned int vCount = VisLogoCountVertices();
	unsigned int p;

    glBegin(GL_QUAD_STRIP);

    for (unsigned int i = 0; i < 20; i++) {
		for (unsigned int j = 0; j < vCount / 20; j++) {
			p = (i + j * 20) % vCount;
			glColor3dv(VisLogoVertexColor(p)->f);
			glNormal3dv(VisLogoVertexNormal(p)->f);
			glVertex3dv(VisLogoVertex(p)->f);

            p = ((i + 1) % 20 + j * 20) % vCount;
			glColor3dv(VisLogoVertexColor(p)->f);
			glNormal3dv(VisLogoVertexNormal(p)->f);
			glVertex3dv(VisLogoVertex(p)->f);
		}
	}

    p = 0; // closing strip
	glColor3dv(VisLogoVertexColor(p)->f);
	glNormal3dv(VisLogoVertexNormal(p)->f);
	glVertex3dv(VisLogoVertex(p)->f);

    p = 1;
	glColor3dv(VisLogoVertexColor(p)->f);
	glNormal3dv(VisLogoVertexNormal(p)->f);
	glVertex3dv(VisLogoVertex(p)->f);

    glEnd();    

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


/*
 * BeholderRotatorTextApp::SetupRotator1
 */
void BeholderRotatorTextApp::SetupRotator1(void) {
    this->beholder.SetView(
        vislib::math::Point<double, 3>(0.0, 0.0, 0.0),
        vislib::math::Point<double, 3>(0.0, 1.0, 0.0),
        vislib::math::Vector<double, 3>(0.0, 0.0, 1.0));

    this->cursor.UnregisterCursorEvent(&this->rotator2);
    this->cursor.RegisterCursorEvent(&this->rotator1);
}


/*
 * BeholderRotatorTextApp::SetupRotator2
 */
void BeholderRotatorTextApp::SetupRotator2(void) {
    this->beholder.SetView(
        vislib::math::Point<double, 3>(0.0, -2.5, 0.0),
        vislib::math::Point<double, 3>(0.0, 0.0, 0.0),
        vislib::math::Vector<double, 3>(0.0, 0.0, 1.0));

    this->cursor.UnregisterCursorEvent(&this->rotator1);
    this->cursor.RegisterCursorEvent(&this->rotator2);
}
