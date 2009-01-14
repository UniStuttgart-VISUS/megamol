/*
 * SimpleFontTest.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */
#include "SimpleFontTest.h"

#include "vislibGlutInclude.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <cstdio>

#include "vislib/types.h"
#include "vislib/graphicstypes.h"
#include "vislib/Rectangle.h"
#include "vislib/sysfunctions.h"
#include "vislib/SimpleFont.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <cstdlib>


/*
 * SimpleFontTest::SimpleFontTest
 */
SimpleFontTest::SimpleFontTest(void) : AbstractGlutApp(), font(NULL) {

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
 * SimpleFontTest::~SimpleFontTest
 */
SimpleFontTest::~SimpleFontTest(void) {
    // intentionally empty
}


/*
 * SimpleFontTest::GLInit
 */
int SimpleFontTest::GLInit(void) {
    this->camera.Parameters()->SetView(
        vislib::math::Point<double, 3>(0.0, -2.5, 0.0),
        vislib::math::Point<double, 3>(0.0, 0.0, 0.0),
        vislib::math::Vector<double, 3>(0.0, 0.0, 1.0));

    if (this->font == NULL) {
            this->font = new vislib::graphics::gl::SimpleFont();
    }
    if (!this->font->Initialise()) return -2;

    ::glEnable(GL_TEXTURE_2D);
    ::glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
//    ::glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    return 0;
}


/*
 * SimpleFontTest::GLDeinit
 */
void SimpleFontTest::GLDeinit(void) {
    if (this->font != NULL) {
        this->font->Deinitialise();
        delete this->font;
        this->font = NULL;
    }
}


/*
 * SimpleFontTest::OnResize
 */
void SimpleFontTest::OnResize(unsigned int w, unsigned int h) {
    AbstractGlutApp::OnResize(w, h);
    this->camera.Parameters()->SetVirtualViewSize(
        static_cast<vislib::graphics::ImageSpaceType>(w),
        static_cast<vislib::graphics::ImageSpaceType>(h));
}


/*
 * SimpleFontTest::OnKeyPress
 */
bool SimpleFontTest::OnKeyPress(unsigned char key, int x, int y) {
    this->cursor.SetPosition(static_cast<vislib::graphics::ImageSpaceType>(x), 
        static_cast<vislib::graphics::ImageSpaceType>(y), true);
    return false; // retval;
}


/*
 * SimpleFontTest::OnMouseEvent
 */
void SimpleFontTest::OnMouseEvent(int button, int state, int x, int y) {
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
 * SimpleFontTest::OnMouseMove
 */
void SimpleFontTest::OnMouseMove(int x, int y) {
    this->cursor.SetPosition(static_cast<vislib::graphics::ImageSpaceType>(x), 
        static_cast<vislib::graphics::ImageSpaceType>(y), true);
}


/*
 * SimpleFontTest::OnSpecialKey
 */
void SimpleFontTest::OnSpecialKey(int key, int x, int y) {
    int modifiers = glutGetModifiers();

    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_SHIFT, (modifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT);
    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_CTRL, (modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL);
    this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_ALT, (modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT);
    this->cursor.SetPosition(static_cast<vislib::graphics::ImageSpaceType>(x), 
        static_cast<vislib::graphics::ImageSpaceType>(y), true);
}


/*
 * SimpleFontTest::Render
 */
void SimpleFontTest::Render(void) {

    glViewport(0, 0, this->GetWidth(), this->GetHeight());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    this->camera.glMultProjectionMatrix();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    this->camera.glMultViewMatrix();

    glScalef(0.7f, 0.7f, 0.7f);
    glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
    glRotatef(float(vislib::sys::GetTicksOfDay() % 3600) * 0.1f, 0.0f, 1.0f, 0.0f);

    glColor3ub(128, 128, 128);
    glBegin(GL_LINES);
        glVertex3i(-1, -1, -1);
        glVertex3i(-1, -1,  1);
        glVertex3i(-1,  1, -1);
        glVertex3i(-1,  1,  1);
        glVertex3i( 1, -1, -1);
        glVertex3i( 1, -1,  1);
        glVertex3i( 1,  1, -1);
        glVertex3i( 1,  1,  1);

        glVertex3i(-1, -1, -1);
        glVertex3i(-1,  1, -1);
        glVertex3i(-1, -1,  1);
        glVertex3i(-1,  1,  1);
        glVertex3i( 1, -1,  1);
        glVertex3i( 1,  1,  1);
        glVertex3i( 1, -1, -1);
        glVertex3i( 1,  1, -1);

        glVertex3i(-1, -1, -1);
        glVertex3i( 1, -1, -1);
        glVertex3i(-1, -1,  1);
        glVertex3i( 1, -1,  1);
        glVertex3i(-1,  1,  1);
        glVertex3i( 1,  1,  1);
        glVertex3i(-1,  1, -1);
        glVertex3i( 1,  1, -1);

        glVertex3i(0, -1, 0);
        glVertex3i(0,  1, 0);
    glEnd();

    glScalef(1.0f, -1.0f, 1.0f);
    glBegin(GL_LINE_LOOP);
        glColor3ub(255, 0, 0);
        glVertex2i(-1, -1);
        glColor3ub(64, 255, 64);
        glVertex2i( 1, -1);
        glVertex2i( 1,  1);
        glVertex2i(-1,  1);
    glEnd();

    if (this->font != NULL) {
        const char *text1 = "This is a demo text.";
        const char *text2 = 
            //"This is a much longer\ndemo text,\nto be used for testing the\n\nsoft wrapping\nof lines when drawing a string into an rectangluar area, and it also\ncontains a hard new line!";
            "Lorem ipsum dolor sit amet, consectetur adipisici elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquid ex ea commodi consequat. Quis aute iure reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint obcaecat cupiditat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

        float w, h;
        h = this->font->LineHeight();
        w = this->font->LineWidth(text1);

        this->font->SetSize(0.2f);

        glColor4ub(64, 128, 255, 192);

        glBegin(GL_LINE_LOOP);
            glVertex2f(0.0f, 0.0f);
            glVertex2f(w, 0.0f);
            glVertex2f(w, -h);
            glVertex2f(0.0f, -h);
        glEnd();

        this->font->DrawString(0.0f, 0.0f, true, text1
            //, vislib::graphics::AbstractFont::ALIGN_CENTER_MIDDLE
            //, vislib::graphics::AbstractFont::ALIGN_RIGHT_BOTTOM
            );

        glColor3ub(0, 255, 0);
        this->font->DrawString(-1.0f, -1.0f, 2.0f, 2.0f, 0.15f, true, text2
            , vislib::graphics::AbstractFont::ALIGN_CENTER_MIDDLE
            //, vislib::graphics::AbstractFont::ALIGN_RIGHT_BOTTOM
            );

        //unsigned int lc = this->font->BlockLines(2.0f, text2);
        //ASSERT(lc == 9);

        //glEnable(GL_BLEND);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        ////glBlendFunc(GL_SRC_COLOR, GL_ONE_MINUS_SRC_COLOR);
        ////glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        //glBindTexture(GL_TEXTURE_2D, this->font->BlockLines(0.0f, "a"));

        //glBegin(GL_QUADS);
        //    glTexCoord2f(0.0f, 0.0f);
        //    glVertex2f(-0.9f, -0.9f);
        //    glTexCoord2f(1.0f, 0.0f);
        //    glVertex2f( 0.9f, -0.9f);
        //    glTexCoord2f(1.0f, 1.0f);
        //    glVertex2f( 0.9f,  0.9f);
        //    glTexCoord2f(0.0f, 1.0f);
        //    glVertex2f(-0.9f,  0.9f);
        //glEnd();

        //glBindTexture(GL_TEXTURE_2D, 0);
        //glDisable(GL_BLEND);
    }

    glFlush();

    glutSwapBuffers();
    glutPostRedisplay();
}
