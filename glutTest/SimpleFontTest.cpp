/*
 * SimpleFontTest.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */
#include "glh/glh_extensions.h"
#include "SimpleFontTest.h"

#include "vislibGlutInclude.h"
#include <cstdio>

#include "vislib/types.h"
#include "vislib/graphicstypes.h"
#include "vislib/Rectangle.h"
#include "vislib/sysfunctions.h"
#include "vislib/SimpleFont.h"

#include "vislib/OutlineFont.h"
#include "vislib/Verdana.inc"

#define _USE_MATH_DEFINES
#include <math.h>
#include <cstdlib>
#include <cfloat>


/****************************************************************************/

/*
 * SimpleFontTest::BoxTest<T>::Draw
 */
template<class T> void SimpleFontTest::BoxTest<T>::Draw(void) const {
    if (!this->active || (this->font == NULL)) return;
    glTranslatef(0.0f, 0.0f, this->z);
    glColor3ubv(this->col);
    glBegin(GL_LINE_LOOP);
    glVertex2f(this->x, this->y);
    glVertex2f(this->x + this->w, this->y);
    glVertex2f(this->x + this->w, this->y + this->h);
    glVertex2f(this->x, this->y + this->h);
    glEnd();
    this->font->DrawString(this->x, this->y, this->w, this->h, this->size, this->flipY, this->txt, this->align);
    glTranslatef(0.0f, 0.0f, -this->z);
}


/*
 * SimpleFontTest::LineTest<T>::Draw
 */
template<class T> void SimpleFontTest::LineTest<T>::Draw(void) const {
    using vislib::graphics::AbstractFont;
    float w, h;
    if (!this->active || (this->font == NULL)) return;
    h = this->font->LineHeight(this->size) * this->font->BlockLines(FLT_MAX, this->txt);
    if (this->flipY) {
        h = -h;
    }
    w = this->font->LineWidth(this->size, this->txt);
    float xo = 0.0f, yo = 0.0f;
    glColor3ubv(this->col);
    if ((this->align == AbstractFont::ALIGN_CENTER_BOTTOM)
            || (this->align == AbstractFont::ALIGN_CENTER_MIDDLE)
            || (this->align == AbstractFont::ALIGN_CENTER_TOP)) {
        xo -= w * 0.5f;
    } else if ((this->align == AbstractFont::ALIGN_RIGHT_BOTTOM)
            || (this->align == AbstractFont::ALIGN_RIGHT_MIDDLE)
            || (this->align == AbstractFont::ALIGN_RIGHT_TOP)) {
        xo -= w;
    }
    if ((this->align == AbstractFont::ALIGN_CENTER_MIDDLE)
            || (this->align == AbstractFont::ALIGN_LEFT_MIDDLE)
            || (this->align == AbstractFont::ALIGN_RIGHT_MIDDLE)) {
        yo -= h * 0.5f;
    } else if ((this->align == AbstractFont::ALIGN_CENTER_BOTTOM)
            || (this->align == AbstractFont::ALIGN_LEFT_BOTTOM)
            || (this->align == AbstractFont::ALIGN_RIGHT_BOTTOM)) {
        yo -= h;
    }
    xo += this->x;
    yo += this->y;
    glTranslatef(xo, yo, this->z);
    glBegin(GL_LINE_LOOP);
    glVertex2f(0.0f, 0.0f);
    glVertex2f(w, 0.0f);
    glVertex2f(w, h);
    glVertex2f(0.0f, h);
    glEnd();
    glTranslatef(-xo, -yo, 0.0f);
    this->font->DrawString(this->x, this->y, this->size, this->flipY, this->txt, this->align);
    glTranslatef(0.0f, 0.0f, -this->z);
}


/****************************************************************************/

/*
 * SimpleFontTest::SimpleFontTest
 */
SimpleFontTest::SimpleFontTest(void) : AbstractGlutApp(), tests(), font1(NULL), rot(true) {

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
    using vislib::graphics::AbstractFont;

    this->camera.Parameters()->SetView(
        vislib::math::Point<double, 3>(0.0, -2.5, 0.0),
        vislib::math::Point<double, 3>(0.0, 0.0, 0.0),
        vislib::math::Vector<double, 3>(0.0, 0.0, 1.0));

    if (this->font1 == NULL) {
        this->font1 = new vislib::graphics::gl::OutlineFont(
            vislib::graphics::gl::FontInfo_Verdana,
            vislib::graphics::gl::OutlineFont::RENDERTYPE_FILL_AND_OUTLINE);
    }
    if (!this->font1->Initialise()) return -2;

    vislib::StringA lorem("Lorem ipsum dolor sit amet, consectetur adipisici elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquid ex ea commodi consequat. Quis aute iure reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint obcaecat cupiditat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.");
    float z = 1.0f, s = 0.1f;
    unsigned int r = 64, g = 128, b = 255;
    this->tests.Add(new LineTest<char>(-1.0f,  1.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_LEFT_TOP, "TopLeft"));
    this->tests.Add(new LineTest<char>( 0.0f,  1.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_CENTER_TOP, "TopCenter"));
    this->tests.Add(new LineTest<char>( 1.0f,  1.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_RIGHT_TOP, "TopRight"));
    this->tests.Add(new LineTest<char>(-1.0f,  0.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_LEFT_MIDDLE, "MiddleLeft"));
#ifdef _WIN32
    this->tests.Add(new LineTest<wchar_t>( 0.0f,  0.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_CENTER_MIDDLE, L"MiddleCänter™"));
#else /* _WIN32 */
    this->tests.Add(new LineTest<wchar_t>( 0.0f,  0.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_CENTER_MIDDLE, L"MiddleCenter"));
#endif /* _WIN32 */
    this->tests.Add(new LineTest<char>( 1.0f,  0.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_RIGHT_MIDDLE, "MiddleRight"));
    this->tests.Add(new LineTest<char>(-1.0f, -1.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_LEFT_BOTTOM, "BottomLeft"));
    this->tests.Add(new LineTest<char>( 0.0f, -1.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_CENTER_BOTTOM, "BottomCenter"));
    this->tests.Add(new LineTest<char>( 1.0f, -1.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_RIGHT_BOTTOM, "BottomRight"));
    z -= 0.2f; r = 128; g = 192; b = 255; s = 0.2f;
    this->tests.Add(new LineTest<char>(-1.0f,  1.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_LEFT_TOP, "Top\nLeft"));
    this->tests.Add(new LineTest<char>( 0.0f,  1.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_CENTER_TOP, "Top\nCenter"));
    this->tests.Add(new LineTest<char>( 1.0f,  1.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_RIGHT_TOP, "Top\nRight"));
    this->tests.Add(new LineTest<char>(-1.0f,  0.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_LEFT_MIDDLE, "Middle\nLeft"));
    this->tests.Add(new LineTest<char>( 0.0f,  0.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_CENTER_MIDDLE, "Middle\nCenter"));
    this->tests.Add(new LineTest<char>( 1.0f,  0.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_RIGHT_MIDDLE, "Middle\nRight"));
    this->tests.Add(new LineTest<char>(-1.0f, -1.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_LEFT_BOTTOM, "Bottom\nLeft"));
    this->tests.Add(new LineTest<char>( 0.0f, -1.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_CENTER_BOTTOM, "Bottom\nCenter"));
    this->tests.Add(new LineTest<char>( 1.0f, -1.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_RIGHT_BOTTOM, "Bottom\nRight"));
    z -= 0.2f; r = 0; g = 255; b = 0; s = 0.15f;
    this->tests.Add(new BoxTest<char>(-1.0f, -1.0f, 2.0f, 2.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_LEFT_TOP, vislib::StringA("TopLeft-Text: ") + lorem));
    z -= 0.2f; r = 64; g = 255; b = 0; s = 0.15f;
    this->tests.Add(new BoxTest<char>(-1.0f, -1.0f, 2.0f, 2.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_CENTER_TOP, vislib::StringA("TopCenter-Text: ") + lorem));
    z -= 0.2f; r = 128; g = 255; b = 0; s = 0.15f;
    this->tests.Add(new BoxTest<char>(-1.0f, -1.0f, 2.0f, 2.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_RIGHT_TOP, vislib::StringA("TopRight-Text: ") + lorem));
    z -= 0.2f; r = 192; g = 255; b = 0; s = 0.15f;
    this->tests.Add(new BoxTest<char>(-1.0f, -1.0f, 2.0f, 2.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_LEFT_MIDDLE, vislib::StringA("MiddleLeft-Text: ") + lorem));
    z -= 0.2f; r = 255; g = 255; b = 0; s = 0.15f;
    this->tests.Add(new BoxTest<char>(-1.0f, -1.0f, 2.0f, 2.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_CENTER_MIDDLE, vislib::StringA("MiddleCenter-Text: ") + lorem));
    z -= 0.2f; r = 255; g = 192; b = 0; s = 0.15f;
    this->tests.Add(new BoxTest<char>(-1.0f, -1.0f, 2.0f, 2.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_RIGHT_MIDDLE, vislib::StringA("MiddleRight-Text: ") + lorem));
    z -= 0.2f; r = 255; g = 128; b = 0; s = 0.15f;
    this->tests.Add(new BoxTest<char>(-1.0f, -1.0f, 2.0f, 2.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_LEFT_BOTTOM, vislib::StringA("BottomLeft-Text: ") + lorem));
    z -= 0.2f; r = 255; g = 64; b = 0; s = 0.15f;
    this->tests.Add(new BoxTest<char>(-1.0f, -1.0f, 2.0f, 2.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_CENTER_BOTTOM, vislib::StringA("BottomCenter-Text: ") + lorem));
    z -= 0.2f; r = 255; g = 0; b = 0; s = 0.15f;
    this->tests.Add(new BoxTest<char>(-1.0f, -1.0f, 2.0f, 2.0f, z, r, g, b, this->font1, s, true, AbstractFont::ALIGN_RIGHT_BOTTOM, vislib::StringA("BottomRight-Text: ") + lorem));

    this->tests.Sort(AbstractTest::Compare);

    ::glEnable(GL_TEXTURE_2D);
    ::glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    return 0;
}


/*
 * SimpleFontTest::GLDeinit
 */
void SimpleFontTest::GLDeinit(void) {
    if (this->font1 != NULL) {
        this->font1->Deinitialise();
        delete this->font1;
        this->font1 = NULL;
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

    switch (key) {
        case 'f': {
            vislib::graphics::AbstractFont *f = NULL;
            if (dynamic_cast<vislib::graphics::gl::SimpleFont*>(this->font1) != NULL) {
                f = new vislib::graphics::gl::OutlineFont(
                    vislib::graphics::gl::FontInfo_Verdana,
                    vislib::graphics::gl::OutlineFont::RENDERTYPE_FILL_AND_OUTLINE);
            } else {
                f = new vislib::graphics::gl::SimpleFont();
            }
            if (f != NULL) {
                if (f->Initialise()) {
                    delete this->font1;
                    this->font1 = f;
                    printf("Font replaced\n");
                } else {
                    printf("Failed to initialize new font\n");
                }
            } else {
                printf("Failed to create new font\n");
            }
            return true;
        }
        case '0':
            for (SIZE_T i = 0; i < this->tests.Count(); i++) {
                this->tests[i]->active = true;
            }
            return true;
        case '1':
            for (SIZE_T i = 0; i < this->tests.Count(); i++) {
                this->tests[i]->active = (i == 0);
            }
            return true;
        case '+':
            for (SIZE_T i = 0; i < this->tests.Count(); i++) {
                if (this->tests[i]->active) {
                    this->tests[i]->active = false;
                    this->tests[(i + 1) % this->tests.Count()]->active = true;
                    break;
                }
            }
            return true;
        case '-':
            for (SIZE_T i = 0; i < this->tests.Count(); i++) {
                if (this->tests[i]->active) {
                    this->tests[i]->active = false;
                    this->tests[(i == 0) ? (this->tests.Count() - 1) : (i - 1)]->active = true;
                    break;
                }
            }
            return true;
        case 'r':
            rot = !rot;
            return true;
    }

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

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(1.1f);

    // find camera position in rotated user space for correct painters algorithm
    double proj[16];
    double view[16];
    GLint win[4];
    ::glGetDoublev(GL_PROJECTION_MATRIX, proj);
    ::glGetDoublev(GL_MODELVIEW_MATRIX, view);
    ::glGetIntegerv(GL_VIEWPORT, win);
    vislib::math::Point<float, 3> camP(this->camera.Parameters()->EyePosition());
    double wx, wy, wz;
    ::gluProject(camP.X(), camP.Y(), camP.Z(), view, proj, win, &wx, &wy, &wz);

    glScalef(0.6f, 0.6f, 0.6f);
    glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
    if (rot) {
        glRotatef(float(vislib::sys::GetTicksOfDay() % 36000) * 0.01f, 0.0f, 1.0f, 0.0f);
    }

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
    glEnd();

    ::glGetDoublev(GL_MODELVIEW_MATRIX, view);
    double x, y, z;
    ::gluUnProject(wx, wy, wz, view, proj, win, &x, &y, &z);

    float camZ = static_cast<float>(z);

    glScalef(1.0f, -1.0f, 1.0f);
    SIZE_T l = this->tests.Count();
    for (SIZE_T i = 0; i < l; i++) {
        if (this->tests[i]->Z() >= camZ) {
            this->tests[i]->Draw();
        }
    }
    for (SIZE_T i = l; i > 0;) {
        i--;
        if (this->tests[i]->Z() < camZ) {
            this->tests[i]->Draw();
        }
    }

    glFlush();

    glutSwapBuffers();
    glutPostRedisplay();
}
