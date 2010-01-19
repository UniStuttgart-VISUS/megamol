/*
 * TimeView.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES 1
#include "TimeView.h"
#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */
#include <GL/gl.h>
#include <GL/glu.h>
#include <cmath>
#include "CoreInstance.h"
#include "vislib/mathfunctions.h"

using namespace megamol::core;


/*
 * special::TimeView::RenderView
 */
void special::TimeView::RenderView(double time) {
    GLint viewport[4];
    ::glGetIntegerv(GL_VIEWPORT, viewport);
    double w, h, e;

    if (viewport[2] < 1) { viewport[2] = 1; }
    if (viewport[3] < 1) { viewport[3] = 1; }

    if (viewport[2] > viewport[3]) {
        w = double(viewport[2]) / double(viewport[3]);
        h = 1.0;
    } else {
        w = 1.0;
        h = double(viewport[3]) / double(viewport[2]);
    }

    ::glClearColor(0.0f, 0.2f, 0.0f, 0.0f);
    ::glClear(GL_COLOR_BUFFER_BIT);

    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();

    ::glScaled(1.0 / w, 1.0 / h, 1.0);

    const double step = 0.2;
    double t, s, c;
    int i1, i2;

    ::glBegin(GL_LINES);

    ::glColor3f(0.0f, 0.4f, 0.0f);
    e = ceil(vislib::math::Max(w, h));
    for (t = -e; t <= e; t += step) {
        ::glVertex2d(-w, t);
        ::glVertex2d(w, t);
        ::glVertex2d(t, -h);
        ::glVertex2d(t, h);
    }

    ::glColor3f(0.0f, 0.6f, 0.0f);
    ::glVertex2d(0.8, 0.0);

    for (i1 = 0; i1 < 12; i1++) {
        for (i2 = 0; i2 < 5; i2++) {
            t = M_PI * static_cast<double>(i1 * 5 + i2) / 30.0;
            s = sin(t);
            c = cos(t);

            ::glVertex2d(0.8 * c, 0.8 * s);

            if (i2 > 0) {
                ::glColor3f(0.0f, 0.8f, 0.0f);
                ::glVertex2d(0.75 * c, 0.75 * s);
                ::glVertex2d(0.8 * c, 0.8 * s);

            } else {
                ::glColor3f(0.0f, 1.0f, 0.0f);
                ::glVertex2d(0.7 * c, 0.7 * s);
                ::glVertex2d(0.85 * c, 0.85 * s);

            }

            ::glColor3f(0.0f, 0.6f, 0.0f);
            ::glVertex2d(0.8 * c, 0.8 * s);
        }
    }

    ::glColor3f(0.0f, 0.6f, 0.0f);
    ::glVertex2d(0.8, 0.0);

    t = time * 0.001 * M_PI;
    ::glColor3f(0.1f, 1.0f, 0.1f);
    ::glVertex2d(0.0, 0.0);
    ::glColor3f(0.9f, 1.0f, 0.9f);
    e *= 2.0;
    ::glVertex2d(e * sin(t), e * cos(t));

    t = 0.1 * time;
    t = t - static_cast<double>(static_cast<int>(t));
    // 0 <= t <= 1
    t = (t > 0.5) ? (1.0 - (t - 0.5) * 4.0) : (t * 4.0 - 1.0);

    ::glLoadIdentity();

    ::glColor3f(0.1f, 1.0f, 0.1f);
    ::glVertex2d(-w, t);
    ::glVertex2d(w, t);
    ::glVertex2d(t, -h);
    ::glVertex2d(t, h);

    ::glEnd();
}


/*
 * special::TimeView::TimeView
 */
special::TimeView::TimeView(void) : view::AbstractView(), Module(),
        width(1), height(1) {
    this->MakeSlotAvailable(view::AbstractView::getRenderViewSlot());
}


/*
 * special::TimeView::~TimeView
 */
special::TimeView::~TimeView(void) {
    this->Release();
}


/*
 * special::TimeView::Render
 */
void special::TimeView::Render(void) {
    ::glViewport(0, 0, this->width, this->height);
    RenderView(this->GetCoreInstance()->GetInstanceTime());
}


/*
 * special::TimeView::UpdateFreeze
 */
void special::TimeView::UpdateFreeze(bool freeze) {
    // TODO: Implement something useful here
}


/*
 * special::TimeView::create
 */
bool special::TimeView::create(void) {
    return true;
}


/*
 * special::TimeView::release
 */
void special::TimeView::release(void) {
}


/*
 * special::TimeView::onRenderView
 */
bool special::TimeView::onRenderView(Call& call) {
    RenderView(this->GetCoreInstance()->GetInstanceTime());
    return true;
}
