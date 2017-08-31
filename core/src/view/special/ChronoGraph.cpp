/*
 * ChronoGraph.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/view/special/ChronoGraph.h"
#include "mmcore/CoreInstance.h"
#include "vislib/math/mathfunctions.h"
#include <cmath>

using namespace megamol::core;


/*
 * view::special::ChronoGraph::ChronoGraph
 */
view::special::ChronoGraph::ChronoGraph() : view::Renderer2DModule() {
    // intentionally empty
}


/*
 * view::special::ChronoGraph::~ChronoGraph
 */
view::special::ChronoGraph::~ChronoGraph() {
    this->Release();
}


/*
 * view::special::ChronoGraph::create
 */
bool view::special::ChronoGraph::create(void) {
    // intentionally empty
    return true;
}


/*
 * view::special::ChronoGraph::GetExtents
 */
bool view::special::ChronoGraph::GetExtents(view::CallRender2D& call) {
    call.SetBoundingBox(-1.0f, -1.0f, 1.0f, 1.0f);
    return true;
}


/*
 * view::special::ChronoGraph::Render
 */
bool view::special::ChronoGraph::Render(view::CallRender2D& call) {
    ::glEnable(GL_LINE_SMOOTH);
    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    float time = static_cast<float>(call.InstanceTime());

    this->renderInfoGrid(time,
        call.GetBoundingBox().Left(),
        call.GetBoundingBox().Bottom(),
        call.GetBoundingBox().Width(),
        call.GetBoundingBox().Height());

    this->renderInfoCircle(time,
        call.GetBoundingBox().Left(),
        call.GetBoundingBox().Bottom(),
        call.GetBoundingBox().Width(),
        call.GetBoundingBox().Height());

    this->renderInfoCircle(time, -1.0f, -1.0f, 2.0f, 2.0f);

    ::glDisable(GL_LINE_SMOOTH);
    ::glDisable(GL_BLEND);

    return true;
}


/*
 * view::special::ChronoGraph::release
 */
void view::special::ChronoGraph::release(void) {
    // intentionally empty
}


/*
 * view::special::ChronoGraph::renderInfoGrid
 */
void view::special::ChronoGraph::renderInfoGrid(float time, float x, float y, float w, float h) {
    const int steps = 10;
    float a;

    ::glEnable(GL_LINE_SMOOTH);
    ::glLineWidth(1.2f);
    ::glColor4ub(255, 255, 255, 64);

    ::glBegin(GL_LINES);

    for (int i = 0; i <= steps; i++){
        a = static_cast<float>(i) / static_cast<float>(steps);
        ::glVertex2f(x, y + h * a);
        ::glVertex2f(x + w, y + h * a);
        ::glVertex2f(x + w * a, y);
        ::glVertex2f(x + w * a, y + h);
    }

    ::glEnd();

    ::glDisable(GL_LINE_SMOOTH);
    ::glLineWidth(1.0f);

    ::glBegin(GL_LINES);

    a = ::fabsf(::fmodf(time, 20.0f) * 0.1f - 1.0f);
    ::glColor4ub(255, 255, 255, 191);
    ::glVertex2f(x, y + h * a);
    ::glVertex2f(x + w, y + h * a);
    ::glVertex2f(x + w * a, y);
    ::glVertex2f(x + w * a, y + h);

    ::glEnd();
}


/*
 * view::special::ChronoGraph::renderInfoCircle
 */
void view::special::ChronoGraph::renderInfoCircle(float time, float x, float y, float w, float h) {
    const int steps = 100;
    float a, cx, cy, px, py;
    float rad = 0.5f * vislib::math::Min(::fabs(w), ::fabs(h));

    ::glEnable(GL_LINE_SMOOTH);
    ::glLineWidth(1.2f);
    ::glColor4ub(255, 255, 255, 64);

    ::glBegin(GL_LINE_LOOP);

    for (int i = 0; i < steps; i++){
        a = 2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(steps);
        ::glVertex2f(x + w * 0.5f + rad * ::cosf(a), y + h * 0.5f + rad * ::sinf(a));
    }

    ::glEnd();

    ::glBegin(GL_LINES);

    //rad = 0.5f * ::sqrtf(w * w + h * h);
    ::glColor4ub(255, 255, 255, 191);
    a = -2.0f * static_cast<float>(M_PI) * 0.01f * ::fmodf(time, 100.0f);
    w *= 0.5f;
    h *= 0.5f;
    cx = x + w;
    cy = y + h;
    ::glVertex2f(cx, cy);
    px = rad * ::cosf(a);
    py = rad * ::sinf(a);
    if (std::abs(px) > w) {
        rad = std::abs(w / ::cosf(a));
        px = rad * ::cosf(a);
        py = rad * ::sinf(a);
    }
    if (std::abs(py) > h) {
        rad = std::abs(h / ::sinf(a));
        px = rad * ::cosf(a);
        py = rad * ::sinf(a);
    }
    ::glVertex2f(cx + px, cy + py);

    ::glEnd();

    ::glDisable(GL_LINE_SMOOTH);
}
