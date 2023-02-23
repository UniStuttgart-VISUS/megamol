/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#define _USE_MATH_DEFINES
#include "mmstd_gl/special/DemoRenderer2D.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"
#include <cmath>

using namespace megamol::mmstd_gl;


/*
 * view::special::DemoRenderer2D::DemoRenderer2D
 */
special::DemoRenderer2D::DemoRenderer2D()
        : Renderer2DModuleGL()
        , mx(0.0f)
        , my(0.0f)
        , fromx(1.0f)
        , fromy(1.0f)
        , tox(3.0f)
        , toy(3.0f)
        , drag(false) {
    // intentionally empty
}


/*
 * view::special::DemoRenderer2D::~DemoRenderer2D
 */
special::DemoRenderer2D::~DemoRenderer2D() {
    this->Release();
}


/*
 * view::special::DemoRenderer2D::create
 */
bool special::DemoRenderer2D::create() {
    // intentionally empty
    return true;
}


/*
 * view::special::DemoRenderer2D::GetExtents
 */
bool special::DemoRenderer2D::GetExtents(CallRender2DGL& call) {

    call.AccessBoundingBoxes().SetBoundingBox(0.0f, 0.0f, 0, 4.0f, 4.0f, 0);

    return true;
}


/*
 * view::special::DemoRenderer2D::Render
 */
bool special::DemoRenderer2D::Render(CallRender2DGL& call) {

    auto bbox = call.GetBoundingBoxes().BoundingBox();

    ::glColor3ub(255, 0, 0);
    ::glBegin(GL_LINE_LOOP);
    ::glVertex2f(bbox.Left(), bbox.Bottom());
    ::glVertex2f(bbox.Right(), bbox.Bottom());
    ::glVertex2f(bbox.Right(), bbox.Top());
    ::glVertex2f(bbox.Left(), bbox.Top());
    ::glEnd();
    const float border = 0.1f;
    ::glColor3ub(255, 255, 0);
    ::glBegin(GL_LINE_LOOP);
    ::glVertex2f(bbox.Left() + border, bbox.Bottom() + border);
    ::glVertex2f(bbox.Right() - border, bbox.Bottom() + border);
    ::glVertex2f(bbox.Right() - border, bbox.Top() - border);
    ::glVertex2f(bbox.Left() + border, bbox.Top() - border);
    ::glEnd();

    ::glColor3ub(255, 255, 255);
    ::glBegin(GL_LINE_LOOP);
    for (float a = 0.0f; a < 2.0f * static_cast<float>(M_PI); a += 0.01f) {
        ::glVertex2f(2.0f * (1.0f + sinf(a)), 2.0f * (1.0f + cosf(a)));
    }
    ::glEnd();

    ::glColor3ub(0, 0, 255);
    ::glBegin(GL_LINES);
    ::glVertex2f(this->mx, 0.0f);
    ::glVertex2f(this->mx, 4.0f);
    ::glVertex2f(0.0f, this->my);
    ::glVertex2f(4.0f, this->my);
    ::glEnd();

    ::glColor3ub(0, 255, 0);
    ::glBegin(GL_LINES);
    ::glVertex2f(this->fromx, this->fromy);
    ::glVertex2f(this->tox, this->toy);
    ::glEnd();

    return true;
}


/*
 * view::special::DemoRenderer2D::release
 */
void special::DemoRenderer2D::release() {
    // intentionally empty
}


/*
 * view::special::DemoRenderer2D::MouseEvent
 */
bool special::DemoRenderer2D::MouseEvent(float x, float y, core::view::MouseFlags flags) {
    if (x < 0.0f)
        this->mx = 0.0f;
    else if (x > 4.0f)
        this->mx = 4.0f;
    else
        mx = x;
    if (y < 0.0f)
        this->my = 0.0f;
    else if (y > 4.0f)
        this->my = 4.0f;
    else
        my = y;

    if ((flags & core::view::MOUSEFLAG_BUTTON_LEFT_DOWN) != 0) {
        if ((flags & core::view::MOUSEFLAG_BUTTON_LEFT_CHANGED) != 0) {
            // pressed!

            float dx = mx - 2.0f;
            float dy = my - 2.0f;

            this->drag = ((dx * dx + dy * dy) < 4.0f);

            if (this->drag) {
                this->fromx = this->tox = mx;
                this->fromy = this->toy = my;
                return true; // consume event
            }
        }
        if (this->drag) {
            this->tox = mx;
            this->toy = my;
            return true;
        }
    } else {
        this->drag = false;
    }

    return false;
}
