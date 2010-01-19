/*
 * DemoRenderer2D.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "DemoRenderer2D.h"
#include <GL/gl.h>
#include <cmath>

using namespace megamol::core;

misc::DemoRenderer2D::DemoRenderer2D() : view::Renderer2DModule() {
}

misc::DemoRenderer2D::~DemoRenderer2D() {
    this->Release();
}

bool misc::DemoRenderer2D::create(void) {
    return true;
}

bool misc::DemoRenderer2D::GetExtents(view::CallRender2D& call) {

    call.SetBoundingBox(0.0f, 0.0f, 4.0f, 4.0f);

    return true;
}

bool misc::DemoRenderer2D::Render(view::CallRender2D& call) {

	::glColor3ub(255, 0, 0);
	::glBegin(GL_LINE_LOOP);
	::glVertex2f(call.GetBoundingBox().Left(), call.GetBoundingBox().Bottom());
	::glVertex2f(call.GetBoundingBox().Right(), call.GetBoundingBox().Bottom());
	::glVertex2f(call.GetBoundingBox().Right(), call.GetBoundingBox().Top());
	::glVertex2f(call.GetBoundingBox().Left(), call.GetBoundingBox().Top());
	::glEnd();
	const float border = 0.1f;
	::glColor3ub(255, 255, 0);
	::glBegin(GL_LINE_LOOP);
	::glVertex2f(call.GetBoundingBox().Left() + border, call.GetBoundingBox().Bottom() + border);
	::glVertex2f(call.GetBoundingBox().Right() - border, call.GetBoundingBox().Bottom() + border);
	::glVertex2f(call.GetBoundingBox().Right() - border, call.GetBoundingBox().Top() - border);
	::glVertex2f(call.GetBoundingBox().Left() + border, call.GetBoundingBox().Top() - border);
	::glEnd();

    ::glColor3ub(255, 255, 255);
    ::glBegin(GL_LINE_LOOP);
    for (float a = 0.0f; a < 2.0f * static_cast<float>(M_PI); a += 0.01f) {
        ::glVertex2f(2.0f * (1.0f + sinf(a)), 2.0f * (1.0f + cosf(a)));
    }
    ::glEnd();

    return true;
}

void misc::DemoRenderer2D::release(void) {
}
