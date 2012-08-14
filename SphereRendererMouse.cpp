//
// SphereRendererMouse.cpp
//
// Copyright (C) 2012 by University of Stuttgart (VISUS).
// All rights reserved.
//


#include "stdafx.h"
#define _USE_MATH_DEFINES
#include <SphereRendererMouse.h>
#include <GL/gl.h>
#include <cmath>

using namespace megamol;

/*
 * protein::special::SphereRendererMouse::SphereRendererMouse
 */
protein::SphereRendererMouse::SphereRendererMouse() : Renderer3DModuleMouse(),
        mouseX(0.0f), mouseY(0.0f) {
    // intentionally empty
}


/*
 * protein::SphereRendererMouse::~SphereRendererMouse
 */
protein::SphereRendererMouse::~SphereRendererMouse() {
    this->Release();
}


/*
 * protein::SphereRendererMouse::create
 */
bool protein::SphereRendererMouse::create(void) {
    // intentionally empty
    return true;
}


/*
 * protein::SphereRendererMouse::GetCapabilities
 */
bool protein::SphereRendererMouse::GetCapabilities(core::Call& call) {
	return true;
}


/*
 * protein::SphereRendererMouse::GetExtents
 */
bool protein::SphereRendererMouse::GetExtents(core::Call& call) {

	// TODO set generic bounding box

    return true;
}


/*
 * protein::SphereRendererMouse::Render
 */
bool protein::SphereRendererMouse::Render(core::Call& call) {

    return true;
}


/*
 * protein::SphereRendererMouse::release
 */
void protein::SphereRendererMouse::release(void) {
    // intentionally empty
}


/*
 * protein::SphereRendererMouse::MouseEvent
 */
bool protein::SphereRendererMouse::MouseEvent(int x, int y, core::view::MouseFlags flags) {

    this->mouseX = x;
    this->mouseY = y;

    printf("Pos (%i %i)\n", this->mouseX, this->mouseY);

    if ((flags & core::view::MOUSEFLAG_BUTTON_LEFT_DOWN) != 0) {
    	printf("Left Button DOWN\n");
    }

    /*if ((flags & core::view::MOUSEFLAG_BUTTON_RIGHT_DOWN) != 0) {
    	printf("Right Button DOWN\n");
    }

    if ((flags & core::view::MOUSEFLAG_BUTTON_MIDDLE_DOWN) != 0) {
    	printf("Middle Button DOWN\n");
    }*/

    return true;
}
