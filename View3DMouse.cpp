//
// View3DMouse.cpp
//
// Copyright (C) 2012 by University of Stuttgart (VISUS).
// All rights reserved.
//

#include "stdafx.h"
#define _USE_MATH_DEFINES

#include "View3DMouse.h"
#include "CallAutoDescription.h"
#include "CallMouseInput.h"

using namespace megamol;


protein::View3DMouse::View3DMouse(void) : core::view::View3D(),
		 mouseSlot("mouse", "Slot to send mouse information to the renderer.") {

    this->mouseSlot.SetCompatibleCall<core::CallAutoDescription<CallMouseInput> >();
    this->MakeSlotAvailable(&this->mouseSlot);
}


void protein::View3DMouse::SetCursor2DButtonState(unsigned int btn, bool down) {

	View3D::SetCursor2DButtonState(btn, down); // Keep camera movement functional

    switch (btn) {
        case 0 : // left
            core::view::MouseFlagsSetFlag(this->mouseFlags,
            		core::view::MOUSEFLAG_BUTTON_LEFT_DOWN, down);
            break;
        case 1 : // right
        	core::view::MouseFlagsSetFlag(this->mouseFlags,
        			core::view::MOUSEFLAG_BUTTON_RIGHT_DOWN, down);
            break;
        case 2 : // middle
        	core::view::MouseFlagsSetFlag(this->mouseFlags,
        			core::view::MOUSEFLAG_BUTTON_MIDDLE_DOWN, down);
            break;
    }
}


void protein::View3DMouse::SetCursor2DPosition(float x, float y) {

	View3D::SetCursor2DPosition(x, y); // Keep camera movement functional

	CallMouseInput *cm = this->mouseSlot.CallAs<CallMouseInput>();
	if (cm) {
		cm->SetMouseInfo(static_cast<int>(x), static_cast<int>(y), this->mouseFlags);
		if ((*cm)(0)) {
			this->mouseX = static_cast<int>(x);
			this->mouseY = static_cast<int>(y);
			core::view::MouseFlagsResetAllChanged(this->mouseFlags);
			// mouse event consumed
			return;
		}
		core::view::MouseFlagsResetAllChanged(this->mouseFlags);
	}
}


protein::View3DMouse::~View3DMouse() {
}



