//
// View3DMouse.cpp
//
// Copyright (C) 2012 by University of Stuttgart (VISUS).
// All rights reserved.
//

#include "stdafx.h"
#define _USE_MATH_DEFINES

#include "View3DMouse.h"
#include "mmcore/CallAutoDescription.h"
#include "CallMouseInput.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/view/MouseFlags.h"

using namespace megamol;


protein::View3DMouse::View3DMouse(void) : core::view::View3D(),
         mouseSlot("mouse", "Slot to send mouse information to the renderer."),
         enableSelectingSlot("enableSelecting", "Enable selecting and picking with the mouse."),
         toggleSelect(false) {

    // Slot for mouse input call
    this->mouseSlot.SetCompatibleCall<core::CallAutoDescription<CallMouseInput> >();
    this->MakeSlotAvailable(&this->mouseSlot);

    // Slot for key modifier
    this->enableSelectingSlot << new core::param::ButtonParam(vislib::sys::KeyCode::KEY_TAB);
    this->enableSelectingSlot.SetUpdateCallback(&View3DMouse::OnButtonEvent);
    this->MakeSlotAvailable(&this->enableSelectingSlot);
}


void protein::View3DMouse::SetCursor2DButtonState(unsigned int btn, bool down) {

    //if(!this->toggleSelect) {
        View3D::SetCursor2DButtonState(btn, down); // Keep camera movement functional
    //}
    //else {
        switch (btn) {
        case 0 : // left
            megamol::core::view::MouseFlagsSetFlag(this->mouseFlags,
                    core::view::MOUSEFLAG_BUTTON_LEFT_DOWN, down);
            break;
        case 1 : // right
            megamol::core::view::MouseFlagsSetFlag(this->mouseFlags,
                    core::view::MOUSEFLAG_BUTTON_RIGHT_DOWN, down);
            break;
        case 2 : // middle
            megamol::core::view::MouseFlagsSetFlag(this->mouseFlags,
                    core::view::MOUSEFLAG_BUTTON_MIDDLE_DOWN, down);
            break;
        }
        //}
}


void protein::View3DMouse::SetCursor2DPosition(float x, float y) {

    if(!this->toggleSelect) {
        View3D::SetCursor2DPosition(x, y); // Keep camera movement functional
        core::view::MouseFlagsResetAllChanged(this->mouseFlags);
    }
    else {
        CallMouseInput *cm = this->mouseSlot.CallAs<CallMouseInput>();
        if (cm) {
            cm->SetMouseInfo(static_cast<int>(x), static_cast<int>(y), this->mouseFlags);
            if ((*cm)(0)) {
                this->mouseX = static_cast<int>(x);
                this->mouseY = static_cast<int>(y);
                megamol::core::view::MouseFlagsResetAllChanged(this->mouseFlags);
                // mouse event consumed
                return;
            }
            megamol::core::view::MouseFlagsResetAllChanged(this->mouseFlags);
        }
    }
}


protein::View3DMouse::~View3DMouse() {
}


bool protein::View3DMouse::OnButtonEvent(core::param::ParamSlot& p) {
    printf("TAB pressed\n"); // DEBUG
    this->toggleSelect = !this->toggleSelect;
    return true;
}


