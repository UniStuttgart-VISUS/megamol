//
// Renderer3DModuleMouse.cpp
//
// Copyright (C) 2012 by University of Stuttgart (VISUS).
// All rights reserved.
//

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "Renderer3DModuleMouse.h"
#include "CallMouseInput.h"

using namespace megamol;


protein_cuda::Renderer3DModuleMouse::Renderer3DModuleMouse(void)
		: core::view::Renderer3DModuleDS(),
        mouseSlot("mouse", "Enables the view to send mouse information to the renderer.") {

	// Setup slot for render callback
    this->mouseSlot.SetCallback(
    		CallMouseInput::ClassName(),
    		CallMouseInput::FunctionName(0),
    		&Renderer3DModuleMouse::MouseEventCallback);

    // Make render slot available
    this->MakeSlotAvailable(&this->mouseSlot);
}


protein_cuda::Renderer3DModuleMouse::~Renderer3DModuleMouse(void) {

}
