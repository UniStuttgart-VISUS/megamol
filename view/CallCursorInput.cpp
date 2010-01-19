/*
 * CallCursorInput.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallCursorInput.h"

using namespace megamol::core;


/*
 * view::CallCursorInput::CallCursorInput
 */
view::CallCursorInput::CallCursorInput(void) : Call(), btn(0), down(false),
        x(0.0f), y(0.0f), mod(MMC_INMOD_SHIFT) {
    // intentionally empty
}


/*
 * view::CallCursorInput::~CallCursorInput
 */
view::CallCursorInput::~CallCursorInput(void) {
    // intentionally empty
}
