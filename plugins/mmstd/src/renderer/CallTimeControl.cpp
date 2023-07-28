/**
 * MegaMol
 * Copyright (c) 2012, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd/renderer/CallTimeControl.h"

using namespace megamol::core;


/*
 * view::CallTimeControl::CallTimeControl
 */
view::CallTimeControl::CallTimeControl() : Call(), m(NULL) {
    // intentionally empty
}


/*
 * view::CallTimeControl::~CallTimeControl
 */
view::CallTimeControl::~CallTimeControl() {
    this->m = NULL; // do not delete
}
