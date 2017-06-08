/*
 * CallGetTransferFunction.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/CallGetTransferFunction.h"

using namespace megamol::core;


/*
 * view::CallGetTransferFunction::CallGetTransferFunction
 */
view::CallGetTransferFunction::CallGetTransferFunction(void) : Call(),
        texID(0), texSize(1), texData(NULL) {
    // intentionally empty
}


/*
 * view::CallGetTransferFunction::~CallGetTransferFunction
 */
view::CallGetTransferFunction::~CallGetTransferFunction(void) {
    // intentionally empty
}
