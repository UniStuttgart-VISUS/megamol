/*
 * CallRegisterAtController.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallRegisterAtController.h"

using namespace megamol::core;


/*
 * special::CallRegisterAtController::CallRegisterAtController
 */
special::CallRegisterAtController::CallRegisterAtController(void) : Call(),
        client(NULL) {
    // intentionally empty
}


/*
 * special::CallRegisterAtController::~CallRegisterAtController
 */
special::CallRegisterAtController::~CallRegisterAtController(void) {
    this->client = NULL; // DO NOT DELETE
}
