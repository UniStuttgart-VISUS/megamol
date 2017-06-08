/*
 * Call.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"

using namespace megamol::core;


/*
 * Call::Call
 */
Call::Call(void) : callee(NULL), caller(NULL), funcMap(NULL) {
    // intentionally empty
}


/*
 * Call::~Call
 */
Call::~Call(void) {
    if (this->caller != NULL) {
        CallerSlot *cr = this->caller;
        this->caller = NULL; // DO NOT DELETE
        cr->ConnectCall(NULL);
    }
    if (this->callee != NULL) {
        this->callee->ConnectCall(NULL);
        this->callee = NULL; // DO NOT DELETE
    }
    ARY_SAFE_DELETE(this->funcMap);
}


/*
 * Call::operator()
 */
bool Call::operator()(unsigned int func) {
    if (this->callee != NULL) {
        return this->callee->InCall(this->funcMap[func], *this);
    }
    return false;
}
