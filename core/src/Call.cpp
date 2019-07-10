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
#include "vislib/sys/Log.h"

using namespace megamol::core;


/*
 * Call::Call
 */
Call::Call(void) : callee(nullptr), caller(nullptr), className(nullptr), funcMap(nullptr) {
    // intentionally empty
}


/*
 * Call::~Call
 */
Call::~Call(void) {
    if (this->caller != nullptr) {
        CallerSlot *cr = this->caller;
        this->caller = nullptr; // DO NOT DELETE
        cr->ConnectCall(nullptr);
    }
    if (this->callee != nullptr) {
        this->callee->ConnectCall(nullptr);
        this->callee = nullptr; // DO NOT DELETE
    }
    ARY_SAFE_DELETE(this->funcMap);
}


/*
 * Call::operator()
 */
bool Call::operator()(unsigned int func) {
    bool res = false;
    if (this->callee != nullptr) {
        res = this->callee->InCall(this->funcMap[func], *this);
    }
    //vislib::sys::Log::DefaultLog.WriteInfo("calling %s, idx %i, result %s (%s)", this->ClassName(), func,
    //    res ? "true" : "false", this->callee == nullptr ? "no callee" : "from callee");
    return res;
}
