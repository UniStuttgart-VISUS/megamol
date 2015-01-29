/*
 * CallDescription.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallDescription.h"
#include "Call.h"

using namespace megamol::core;


/*
 * CallDescription::CallDescription
 */
CallDescription::CallDescription(void) : ObjectDescription() {
}


/*
 * CallDescription::~CallDescription
 */
CallDescription::~CallDescription(void) {
}


/*
 * CallDescription::describeCall
 */
Call * CallDescription::describeCall(Call * call) const {
    if (call != NULL) {
        call->funcMap = new unsigned int[this->FunctionCount()];
    }
    return call;
}
