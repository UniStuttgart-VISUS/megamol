/*
 * CallDescription.cpp
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/factories/CallDescription.h"
#include "mmcore/Call.h"

using namespace megamol::core;


/*
 * factories::CallDescription::CallDescription
 */
factories::CallDescription::CallDescription(void) : ObjectDescription() {
    // intentionally empty
}


/*
 * factories::CallDescription::~CallDescription
 */
factories::CallDescription::~CallDescription(void) {
    // intentionally empty
}


/*
 * factories::CallDescription::describeCall
 */
Call * factories::CallDescription::describeCall(Call * call) const {
    if (call != NULL) {
        call->funcMap = new unsigned int[this->FunctionCount()];
    }
    return call;
}
