/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/CallDescription.h"
#include "mmcore/Call.h"

using namespace megamol::core;

/*
 * factories::CallDescription::describeCall
 */
Call* factories::CallDescription::describeCall(Call* call) const {
    if (call != nullptr) {
        call->funcMap = new unsigned int[this->FunctionCount()];
    }
    return call;
}
