/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd/data/AbstractGetDataCall.h"

using namespace megamol::core;


/*
 * AbstractGetDataCall::AbstractGetDataCall
 */
AbstractGetDataCall::AbstractGetDataCall() : datahash(0), unlocker(NULL) {
    // intentionally empty
}


/*
 * AbstractGetDataCall::~AbstractGetDataCall
 */
AbstractGetDataCall::~AbstractGetDataCall() {
    this->Unlock();
}


/*
 * AbstractGetDataCall::operator=
 */
AbstractGetDataCall& AbstractGetDataCall::operator=(const AbstractGetDataCall& rhs) {
    this->datahash = rhs.datahash;
    this->unlocker = rhs.unlocker; // this is dangerous but documented!
    return *this;
}
