/*
 * AbstractGetDataCall.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
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
