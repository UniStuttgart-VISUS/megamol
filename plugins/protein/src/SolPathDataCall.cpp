/*
 * SolPathDataCall.cpp
 *
 * Copyright (C) 2010 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "protein/SolPathDataCall.h"

using namespace megamol::protein;


/*
 * SolPathDataCall::SolPathDataCall
 */
SolPathDataCall::SolPathDataCall(void)
        : core::AbstractGetData3DCall()
        , count(0)
        , lines(NULL)
        , minTime(0.0f)
        , maxTime(0.0f)
        , minSpeed(0.0f)
        , maxSpeed(0.0f) {
    // intentionally empty
}


/*
 * SolPathDataCall::~SolPathDataCall
 */
SolPathDataCall::~SolPathDataCall(void) {
    this->Unlock(); // just for paranoia reasons
    this->count = 0;
    this->lines = NULL; // DO NOT DELETE!
    this->minTime = 0.0f;
    this->maxTime = 0.0f;
    this->minSpeed = 0.0f;
    this->maxSpeed = 0.0f;
}
