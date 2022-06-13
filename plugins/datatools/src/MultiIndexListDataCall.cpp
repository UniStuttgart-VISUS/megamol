/*
 * MultiIndexListDataCall.cpp
 *
 * Copyright (C) 2016 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#include "datatools/MultiIndexListDataCall.h"

using namespace megamol;

datatools::MultiIndexListDataCall::MultiIndexListDataCall()
        : AbstractGetDataCall()
        , lsts(nullptr)
        , lsts_len(0)
        , frameCnt(0)
        , frameID(0) {
    // intentionally empty
}

datatools::MultiIndexListDataCall::~MultiIndexListDataCall() {
    lsts = nullptr; // not our memory, we do not delete
}
