/*
 * CallRegisterAtController.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"

#include "mmcore/cluster/CallRegisterAtController.h"

using namespace megamol::core;


/*
 * cluster::CallRegisterAtController::CallRegisterAtController
 */
cluster::CallRegisterAtController::CallRegisterAtController(void)
        : Call()
        , client(NULL)
        , statRun(false)
        , statPeerCnt(0)
        , statClstrName() {
    // intentionally empty
}


/*
 * cluster::CallRegisterAtController::~CallRegisterAtController
 */
cluster::CallRegisterAtController::~CallRegisterAtController(void) {
    this->client = NULL; // DO NOT DELETE
}
