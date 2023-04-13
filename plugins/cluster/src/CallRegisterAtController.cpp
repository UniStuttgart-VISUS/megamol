/*
 * CallRegisterAtController.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "cluster/CallRegisterAtController.h"

using namespace megamol::core;


/*
 * cluster::CallRegisterAtController::CallRegisterAtController
 */
cluster::CallRegisterAtController::CallRegisterAtController()
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
cluster::CallRegisterAtController::~CallRegisterAtController() {
    this->client = NULL; // DO NOT DELETE
}
