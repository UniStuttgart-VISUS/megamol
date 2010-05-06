/*
 * CallRegisterAtController.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/CallRegisterAtController.h"

using namespace megamol::core;


///*
// * cluster::CallRegisterAtController::CALL_REGISTER
// */
//const unsigned int cluster::CallRegisterAtController::CALL_REGISTER = 0;
//
//
///*
// * cluster::CallRegisterAtController::CALL_UNREGISTER
// */
//const unsigned int cluster::CallRegisterAtController::CALL_UNREGISTER = 1;
//
//
///*
// * cluster::CallRegisterAtController::CALL_GETSTATUS
// */
//const unsigned int cluster::CallRegisterAtController::CALL_GETSTATUS = 2;


/*
 * cluster::CallRegisterAtController::CallRegisterAtController
 */
cluster::CallRegisterAtController::CallRegisterAtController(void) : Call(),
        client(NULL), statRun(false), statPeerCnt(0) {
    // intentionally empty
}


/*
 * cluster::CallRegisterAtController::~CallRegisterAtController
 */
cluster::CallRegisterAtController::~CallRegisterAtController(void) {
    this->client = NULL; // DO NOT DELETE
}
