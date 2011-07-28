/*
 * ClientViewRegistration.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/simple/ClientViewRegistration.h"

using namespace megamol::core;


/*
 * cluster::simple::ClientViewRegistration::ClientViewRegistration
 */
cluster::simple::ClientViewRegistration::ClientViewRegistration(void) : Call(),
        client(NULL), view(NULL) {
    // intentionally empty
}


/*
 * cluster::simple::ClientViewRegistration::~ClientViewRegistration
 */
cluster::simple::ClientViewRegistration::~ClientViewRegistration(void) {
    this->client = NULL; // DO NOT DELETE
    this->view = NULL; // DO NOT DELETE
}
