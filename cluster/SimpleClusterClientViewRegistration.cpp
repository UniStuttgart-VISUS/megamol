/*
 * SimpleClusterClientViewRegistration.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/SimpleClusterClientViewRegistration.h"

using namespace megamol::core;


/*
 * cluster::SimpleClusterClientViewRegistration::SimpleClusterClientViewRegistration
 */
cluster::SimpleClusterClientViewRegistration::SimpleClusterClientViewRegistration(void) : Call(),
        client(NULL), view(NULL) {
    // intentionally empty
}


/*
 * cluster::SimpleClusterClientViewRegistration::~SimpleClusterClientViewRegistration
 */
cluster::SimpleClusterClientViewRegistration::~SimpleClusterClientViewRegistration(void) {
    this->client = NULL; // DO NOT DELETE
    this->view = NULL; // DO NOT DELETE
}
