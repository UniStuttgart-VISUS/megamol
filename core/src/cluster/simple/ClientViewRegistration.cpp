/*
 * ClientViewRegistration.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/cluster/simple/ClientViewRegistration.h"

#include "mmcore/cluster/simple/View.h"


using namespace megamol::core;


/*
 * cluster::simple::ClientViewRegistration::ClientViewRegistration
 */
cluster::simple::ClientViewRegistration::ClientViewRegistration(void) : Call(),
        client(NULL), view(NULL), heartbeat(NULL),
        isRawMessageDispatching(false) {
    // intentionally empty
}


/*
 * cluster::simple::ClientViewRegistration::~ClientViewRegistration
 */
cluster::simple::ClientViewRegistration::~ClientViewRegistration(void) {
    this->client = NULL; // DO NOT DELETE
    this->view = NULL; // DO NOT DELETE
}


/*
 * cluster::simple::ClientViewRegistration::GetRawMessageDispatchListener
 */
vislib::net::SimpleMessageDispatchListener *
cluster::simple::ClientViewRegistration::GetRawMessageDispatchListener(void) {
    if (this->isRawMessageDispatching) {
        return dynamic_cast<vislib::net::SimpleMessageDispatchListener *>(
            this->view);
    } else {
        return NULL;
    }
}
