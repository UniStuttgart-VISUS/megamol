/*
 * NodeIdentifier.cpp
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/net/cluster/NodeIdentifier.h"

#include "vislib/assert.h"


/*
 * vislib::net::cluster::NodeIdentifier::NodeIdentifier
 */
vislib::net::cluster::NodeIdentifier::NodeIdentifier(void) {
    ASSERT(this->IsNull());
}


/*
 * vislib::net::cluster::NodeIdentifier::~NodeIdentifier
 */
vislib::net::cluster::NodeIdentifier::~NodeIdentifier(void) {
}


/*
 * vislib::net::cluster::NodeIdentifier::operator =
 */
vislib::net::cluster::NodeIdentifier& 
vislib::net::cluster::NodeIdentifier::operator =(const NodeIdentifier& rhs) {
    if (this != &rhs) {
        this->id = rhs.id;
    }
    return *this;
}
