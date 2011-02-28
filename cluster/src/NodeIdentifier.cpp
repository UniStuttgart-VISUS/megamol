/*
 * NodeIdentifier.cpp
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/NodeIdentifier.h"


/*
 * vislib::net::cluster::NodeIdentifier::NodeIdentifier
 */
vislib::net::cluster::NodeIdentifier::NodeIdentifier(void) {
    // TODO: Implement

}


/*
 * vislib::net::cluster::NodeIdentifier::~NodeIdentifier
 */
vislib::net::cluster::NodeIdentifier::~NodeIdentifier(void) {
    VLSTACKTRACE("NodeIdentifier::~NodeIdentifier", __FILE__, __LINE__);
}


/*
 * vislib::net::cluster::NodeIdentifier::operator =
 */
vislib::net::cluster::NodeIdentifier& 
vislib::net::cluster::NodeIdentifier::operator =(const NodeIdentifier& rhs) {
    VLSTACKTRACE("NodeIdentifier::operator =", __FILE__, __LINE__);
    if (this != &rhs) {
        this->id = rhs.id;
    }
    return *this;
}
