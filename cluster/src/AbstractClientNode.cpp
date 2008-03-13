/*
 * AbstractClientNode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractClientNode.h"


/*
 * vislib::net::cluster::AbstractClientNode::~AbstractClientNode
 */
vislib::net::cluster::AbstractClientNode::~AbstractClientNode(void) {
}


/*
 * vislib::net::cluster::AbstractClientNode::AbstractClientNode
 */
vislib::net::cluster::AbstractClientNode::AbstractClientNode(void) 
        : AbstractClusterNode() {
}


/*
 * vislib::net::cluster::AbstractClientNode::AbstractClientNode
 */
vislib::net::cluster::AbstractClientNode::AbstractClientNode(
        const AbstractClientNode& rhs) : AbstractClusterNode(rhs) {
}


/*
 * vislib::net::cluster::AbstractClientNode::operator =
 */
vislib::net::cluster::AbstractClientNode& 
vislib::net::cluster::AbstractClientNode::operator =(
        const AbstractClientNode& rhs) {
    AbstractClusterNode::operator =(rhs);
    return *this;
}
