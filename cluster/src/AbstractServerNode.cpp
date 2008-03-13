/*
 * AbstractServerNode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractServerNode.h"


/*
 * vislib::net::cluster::AbstractServerNode::~AbstractServerNode
 */
vislib::net::cluster::AbstractServerNode::~AbstractServerNode(void) {
}


/*
 * vislib::net::cluster::AbstractServerNode::AbstractServerNode
 */
vislib::net::cluster::AbstractServerNode::AbstractServerNode(void) 
        : AbstractClusterNode(), TcpServer::Listener() {
}


/*
 * vislib::net::cluster::AbstractServerNode::AbstractServerNode
 */
vislib::net::cluster::AbstractServerNode::AbstractServerNode(
        const AbstractServerNode& rhs) 
        : AbstractClusterNode(rhs), TcpServer::Listener(rhs) {
}


/*
 * vislib::net::cluster::AbstractServerNode::operator =
 */
vislib::net::cluster::AbstractServerNode& 
vislib::net::cluster::AbstractServerNode::operator =(
        const AbstractServerNode& rhs) {
    AbstractClusterNode::operator =(rhs);
    TcpServer::Listener::operator =(rhs);
    return *this;
}
