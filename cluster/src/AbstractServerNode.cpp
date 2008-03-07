/*
 * AbstractServerNode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractServerNode.h"

#include "vislib/IllegalParamException.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::net::cluster::AbstractServerNode::~AbstractServerNode
 */
vislib::net::cluster::AbstractServerNode::~AbstractServerNode(void) {
    // TODO: Implement
}


/*
 * vislib::net::cluster::AbstractServerNode::OnNewConnection
 */
bool vislib::net::cluster::AbstractServerNode::OnNewConnection(Socket& socket,
        const SocketAddress& addr) throw() {
    return false;
}


/*
 * vislib::net::cluster::AbstractServerNode::OnServerStopped
 */
void vislib::net::cluster::AbstractServerNode::OnServerStopped(void) throw() {
}


/*
 * vislib::net::cluster::AbstractServerNode::AbstractServerNode
 */
vislib::net::cluster::AbstractServerNode::AbstractServerNode(void) 
        : AbstractClusterNode() {
}


/*
 * vislib::net::cluster::AbstractServerNode::AbstractServerNode
 */
vislib::net::cluster::AbstractServerNode::AbstractServerNode(
        const AbstractServerNode& rhs) : AbstractClusterNode(rhs) {
    throw UnsupportedOperationException("AbstractServerNode", __FILE__, 
        __LINE__);
}


/*
 * vislib::net::cluster::AbstractServerNode::operator =
 */
vislib::net::cluster::AbstractServerNode& 
vislib::net::cluster::AbstractServerNode::operator =(
        const AbstractServerNode& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    AbstractClusterNode::operator =(rhs);
    return *this;
}
