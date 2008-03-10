/*
 * AbstractServerNode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractServerNode.h"

#include "vislib/IllegalParamException.h"
#include "vislib/SocketException.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::net::cluster::AbstractServerNode::~AbstractServerNode
 */
vislib::net::cluster::AbstractServerNode::~AbstractServerNode(void) {
    // TODO: Implement
}


/*
 * vislib::net::cluster::AbstractServerNode::Initialise
 */
void vislib::net::cluster::AbstractServerNode::Initialise(
        sys::CmdLineProviderA& inOutCmdLine) {
    AbstractClusterNode::Initialise(inOutCmdLine);
}


/*
 * vislib::net::cluster::AbstractServerNode::Initialise
 */
void vislib::net::cluster::AbstractServerNode::Initialise(
        sys::CmdLineProviderW& inOutCmdLine) {
    AbstractClusterNode::Initialise(inOutCmdLine);
}


/*
 * vislib::net::cluster::AbstractServerNode::OnNewConnection
 */
bool vislib::net::cluster::AbstractServerNode::OnNewConnection(Socket& socket,
        const SocketAddress& addr) throw() {
    this->socketsLock.Lock();
    this->sockets.Add(socket);
    this->socketsLock.Unlock();
    return true;
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
 * vislib::net::cluster::AbstractServerNode::forEachPeer
 */
SIZE_T vislib::net::cluster::AbstractServerNode::forEachPeer(
        ForeachPeerFunc func, void *context) {
    SIZE_T retval = 0;

    this->socketsLock.Lock();
    for (SIZE_T i = 0; i < this->sockets.Count(); i++) {
        try {
            func(this, this->sockets[i], context);
            retval++;
        } catch (Exception e) {
            TRACE(Trace::LEVEL_VL_WARN, "ForeachPeerFunc failed for node %u "
                "with an exception: %s", i, e.GetMsgA());
        } catch (...) {
            TRACE(Trace::LEVEL_VL_WARN, "ForeachPeerFunc failed for node %u "
                "with a non-VISlib exception.", i);
        }
    }
    this->socketsLock.Unlock();

    return retval;
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
