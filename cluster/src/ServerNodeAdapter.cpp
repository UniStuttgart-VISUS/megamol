/*
 * ServerNodeAdapter.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/ServerNodeAdapter.h"

#include "vislib/IllegalParamException.h"
#include "vislib/SocketException.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"



/*
 * vislib::net::cluster::ServerNodeAdapter::~ServerNodeAdapter
 */
vislib::net::cluster::ServerNodeAdapter::~ServerNodeAdapter(void) {
}


/*
 * vislib::net::cluster::ServerNodeAdapter::Initialise
 */
void vislib::net::cluster::ServerNodeAdapter::Initialise(
        sys::CmdLineProviderA& inOutCmdLine) {
    Super::Initialise(inOutCmdLine);
    // TODO: Get server parameters
}


/*
 * vislib::net::cluster::ServerNodeAdapter::Initialise
 */
void vislib::net::cluster::ServerNodeAdapter::Initialise(
        sys::CmdLineProviderW& inOutCmdLine) {
    Super::Initialise(inOutCmdLine);
    // TODO: Get server parameters
}


/*
 * vislib::net::cluster::ServerNodeAdapter::OnNewConnection
 */
bool vislib::net::cluster::ServerNodeAdapter::OnNewConnection(Socket& socket,
        const SocketAddress& addr) throw() {
    this->socketsLock.Lock();
    this->sockets.Add(socket);
    this->socketsLock.Unlock();
    return true;
}


/*
 * vislib::net::cluster::ServerNodeAdapter::OnServerStopped
 */
void vislib::net::cluster::ServerNodeAdapter::OnServerStopped(void) throw() {
}


/*
 * vislib::net::cluster::ServerNodeAdapter::ServerNodeAdapter
 */
vislib::net::cluster::ServerNodeAdapter::ServerNodeAdapter(void) : Super() {
}


/*
 * vislib::net::cluster::ServerNodeAdapter::ServerNodeAdapter
 */
vislib::net::cluster::ServerNodeAdapter::ServerNodeAdapter(
        const ServerNodeAdapter& rhs) : Super(rhs) {
    throw UnsupportedOperationException("ServerNodeAdapter", __FILE__, 
        __LINE__);
}


/*
 * vislib::net::cluster::ServerNodeAdapter::countPeers
 */
SIZE_T vislib::net::cluster::ServerNodeAdapter::countPeers(void) const {
    SIZE_T retval = 0;
    this->socketsLock.Lock();
    retval = this->sockets.Count();
    this->socketsLock.Unlock();
    return retval;
}


/*
 * vislib::net::cluster::ServerNodeAdapter::forEachPeer
 */
SIZE_T vislib::net::cluster::ServerNodeAdapter::forEachPeer(
        ForeachPeerFunc func, void *context) {
    SIZE_T retval = 0;

    this->socketsLock.Lock();
    for (SIZE_T i = 0; i < this->sockets.Count(); i++) {
        try {
            bool isContinue = func(this, this->sockets[i], context);
            retval++;

            if (!isContinue) {
                break;
            }
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
 * vislib::net::cluster::ServerNodeAdapter::operator =
 */
vislib::net::cluster::ServerNodeAdapter& 
vislib::net::cluster::ServerNodeAdapter::operator =(
        const ServerNodeAdapter& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    Super::operator =(rhs);
    return *this;
}
