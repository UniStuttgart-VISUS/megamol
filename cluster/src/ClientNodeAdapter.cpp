/*
 * ClientNodeAdapter.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/ClientNodeAdapter.h"

#include "vislib/Trace.h"


/*
 * vislib::net::cluster::ClientNodeAdapter::~ClientNodeAdapter
 */
vislib::net::cluster::ClientNodeAdapter::~ClientNodeAdapter(void) {
}

/*
 * vislib::net::cluster::ClientNodeAdapter::Initialise
 */
void vislib::net::cluster::ClientNodeAdapter::Initialise(
        sys::CmdLineProviderA& inOutCmdLine) {
    Super::Initialise(inOutCmdLine);
    // TODO: Get client parameters
}


/*
 * vislib::net::cluster::ClientNodeAdapter::Initialise
 */
void vislib::net::cluster::ClientNodeAdapter::Initialise(
        sys::CmdLineProviderW& inOutCmdLine) {
    Super::Initialise(inOutCmdLine);
    // TODO: Get client parameters
}


/*
 * vislib::net::cluster::ClientNodeAdapter::ClientNodeAdapter
 */
vislib::net::cluster::ClientNodeAdapter::ClientNodeAdapter(void) : Super() {
}


/*
 * vislib::net::cluster::ClientNodeAdapter::ClientNodeAdapter
 */
vislib::net::cluster::ClientNodeAdapter::ClientNodeAdapter(
        const ClientNodeAdapter& rhs) : Super(rhs) {
}


/*
 * vislib::net::cluster::ClientNodeAdapter::countPeers
 */
SIZE_T vislib::net::cluster::ClientNodeAdapter::countPeers(void) const {
    return (this->socket.IsValid() ? 1 : 0);
}


/*
 * vislib::net::cluster::ClientNodeAdapter::forEachPeer
 */
SIZE_T vislib::net::cluster::ClientNodeAdapter::forEachPeer(
        ForeachPeerFunc func, void *context) {
    SIZE_T retval = 0;

    try {
        func(this, this->socket, context);
        retval = 1;
    } catch (Exception e) {
        TRACE(Trace::LEVEL_VL_WARN, "ForeachPeerFunc failed "
            "with an exception: %s", e.GetMsgA());
    } catch (...) {
        TRACE(Trace::LEVEL_VL_WARN, "ForeachPeerFunc failed "
            "with a non-VISlib exception.");
    }

    return retval;
}


/*
 * vislib::net::cluster::ClientNodeAdapter::operator =
 */
vislib::net::cluster::ClientNodeAdapter& 
vislib::net::cluster::ClientNodeAdapter::operator =(
        const ClientNodeAdapter& rhs) {
    Super::operator =(rhs);
    return *this;
}
