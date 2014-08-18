/*
 * ClusterDiscoveryListener.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */


#include "vislib/ClusterDiscoveryListener.h"


/*
 * vislib::net::ClusterDiscoveryListener::ClusterDiscoveryListener
 */
vislib::net::ClusterDiscoveryListener::ClusterDiscoveryListener(void) {
}


/*
 * vislib::net::ClusterDiscoveryListener::~ClusterDiscoveryListener
 */
vislib::net::ClusterDiscoveryListener::~ClusterDiscoveryListener(void) {
}


/*
 * vislib::net::ClusterDiscoveryListener::OnUserMessage
 */
void vislib::net::ClusterDiscoveryListener::OnUserMessage(
        const ClusterDiscoveryService& src, 
        const ClusterDiscoveryService::PeerHandle& hPeer, const UINT32 msgType,
        const BYTE *msgBody) {
    /* Does nothing. */
}
