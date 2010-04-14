/*
 * DiscoveryListener.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */


#include "vislib/DiscoveryListener.h"


/*
 * vislib::net::cluster::DiscoveryListener::DiscoveryListener
 */
vislib::net::cluster::DiscoveryListener::DiscoveryListener(void) {
}


/*
 * vislib::net::cluster::DiscoveryListener::~DiscoveryListener
 */
vislib::net::cluster::DiscoveryListener::~DiscoveryListener(void) {
}


/*
 * vislib::net::cluster::DiscoveryListener::OnUserMessage
 */
void vislib::net::cluster::DiscoveryListener::OnUserMessage(
        DiscoveryService& src, 
        const DiscoveryService::PeerHandle& hPeer, 
        const bool isClusterMember,
        const UINT32 msgType, const BYTE *msgBody) throw() {
    /* Does nothing. */
}
