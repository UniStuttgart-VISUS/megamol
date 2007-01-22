/*
 * testdiscovery.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testdiscovery.h"

#include "vislib/ClusterDiscoveryService.h"


void TestClusterDiscoveryService(void) {
    using namespace vislib::net;

    ClusterDiscoveryService cds("testnet", SocketAddress(SocketAddress::FAMILY_INET, IPAddress(), 28181), IPAddress("129.69.215.255"));
}
