/*
 * testdiscovery.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testdiscovery.h"
#include "vislib/ClusterDiscoveryService.h"

#include "testhelper.h"


void TestClusterDiscoveryService(void) {
    using namespace vislib::net;
    
    ClusterDiscoveryService cds("testnet", 
        SocketAddress(SocketAddress::FAMILY_INET, IPAddress("129.69.215.38"), 28181), 
        IPAddress("129.69.215.255"), 
        SocketAddress(SocketAddress::FAMILY_INET, IPAddress("129.69.215.38"), 12345));
    cds.Start();
    vislib::sys::Thread::Sleep(5 * 1000);
    cds.Stop();
}
