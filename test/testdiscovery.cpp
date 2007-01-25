/*
 * testdiscovery.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testdiscovery.h"
#include "vislib/ClusterDiscoveryService.h"

#include <iostream>

#include "testhelper.h"


class MyListener : public vislib::net::ClusterDiscoveryListener {
    
    virtual void OnNodeFound(const vislib::net::ClusterDiscoveryService& src, 
            const vislib::net::SocketAddress& addr);
};


/*
 * MyListener::OnNodeFound
 */
void MyListener::OnNodeFound(const vislib::net::ClusterDiscoveryService& src, 
        const vislib::net::SocketAddress& addr) {
    std::cout << addr.ToStringA() << " was found for \"" << src.GetName() 
        << "\"" << std::endl;
}


void TestClusterDiscoveryService(void) {
    using namespace vislib::net;

    MyListener myListener;
    
    ClusterDiscoveryService cds("testnet", 
        SocketAddress(SocketAddress::FAMILY_INET, IPAddress("129.69.215.38"), 28181), 
        IPAddress("129.69.215.255"), 
        SocketAddress(SocketAddress::FAMILY_INET, IPAddress("129.69.215.38"), 12345));
    cds.AddListener(&myListener);
    cds.Start();
    vislib::sys::Thread::Sleep(5 * 1000);
    cds.Stop();
}
