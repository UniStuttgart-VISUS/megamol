/*
 * DiscoveryTestApp.cpp
 *
 * Copyright (C) 2009 by Visualisierungsinstitut der Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#include "DiscoveryTestApp.h"

#include <iostream>

#include "vislib/DNS.h"
#include "vislib/IPEndPoint.h"
#include "vislib/NetworkInformation.h"
#include "vislib/SystemInformation.h"

using namespace vislib::net;
using namespace vislib::net::cluster;
using namespace vislib::sys;


/*
 * DiscoveryTestApp::GetInstance
 */
DiscoveryTestApp& DiscoveryTestApp::GetInstance(void) {
    static DiscoveryTestApp *instance = NULL;

    if (instance == NULL) {
        instance = new DiscoveryTestApp();
    }

    return *instance;
}


/*
 * DiscoveryTestApp::~DiscoveryTestApp
 */
DiscoveryTestApp::~DiscoveryTestApp(void) {
}


/*
 * DiscoveryTestApp::Initialise
 */
void DiscoveryTestApp::Initialise(CmdLineProviderA& inOutCmdLine) {
}


/*
 * DiscoveryTestApp::Initialise
 */
void DiscoveryTestApp::Initialise(CmdLineProviderW& inOutCmdLine) {
}


/*
 * DiscoveryTestApp::OnNodeFound
 */
void DiscoveryTestApp::OnNodeFound(DiscoveryService& src,
        const DiscoveryService::PeerHandle& hPeer) throw() {
    std::cout << "Discovery service \"" << src.GetName().c_str() 
        << "\" discovered new peer node " << src[hPeer].ToStringA().c_str()
        << std::endl
        << "Now, the following nodes are known to the service:" << std::endl;
    for (size_t i = 0; i < src.CountPeers(); i++) {
        std::cout << "\t" << src[i].ToStringA().c_str() << std::endl;
    }

    const char *msg = "Hello, nodes!";
    const_cast<DiscoveryService&>(src).SendUserMessage(
        DiscoveryService::MSG_TYPE_USER + 0, &msg, ::strlen(msg) + 1);

}


/*
 * DiscoveryTestApp::OnNodeLost
 */
void DiscoveryTestApp::OnNodeLost(DiscoveryService& src,
        const DiscoveryService::PeerHandle& hPeer,
        const DiscoveryListener::NodeLostReason reason) throw() {
    std::cout << "Discovery service " << src.GetName().c_str() 
        << " lost peer node " << src[hPeer].ToStringA().c_str()
        << " for reason " << reason 
        << std::endl
        << "from the following list of known nodes:" << std::endl;
    for (size_t i = 0; i < src.CountPeers(); i++) {
        std::cout << "\t" << src[hPeer].ToStringA().c_str() << std::endl;
    }
}


/*
 * DiscoveryTestApp::OnUserMessage
 */
void DiscoveryTestApp::OnUserMessage(DiscoveryService& src,
        const DiscoveryService::PeerHandle& hPeer,
        const bool isClusterMember,
        const uint32_t msgType, const uint8_t *msgBody) throw() {
    std::cout << "Discovery service " << src.GetName().c_str() 
        << " received user message " << msgType 
        << " from peer node " << src[hPeer].ToStringA().c_str()
        << std::endl;
}


/*
 * DiscoveryTestApp::Run
 */
unsigned int DiscoveryTestApp::Run(void) {
    char dowel;
    size_t cntCfgs = 0;
    IPAddress adapter;
    IPAddress bcastAddress;
    DiscoveryService::DiscoveryConfig cfg;

    try {
        Socket::Startup();

        DNS::GetHostAddress(adapter, SystemInformation::ComputerNameW().c_str());
        //for (unsigned int i = 0; i < NetworkInformation::AdapterCount(); i++) {
        //    NetworkInformation::Adapter ai = NetworkInformation::AdapterInformation(i);
        //    if ((adapter & ai.SubnetMask()) == (ai.BroadcastAddress() & ai.SubnetMask())) {
        //        bcastAddress = ai.BroadcastAddress();
        //        break;
        //    }
        //}
        //cfg = DiscoveryService::DiscoveryConfig(IPEndPoint(adapter, 40000),adapter, bcastAddress);
        
        cfg = DiscoveryService::DiscoveryConfig(IPEndPoint(adapter, 40000), adapter);

        this->cds.AddListener(this);
        this->cds.Start("CDS2Test", &cfg, 1, 2, 0/*DiscoveryService::FLAG_SHARE_SOCKETS*/, DiscoveryService::DEFAULT_REQUEST_INTERVAL, 3);
        //this->cds.Start("MegaMolRenderCluster", &cfg, 1, 2, 0/*DiscoveryService::FLAG_SHARE_SOCKETS*/, DiscoveryService::DEFAULT_REQUEST_INTERVAL, 1);

        Socket::Cleanup();
    } catch (the::exception& e) {
        std::cerr << "VISlib exception caught: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception caught." << std::endl;
    }

    std::cin >> dowel;

    try {
        this->cds.Stop();
        Socket::Cleanup();
    } catch (the::exception& e) {
        std::cerr << "VISlib exception caught: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception caught." << std::endl;
    }

    return 0;
}


/*
 * DiscoveryTestApp::DiscoveryTestApp
 */
DiscoveryTestApp::DiscoveryTestApp(void) {
}
