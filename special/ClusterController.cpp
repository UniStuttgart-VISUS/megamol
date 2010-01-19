/*
 * ClusterController.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ClusterController.h"
#include <climits>
#include "CoreInstance.h"
#include "param/BoolParam.h"
#include "param/StringParam.h"
#include "param/IntParam.h"
#include "special/CallRegisterAtController.h"
#include "special/ClusterControllerClient.h"
#include "vislib/assert.h"
#include "vislib/AutoLock.h"
#include "vislib/DiscoveryService.h"
#include "vislib/IPAddress.h"
#include "vislib/IPAddress6.h"
#include "vislib/Log.h"
#include "vislib/NetworkInformation.h"
#include "vislib/Socket.h"

using namespace megamol::core;
using vislib::sys::Log;
using vislib::net::cluster::DiscoveryService;
using vislib::net::NetworkInformation;


/*
 * special::ClusterController::DEFAULT_CLUSTERNAME
 */
const char * special::ClusterController::DEFAULT_CLUSTERNAME = "MMRndCluster";


/*
 * special::ClusterController::ClusterController
 */
special::ClusterController::ClusterController() : job::AbstractJobThread(),
        Module(), vislib::net::cluster::DiscoveryListener(),
        clusterName("ClusterName", "Name of the rendering cluster"),
        cdsAdapterAddress("cdsAdapterAddress", "Address of the adapter for the cluster discovery"),
        cdsBCastAddress("cdsBCastAddress", "Broadcast IP address for the cluster discovery"),
        cdsPort("cdsPort", "Port for the cluster discovery"),
        cdsRun("cdsRun", "Start/Stop flag for the cluster discovery"),
        discoveryService(),
        registerSlot("register", "Slot to register modules at, which wish to use this controller"),
        clients(), clientsLock() {
    vislib::net::Socket::Startup();

    this->discoveryService.AddListener(this);

    this->clusterName << new param::StringParam(DEFAULT_CLUSTERNAME);
    this->MakeSlotAvailable(&this->clusterName);

    this->cdsAdapterAddress << new param::StringParam("");
    this->MakeSlotAvailable(&this->cdsAdapterAddress);

    this->cdsBCastAddress << new param::StringParam("");
    this->MakeSlotAvailable(&this->cdsBCastAddress);

    // Note: Port minval should be 49152, which is the lowest port for the
    //  private dynamic ip port range. However, we do not want to enforce
    //  using only these ports.
    this->cdsPort << new param::IntParam(
        vislib::net::cluster::DiscoveryService::DEFAULT_PORT, 1, USHRT_MAX);
    this->MakeSlotAvailable(&this->cdsPort);

    this->cdsRun << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->cdsRun);

    this->registerSlot.SetCallback("CallRegisterAtController", "register",
        &ClusterController::registerModule);
    this->registerSlot.SetCallback("CallRegisterAtController", "unregister",
        &ClusterController::unregisterModule);
    this->MakeSlotAvailable(&this->registerSlot);
}


/*
 * special::ClusterController::~ClusterController
 */
special::ClusterController::~ClusterController() {
    this->Release();
    ASSERT(!this->discoveryService.IsRunning());
    ASSERT(this->clients.IsEmpty());
    try {
        vislib::net::Socket::Cleanup();
    } catch(...) {
    }
}


/*
 * special::ClusterController::create
 */
bool special::ClusterController::create(void) {

    const utility::Configuration& cfg
        = this->GetCoreInstance()->Configuration();

    if (cfg.IsConfigValueSet("ClusterName")) {
        this->clusterName.Param<param::StringParam>()->SetValue(
            cfg.ConfigValue("ClusterName").PeekBuffer());
    }
    if (cfg.IsConfigValueSet("CDSAdapterAddress")) {
        this->cdsAdapterAddress.Param<param::StringParam>()->SetValue(
            cfg.ConfigValue("CDSAdapterAddress").PeekBuffer());
    }
    if (cfg.IsConfigValueSet("CDSBroadcastAddress")) {
        this->cdsBCastAddress.Param<param::StringParam>()->SetValue(
            cfg.ConfigValue("CDSBroadcastAddress").PeekBuffer());
    }
    if (cfg.IsConfigValueSet("CDSPort")) {
        try {
            this->cdsPort.Param<param::IntParam>()->SetValue(
                vislib::CharTraitsW::ParseInt(
                cfg.ConfigValue("CDSPort")));
        } catch(...) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                "Unable to parse configuration value \"CDSPort\" as number. "
                "Configuration value ignored.");
        }
    }
    if (cfg.IsConfigValueSet("CDSRun")) {
        try {
            this->cdsRun.Param<param::BoolParam>()->SetValue(
                vislib::CharTraitsW::ParseBool(
                cfg.ConfigValue("CDSRun")));
        } catch(...) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                "Unable to parse configuration value \"CDSRun\" as boolean. "
                "Configuration value ignored.");
        }
    }

    // to trigger initialization on first parameter realization
    this->clusterName.ForceSetDirty();
    return true;
}


/*
 * special::ClusterController::release
 */
void special::ClusterController::release(void) {
    this->stopDiscoveryService();

    this->clientsLock.Lock();
    vislib::SingleLinkedList<ClusterControllerClient *>::Iterator iter
        = this->clients.GetIterator();
    while (iter.HasNext()) {
        ClusterControllerClient *c = iter.Next();
        if (c == NULL) continue;
        c->OnClusterUnavailable();
        c->ctrlr = NULL; // implicitly disconnect clients
    }
    this->clients.Clear();
    this->clientsLock.Unlock();

    // TODO: Implement

}


/*
 * special::ClusterController::Run
 */
DWORD special::ClusterController::Run(void *userData) {
    const unsigned int sleepTime = 250;

    while (!this->shouldTerminate()) {

        if (this->clusterName.IsDirty() || this->cdsAdapterAddress.IsDirty()
                || this->cdsBCastAddress.IsDirty() || this->cdsPort.IsDirty()
                || this->cdsRun.IsDirty()) {
            this->stopDiscoveryService();

            // update the cluster discovery service
            this->clusterName.ResetDirty();
            this->cdsAdapterAddress.ResetDirty();
            this->cdsBCastAddress.ResetDirty();
            this->cdsPort.ResetDirty();
            this->cdsRun.ResetDirty();

            vislib::net::IPAddress addr4, bcast4;
            vislib::net::IPAddress6 addr6, bcast6;
            bool ip4valid = false, ip6valid = false;
            //unsigned short port;

            vislib::StringA clustName = T2A(this->clusterName.Param<
                param::StringParam>()->Value());
            if (clustName.IsEmpty()) {
                clustName = DEFAULT_CLUSTERNAME;
                this->clusterName.Param<param::StringParam>()->SetValue(
                    clustName.PeekBuffer());
            }

            //bool run = this->cdsRun.Param<param::BoolParam>()->Value();

            //if (!this->getServiceNetConfig(addr4, bcast4, ip4valid, addr6,
            //        bcast6, ip6valid, port)) {
                //run = false;
            //}

            //if (run) {

            //    vislib::SmartPtr<DiscoveryService::DiscoveryConfig> config;

            //    if (ip4valid) {
            //        config = new DiscoveryService::DiscoveryConfig(
            //            vislib::net::IPEndPoint(addr4, port),
            //            /*addr4*/vislib::net::IPAddress::ANY, bcast4, port);

            //    } else if (ip6valid) {
            //        config = new DiscoveryService::DiscoveryConfig(
            //            vislib::net::IPEndPoint(addr6, port),
            //            addr6, bcast6, port);

            //    } else {
            //        ASSERT(false);
            //    }

            //    try {
            //        UINT32 flags = 0; // DiscoveryService::FLAG_SHARE_SOCKETS;
            //        this->discoveryService.Start(clustName.PeekBuffer(),
            //            config.operator->(), 1, 0, flags);

            //    } catch (vislib::Exception e) {
            //        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            //            "Failed to start CDS: %s\n", e.GetMsgA());
            //    } catch (...) {
            //        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            //            "Failed to start CDS: unexpected exception\n");
            //    }

            //    if (this->discoveryService.IsRunning()) {
            //        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "CDS started");
            //    } else {
            //        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            //            "Failed to start CDS: unknown reason\n");
            //    }

            //}
        }

        vislib::sys::Thread::Sleep(sleepTime);

    }
    this->stopDiscoveryService();

    return 0;
}


/*
 * special::ClusterController::OnNodeFound
 */
void special::ClusterController::OnNodeFound(const DiscoveryService& src,
        const DiscoveryService::PeerHandle& hPeer) throw() {
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Cluster Node found: %s\n",
        src.GetDiscoveryAddress4(hPeer).ToStringA().PeekBuffer());

    // TODO: Implement

}


/*
 * special::ClusterController::OnNodeLost
 */
void special::ClusterController::OnNodeLost(const DiscoveryService& src,
        const DiscoveryService::PeerHandle& hPeer,
        const DiscoveryListener::NodeLostReason reason) throw() {
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Cluster Node lost: %s\n",
        src.GetDiscoveryAddress4(hPeer).ToStringA().PeekBuffer());

    // TODO: Implement

}


/*
 * special::ClusterController::OnUserMessage
 */
void special::ClusterController::OnUserMessage(const DiscoveryService& src,
        const DiscoveryService::PeerHandle& hPeer,
        const UINT32 msgType, const BYTE *msgBody) throw() {
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Cluster User Message: from %s\n",
        src.GetDiscoveryAddress4(hPeer).ToStringA().PeekBuffer());

    // TODO: Implement

}


/*
 * special::ClusterController::stopDiscoveryService
 */
void special::ClusterController::stopDiscoveryService(void) {
    if (this->discoveryService.IsRunning()) {
        if (this->discoveryService.Stop()) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "CDS stopped");
        } else {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Failed to stop CDS");
        }
    }
}


///*
// * special::ClusterController::getServiceNetConfig
// */
//bool special::ClusterController::getServiceNetConfig(
//        vislib::net::IPAddress& addr4, vislib::net::IPAddress& bcast4,
//        bool& ip4Valid, vislib::net::IPAddress6& addr6,
//        vislib::net::IPAddress6& bcast6, bool& ip6Valid,
//        unsigned short& port) {
//    ip4Valid = ip6Valid = false; // just to be sure
//
//    // port
//    int iport = this->cdsPort.Param<param::IntParam>()->Value();
//    if (iport < 1) {
//        iport = 1;
//        this->cdsPort.Param<param::IntParam>()->SetValue(iport);
//    } else
//    if (iport > USHRT_MAX) {
//        iport = USHRT_MAX;
//        this->cdsPort.Param<param::IntParam>()->SetValue(iport);
//    }
//    port = static_cast<unsigned short>(iport);
//
//    // addresses
//    vislib::StringA adapAddr = T2A(this->cdsAdapterAddress.Param<
//        param::StringParam>()->Value());
//    vislib::StringA bcastAddr = T2A(this->cdsBCastAddress.Param<
//        param::StringParam>()->Value());
//
//    // parse adapter address
//    bool adapAddr4Valid = false;
//    bool adapAddr6Valid = false;
//    if (!adapAddr.IsEmpty()) {
//        try {
//            addr4 = vislib::net::IPAddress::Create(adapAddr);
//            adapAddr4Valid = true;
//        } catch(...) {
//            try {
//                addr6 = vislib::net::IPAddress6::Create(adapAddr);
//                adapAddr6Valid = true;
//            } catch(...) {
//            }
//        }
//    }
//
//    // parse broadcast address
//    bool bcastAddr4Valid = false;
//    bool bcastAddr6Valid = false;
//    if (!bcastAddr.IsEmpty()) {
//        try {
//            bcast4 = vislib::net::IPAddress::Create(bcastAddr);
//            bcastAddr4Valid = true;
//        } catch(...) {
//            try {
//                bcast6 = vislib::net::IPAddress6::Create(bcastAddr);
//                bcastAddr6Valid = true;
//            } catch(...) {
//            }
//        }
//    }
//
//
//    if (!adapAddr4Valid && !adapAddr6Valid) {
//
//        if (bcastAddr6Valid) {
//            // try to find a matching adapter
//            // TODO: how?
//
//        } else if (bcastAddr4Valid) {
//            // try to find a matching adapter
//            unsigned int adapCnt = NetworkInformation::AdapterCount();
//            for (unsigned int i = 0; i < adapCnt; i++) {
//                const NetworkInformation::Adapter& adap
//                    = NetworkInformation::AdapterInformation(i);
//                if ((adap.AddressValidity()
//                            == NetworkInformation::Adapter::VALID)
//                        && (adap.BroadcastAddressValidity()
//                            != NetworkInformation::Adapter::NOT_VALID)) {
//                    if (adap.Address() == vislib::net::IPAddress::ANY) {
//                        // does not make any sense using ANY address
//                        continue;
//                    }
//
//                    if (adap.BroadcastAddress() == bcast4) {
//                        addr4 = adap.Address();
//                        adapAddr4Valid = true;
//                        break;
//                    }
//                }
//            }
//
//        } else {
//            // if there is only one network adapter running, use it!
//
//            // TODO: how to support IPv6?
//            unsigned int cnt = 0;
//            unsigned int adapCnt = NetworkInformation::AdapterCount();
//            bool bcastSet = false;
//            for (unsigned int i = 0; i < adapCnt; i++) {
//                const NetworkInformation::Adapter& adap
//                    = NetworkInformation::AdapterInformation(i);
//                if (adap.AddressValidity()
//                        == NetworkInformation::Adapter::VALID) {
//                    if (adap.Address() == vislib::net::IPAddress::ANY) {
//                        // does not make any sense using ANY address
//                        continue;
//                    }
//                    if (cnt == 0) {
//                        addr4 = adap.Address();
//                        if ((!bcastAddr4Valid && !bcastAddr6Valid)
//                                && (adap.BroadcastAddressValidity()
//                                != NetworkInformation::Adapter::NOT_VALID)) {
//                            bcast4 = adap.BroadcastAddress();
//                            bcastSet = true;
//                        }
//                    }
//                    cnt++;
//                }
//            }
//            if (cnt == 1) {
//                adapAddr4Valid = true;
//                if (bcastSet) {
//                    bcastAddr4Valid = true;
//                }
//            }
//
//        }
//
//    } else if (adapAddr6Valid) {
//        // test if the specified adapter really exists
//        // TODO: Implement
//
//    } else if (adapAddr4Valid) {
//        // test if the specified adapter really exists
//        bool adapFound = false;
//        unsigned int adapCnt = NetworkInformation::AdapterCount();
//
//        for (unsigned int i = 0; i < adapCnt; i++) {
//            const NetworkInformation::Adapter& adap
//                = NetworkInformation::AdapterInformation(i);
//
//            if ((adap.AddressValidity() != NetworkInformation::Adapter::VALID)
//                    || (adap.Address() == vislib::net::IPAddress::ANY)) {
//                continue;
//            }
//
//            if (addr4 == adap.Address()) {
//                adapFound = true;
//                if (!bcastAddr4Valid && (adap.BroadcastAddressValidity()
//                        != NetworkInformation::Adapter::NOT_VALID)) {
//                    bcast4 = adap.BroadcastAddress();
//                    bcastAddr4Valid = true;
//                }
//            }
//        }
//
//        if (!adapFound) {
//            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
//                "No IPv4 adapter with address \"%s\" found.",
//                addr4.ToStringA().PeekBuffer());
//            adapAddr4Valid = false;
//        }
//    }
//
//    if (adapAddr4Valid && bcastAddr4Valid) {
//        // all IPv4 addresses are valid, so we can start the cds
//        adapAddr = addr4.ToStringA();
//        bcastAddr = bcast4.ToStringA();
//        ip4Valid = true;
//
//    } else if (adapAddr6Valid && bcastAddr6Valid) {
//        // all IPv6 addresses are valid, so we can start the cds
//        adapAddr = addr6.ToStringA();
//        bcastAddr = bcast6.ToStringA();
//        ip6Valid = true;
//
//    } else {
//        // no, we are not going to run the service
//        adapAddr = "Invalid";
//        bcastAddr = "Invalid";
//        ip4Valid = ip6Valid = false;
//
//    }
//
//    vislib::StringA clustName = T2A(this->clusterName.Param<
//        param::StringParam>()->Value());
//    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "CDS: %s (%s:%d; %s) %s",
//        clustName.PeekBuffer(), adapAddr.PeekBuffer(), iport,
//        bcastAddr.PeekBuffer(),
//        (ip4Valid || ip6Valid) ? "Active" : "Stopped");
//
//    return ip4Valid || ip6Valid;
//}


/*
 * special::ClusterController::registerModule
 */
bool special::ClusterController::registerModule(Call& call) {
    CallRegisterAtController *c = dynamic_cast<CallRegisterAtController *>(&call);
    if ((c == NULL) || (c->Client() == NULL)) return false;
    vislib::sys::AutoLock lock(this->clientsLock);
    if (this->clients.Find(c->Client()) == NULL) {
        this->clients.Add(c->Client());
        if (this->discoveryService.IsRunning()) {
            // inform client that cluster is now available
            c->Client()->OnClusterAvailable();
        }
    }
    return true;
}


/*
 * special::ClusterController::unregisterModule
 */
bool special::ClusterController::unregisterModule(Call& call) {
    CallRegisterAtController *c = dynamic_cast<CallRegisterAtController *>(&call);
    if ((c == NULL) || (c->Client() == NULL)) return false;
    vislib::sys::AutoLock lock(this->clientsLock);
    this->clients.RemoveAll(c->Client());
    return true;
}
