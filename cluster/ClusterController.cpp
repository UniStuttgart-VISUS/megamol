/*
 * ClusterController.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/ClusterController.h"
#include "cluster/CallRegisterAtController.h"
#include "cluster/ClusterControllerClient.h"
#include "CoreInstance.h"
#include "param/BoolParam.h"
#include "param/StringParam.h"
#include "utility/Configuration.h"
#include "vislib/assert.h"
#include "vislib/AutoLock.h"
#include "vislib/DiscoveryService.h"
#include "vislib/Log.h"
#include "vislib/NetworkInformation.h"
#include "vislib/SmartPtr.h"
#include "vislib/Socket.h"
#include "vislib/StringConverter.h"
#include "vislib/SystemInformation.h"

using namespace megamol::core;
using vislib::sys::Log;
using vislib::net::cluster::DiscoveryService;
using vislib::net::NetworkInformation;


/*
 * cluster::ClusterController::DEFAULT_CLUSTERNAME
 */
const char * cluster::ClusterController::DEFAULT_CLUSTERNAME = "MM04RndCluster";


/*
 * cluster::ClusterController::ClusterController
 */
cluster::ClusterController::ClusterController() : job::AbstractJobThread(),
        Module(), vislib::net::cluster::DiscoveryListener(),
        cdsNameSlot("cdsName", "Name of the rendering cluster"),
        cdsAddressSlot("cdsAddress", "The ip end point address including port to be used by the cluster discovery service."),
        cdsRunSlot("cdsRun", "Start/Stop flag for the cluster discovery"),
        discoveryService(),
        registerSlot("register", "Slot to register modules at, which wish to use this controller"),
        clients(), clientsLock() {
    vislib::net::Socket::Startup();

    this->discoveryService.AddListener(this);

    this->cdsNameSlot << new param::StringParam(DEFAULT_CLUSTERNAME);
    this->MakeSlotAvailable(&this->cdsNameSlot);

    vislib::TString addressValue;
    addressValue.Format(_T("%s:%d"), this->defaultAddress().PeekBuffer(), this->defaultPort());
    this->cdsAddressSlot << new param::StringParam(addressValue); // TODO: make a good guess
    this->MakeSlotAvailable(&this->cdsAddressSlot);

    this->cdsRunSlot << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->cdsRunSlot);
    if (this->cdsRunSlot.Param<param::BoolParam>()->Value()) {
        this->cdsRunSlot.ForceSetDirty();
    }

    this->registerSlot.SetCallback("CallRegisterAtController", "register",
        &ClusterController::registerModule);
    this->registerSlot.SetCallback("CallRegisterAtController", "unregister",
        &ClusterController::unregisterModule);
    this->MakeSlotAvailable(&this->registerSlot);
}


/*
 * cluster::ClusterController::~ClusterController
 */
cluster::ClusterController::~ClusterController() {
    this->Release();
    ASSERT(!this->discoveryService.IsRunning());
    ASSERT(this->clients.IsEmpty());
    try {
        vislib::net::Socket::Cleanup();
    } catch(...) {
    }
}


/*
 * cluster::ClusterController::SendUserMsg
 */
void cluster::ClusterController::SendUserMsg(const UINT32 msgType,
        const BYTE *msgBody, const SIZE_T msgSize) {
    using vislib::sys::Log;
    try {
        UINT rv = this->discoveryService.SendUserMessage(
            msgType, msgBody, msgSize);
        if (rv != 0) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Failed to send user message %u: failed after %u communication trails\n",
                msgType, rv);
        }
    } catch (vislib::Exception ex) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Failed to send user message %u: %s\n",
            msgType, ex.GetMsgA());
    } catch (...) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Failed to send user message %u: unknown exception\n",
            msgType);
    }
}


/*
 * cluster::ClusterController::SendUserMsg
 */
void cluster::ClusterController::SendUserMsg(
        const cluster::ClusterController::PeerHandle& hPeer,
        const UINT32 msgType, const BYTE *msgBody, const SIZE_T msgSize) {
    using vislib::sys::Log;
    try {
        UINT rv = this->discoveryService.SendUserMessage(
            hPeer, msgType, msgBody, msgSize);
        if (rv != 0) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Failed to send user message %u: failed after %u communication trails\n",
                msgType, rv);
        }
    } catch (vislib::Exception ex) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Failed to send user message %u: %s\n",
            msgType, ex.GetMsgA());
    } catch (...) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Failed to send user message %u: unknown exception\n",
            msgType);
    }
}


/*
 * cluster::ClusterController::create
 */
bool cluster::ClusterController::create(void) {

    const utility::Configuration& cfg
        = this->GetCoreInstance()->Configuration();

    if (cfg.IsConfigValueSet("cdsname")) {
        this->cdsNameSlot.Param<param::StringParam>()->SetValue(
            cfg.ConfigValue("cdsname").PeekBuffer());
    }
    if (cfg.IsConfigValueSet("cdsaddress")) {
        this->cdsAddressSlot.Param<param::StringParam>()->SetValue(
            cfg.ConfigValue("cdsaddress").PeekBuffer());
    }
    if (cfg.IsConfigValueSet("cdsrun")) {
        try {
            this->cdsRunSlot.Param<param::BoolParam>()->SetValue(
                vislib::CharTraitsW::ParseBool(
                cfg.ConfigValue("cdsrun")));
        } catch(...) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                "Unable to parse configuration value \"cdsrun\" as boolean. "
                "Configuration value ignored.");
        }
    }

    if (this->cdsRunSlot.Param<param::BoolParam>()->Value()) {
        this->cdsRunSlot.ForceSetDirty();
    }

    return true;
}


/*
 * cluster::ClusterController::release
 */
void cluster::ClusterController::release(void) {
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

}


/*
 * cluster::ClusterController::Run
 */
DWORD cluster::ClusterController::Run(void *userData) {
    const unsigned int sleepTime = 250;

    while (!this->shouldTerminate()) {

        // update cluster discovery settings
        if (this->cdsNameSlot.IsDirty() || this->cdsAddressSlot.IsDirty() || this->cdsRunSlot.IsDirty()) {
            // stop current cluster discovery service
            this->stopDiscoveryService();

            this->clientsLock.Lock();
            vislib::SingleLinkedList<ClusterControllerClient *>::Iterator iter
                = this->clients.GetIterator();
            while (iter.HasNext()) {
                ClusterControllerClient *c = iter.Next();
                if (c == NULL) continue;
                c->OnClusterUnavailable();
            }
            this->clientsLock.Unlock();

            // reset dirty flags of settings
            this->cdsNameSlot.ResetDirty();
            this->cdsAddressSlot.ResetDirty();
            this->cdsRunSlot.ResetDirty();

            bool run = this->cdsRunSlot.Param<param::BoolParam>()->Value();
            if (run) {
                try {
                    vislib::StringA name(
                        this->cdsNameSlot.Param<param::StringParam>()->Value());
                    vislib::TString address = this->cdsAddressSlot.Param<param::StringParam>()->Value();

                    vislib::net::IPEndPoint endPoint;
                    float wildness;

                    wildness = NetworkInformation::GuessLocalEndPoint(endPoint, address);

                    if (wildness > 0.8f) {
                        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                            "Guessing ip end point \"%s\" from input \"%s\" with wildness \"%f\". Too wild! Service will not be started.\n",
                            endPoint.ToStringA().PeekBuffer(),
                            vislib::StringA(address).PeekBuffer(),
                            wildness);
                        this->cdsRunSlot.Param<param::BoolParam>()->SetValue(false, false);

                    } else {
                        Log::DefaultLog.WriteMsg((wildness > 0.1f) ? Log::LEVEL_WARN : Log::LEVEL_INFO,
                            "Guessing ip end point \"%s\" from input \"%s\" with wildness \"%f\".\n",
                            endPoint.ToStringA().PeekBuffer(),
                            vislib::StringA(address).PeekBuffer(),
                            wildness);

                        vislib::SmartPtr<DiscoveryService::DiscoveryConfig> cfg;
                        if (endPoint.GetAddressFamily() == vislib::net::IPEndPoint::FAMILY_INET6) {
                            cfg = new DiscoveryService::DiscoveryConfig(
                                endPoint,
                                endPoint.GetIPAddress6(),
                                endPoint.GetPort());
                        } else {
                            cfg = new DiscoveryService::DiscoveryConfig(
                                endPoint,
                                endPoint.GetIPAddress(),
                                endPoint.GetPort());
                        }

                        this->discoveryService.Start(name, cfg.operator->(), 1, 0, 0,
                            DiscoveryService::DEFAULT_REQUEST_INTERVAL,
                            DiscoveryService::DEFAULT_REQUEST_INTERVAL / 2,
                            2); // using 2 to ensure that slow systems have enough time

                        if (this->discoveryService.IsRunning()) {
                            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                                "Cluster \"%s\" discovery service is running.\n",
                                name.PeekBuffer());

                            vislib::sys::AutoLock(this->clientsLock);
                            vislib::SingleLinkedList<ClusterControllerClient *>::Iterator iter
                                = this->clients.GetIterator();
                            while (iter.HasNext()) {
                                ClusterControllerClient *c = iter.Next();
                                if (c == NULL) continue;
                                c->OnClusterAvailable();
                            }

                        } else {
                            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                                "Cluster \"%s\" discovery service is not running after start.\n",
                                name.PeekBuffer());
                            this->cdsRunSlot.Param<param::BoolParam>()->SetValue(false, false);

                        }
                    }

                } catch(vislib::Exception ex) {
                    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "Failed to start cluster discovery service: %s\n",
                        ex.GetMsgA());
                    this->cdsRunSlot.Param<param::BoolParam>()->SetValue(false, false);

                } catch(...) {
                    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "Failed to start cluster discovery service: unknown exception\n");
                    this->cdsRunSlot.Param<param::BoolParam>()->SetValue(false, false);

                }
            }
        }

        vislib::sys::Thread::Sleep(sleepTime);

    }
    this->stopDiscoveryService();

    return 0;
}


/*
 * cluster::ClusterController::OnNodeFound
 */
void cluster::ClusterController::OnNodeFound(DiscoveryService& src,
        const DiscoveryService::PeerHandle& hPeer) throw() {
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Cluster Node found: %s\n",
        src.GetDiscoveryAddress4(hPeer).ToStringA().PeekBuffer());
    vislib::sys::AutoLock lock(this->clientsLock);
    vislib::SingleLinkedList<ClusterControllerClient *>::Iterator iter
        = this->clients.GetIterator();
    while (iter.HasNext()) {
        ClusterControllerClient *c = iter.Next();
        if (c == NULL) continue;
        c->OnNodeFound(hPeer);
    }
}


/*
 * cluster::ClusterController::OnNodeLost
 */
void cluster::ClusterController::OnNodeLost(DiscoveryService& src,
        const DiscoveryService::PeerHandle& hPeer,
        const DiscoveryListener::NodeLostReason reason) throw() {
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Cluster Node lost: %s\n",
        src.GetDiscoveryAddress4(hPeer).ToStringA().PeekBuffer());
    vislib::sys::AutoLock lock(this->clientsLock);
    vislib::SingleLinkedList<ClusterControllerClient *>::Iterator iter
        = this->clients.GetIterator();
    while (iter.HasNext()) {
        ClusterControllerClient *c = iter.Next();
        if (c == NULL) continue;
        c->OnNodeLost(hPeer);
    }
}


/*
 * cluster::ClusterController::OnUserMessage
 */
void cluster::ClusterController::OnUserMessage(DiscoveryService& src,
        const DiscoveryService::PeerHandle& hPeer,
        const UINT32 msgType, const BYTE *msgBody) throw() {
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Cluster User Message: from %s\n",
        src.GetDiscoveryAddress4(hPeer).ToStringA().PeekBuffer());
    vislib::sys::AutoLock lock(this->clientsLock);
    vislib::SingleLinkedList<ClusterControllerClient *>::Iterator iter
        = this->clients.GetIterator();
    while (iter.HasNext()) {
        ClusterControllerClient *c = iter.Next();
        if (c == NULL) continue;
        c->OnUserMsg(hPeer, msgType, msgBody);
    }
}


/*
 * cluster::ClusterController::defaultAddress
 */
vislib::TString cluster::ClusterController::defaultAddress(void) {
    NetworkInformation::AdapterList adapters;
    NetworkInformation::GetAdaptersForType(adapters, NetworkInformation::Adapter::TYPE_ETHERNET);
    while (adapters.Count() > 0) {
        if (adapters[0].GetStatus() != NetworkInformation::Adapter::OPERSTATUS_UP) {
            adapters.RemoveFirst();
        } else break;
    }
    if (adapters.Count() > 0) {
        NetworkInformation::UnicastAddressList ual = adapters[0].GetUnicastAddresses();
        for (SIZE_T i = 0; ual.Count(); i++) {
            if (ual[i].GetAddressFamily() == vislib::net::IPAgnosticAddress::FAMILY_INET) {
                return W2T(ual[i].GetAddress().ToStringW());
            }
        }
        if (ual.Count() > 0) {
            vislib::StringW addressValue(ual[0].GetAddress().ToStringW());
            if (ual[0].GetAddressFamily() == vislib::net::IPAgnosticAddress::FAMILY_INET6) {
                addressValue.Prepend(L"[");
                addressValue.Append(L"]");
                return addressValue;
            }
        }
    }
    vislib::TString addressValue;
    vislib::sys::SystemInformation::ComputerName(addressValue);
    return addressValue;
}


/*
 * cluster::ClusterController::defaultPort
 */
UINT16 cluster::ClusterController::defaultPort(void) {
    return DiscoveryService::DEFAULT_PORT;
}


/*
 * cluster::ClusterController::stopDiscoveryService
 */
void cluster::ClusterController::stopDiscoveryService(void) {
    if (this->discoveryService.IsRunning()) {
        if (this->discoveryService.Stop()) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "CDS stopped");
        } else {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Failed to stop CDS");
        }
    }
}


/*
 * cluster::ClusterController::registerModule
 */
bool cluster::ClusterController::registerModule(Call& call) {
    CallRegisterAtController *c = dynamic_cast<CallRegisterAtController *>(&call);
    if ((c == NULL) || (c->Client() == NULL)) return false;
    vislib::sys::AutoLock lock(this->clientsLock);
    if (this->clients.Find(c->Client()) == NULL) {
        this->clients.Add(c->Client());
        c->Client()->ctrlr = this;
        if (this->discoveryService.IsRunning()) {
            // inform client that cluster is now available
            c->Client()->OnClusterAvailable();
        }
    }
    return true;
}


/*
 * cluster::ClusterController::unregisterModule
 */
bool cluster::ClusterController::unregisterModule(Call& call) {
    CallRegisterAtController *c = dynamic_cast<CallRegisterAtController *>(&call);
    if ((c == NULL) || (c->Client() == NULL) || (c->Client()->ctrlr != this)) return false;
    vislib::sys::AutoLock lock(this->clientsLock);
    c->Client()->ctrlr = NULL;
    this->clients.RemoveAll(c->Client());
    return true;
}
