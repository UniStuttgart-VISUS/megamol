/*
 * ClusterController.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/cluster/ClusterController.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/cluster/CallRegisterAtController.h"
#include "mmcore/cluster/ClusterControllerClient.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/Configuration.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/net/DiscoveryService.h"
#include "vislib/SmartPtr.h"
#include "vislib/StringConverter.h"
#include "vislib/Trace.h"
#include "vislib/assert.h"
#include "vislib/net/NetworkInformation.h"
#include "vislib/net/Socket.h"
#include "vislib/sys/AutoLock.h"
#include <climits>

using namespace megamol::core;
using megamol::core::utility::log::Log;
using vislib::net::IPAgnosticAddress;
using vislib::net::IPEndPoint;
using vislib::net::NetworkInformation;
using vislib::net::cluster::DiscoveryService;


/*
 * cluster::ClusterController::DEFAULT_CLUSTERNAME
 */
const char* cluster::ClusterController::DEFAULT_CLUSTERNAME = "MM04RC";


/*
 * cluster::ClusterController::ClusterController
 */
cluster::ClusterController::ClusterController()
        : job::AbstractThreadedJob()
        , Module()
        , vislib::net::cluster::DiscoveryListener()
        , cdsNameSlot("cdsName", "Name of the rendering cluster")
        , cdsPortSlot("cdsPort", "The ip port to be used by the cluster discovery service.")
        , cdsRunSlot("cdsRun", "Start/Stop flag for the cluster discovery")
        , shutdownClusterSlot("shutdownCluster", "Shuts down the whole cluster")
        , discoveryService()
        , registerSlot("register", "Slot to register modules at, which wish to use this controller")
        , clients()
        , clientsLock() {
    vislib::net::Socket::Startup();

#ifdef _DEBUG
    // otherwise i will die!
    vislib::Trace::GetInstance().SetLevel(0);
#endif /* _DEBUG */

    this->discoveryService.AddListener(this);

    this->cdsNameSlot << new param::StringParam(DEFAULT_CLUSTERNAME);
    this->MakeSlotAvailable(&this->cdsNameSlot);

    this->cdsPortSlot << new param::IntParam(this->defaultPort(), 0, USHRT_MAX);
    this->MakeSlotAvailable(&this->cdsPortSlot);

    this->cdsRunSlot << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->cdsRunSlot);
    if (this->cdsRunSlot.Param<param::BoolParam>()->Value()) {
        this->cdsRunSlot.ForceSetDirty();
    }

    this->shutdownClusterSlot << new param::ButtonParam();
    this->shutdownClusterSlot.SetUpdateCallback(&ClusterController::onShutdownCluster);
    this->MakeSlotAvailable(&this->shutdownClusterSlot);

    this->registerSlot.SetCallback(cluster::CallRegisterAtController::ClassName(),
        cluster::CallRegisterAtController::FunctionName(cluster::CallRegisterAtController::CALL_REGISTER),
        &ClusterController::registerModule);
    this->registerSlot.SetCallback(cluster::CallRegisterAtController::ClassName(),
        cluster::CallRegisterAtController::FunctionName(cluster::CallRegisterAtController::CALL_UNREGISTER),
        &ClusterController::unregisterModule);
    this->registerSlot.SetCallback(cluster::CallRegisterAtController::ClassName(),
        cluster::CallRegisterAtController::FunctionName(cluster::CallRegisterAtController::CALL_GETSTATUS),
        &ClusterController::queryStatus);
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
    } catch (...) {}
}


/*
 * cluster::ClusterController::SendUserMsg
 */
void cluster::ClusterController::SendUserMsg(const UINT32 msgType, const BYTE* msgBody, const SIZE_T msgSize) {
    using megamol::core::utility::log::Log;
    try {
        UINT rv = this->discoveryService.SendUserMessage(msgType, msgBody, msgSize);
        if (rv != 0) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Failed to send user message %u: failed after %u communication trails\n", msgType, rv);
        }
    } catch (vislib::Exception ex) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Failed to send user message %u: %s\n", msgType, ex.GetMsgA());
    } catch (...) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Failed to send user message %u: unknown exception\n", msgType);
    }
}


/*
 * cluster::ClusterController::SendUserMsg
 */
void cluster::ClusterController::SendUserMsg(const cluster::ClusterController::PeerHandle& hPeer, const UINT32 msgType,
    const BYTE* msgBody, const SIZE_T msgSize) {
    using megamol::core::utility::log::Log;
    try {
        UINT rv = this->discoveryService.SendUserMessage(hPeer, msgType, msgBody, msgSize);
        if (rv != 0) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Failed to send user message %u: failed after %u communication trails\n", msgType, rv);
        }
    } catch (vislib::Exception ex) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Failed to send user message %u: %s\n", msgType, ex.GetMsgA());
    } catch (...) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Failed to send user message %u: unknown exception\n", msgType);
    }
}


/*
 * cluster::ClusterController::create
 */
bool cluster::ClusterController::create(void) {

    const utility::Configuration& cfg = this->GetCoreInstance()->Configuration();

    if (cfg.IsConfigValueSet("cdsname")) {
        auto cdsname_wstr = std::wstring(cfg.ConfigValue("cdsname").PeekBuffer());
        std::string cdsname_str;
        cdsname_str.resize(cdsname_wstr.length());
        std::wcstombs(cdsname_str.data(), cdsname_wstr.data(), cdsname_wstr.length());
        this->cdsNameSlot.Param<param::StringParam>()->SetValue(cdsname_str);
    }
    if (cfg.IsConfigValueSet("cdsaddress")) { // for legacy configuration
        try {
            vislib::StringW port = cfg.ConfigValue("cdsaddress");
            vislib::StringW::Size pos = port.FindLast(L':');
            if (pos == vislib::StringW::INVALID_POS) {
                throw vislib::Exception(
                    "cdsaddress configuration value does not seem to contain a port", __FILE__, __LINE__);
            }
            this->cdsPortSlot.Param<param::IntParam>()->SetValue(
                vislib::CharTraitsW::ParseInt(port.Substring(pos + 1)));
        } catch (...) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Unable to parse configuration value \"cdsaddress\" as port int. "
                                                      "Configuration value ignored.");
        }
    }
    if (cfg.IsConfigValueSet("cdsport")) {
        try {
            this->cdsPortSlot.Param<param::IntParam>()->SetValue(
                vislib::CharTraitsW::ParseInt(cfg.ConfigValue("cdsport")));
        } catch (...) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Unable to parse configuration value \"cdsport\" as int. "
                                                      "Configuration value ignored.");
        }
    }
    if (cfg.IsConfigValueSet("cdsrun")) {
        try {
            this->cdsRunSlot.Param<param::BoolParam>()->SetValue(
                vislib::CharTraitsW::ParseBool(cfg.ConfigValue("cdsrun")));
        } catch (...) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Unable to parse configuration value \"cdsrun\" as boolean. "
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
    vislib::SingleLinkedList<ClusterControllerClient*>::Iterator iter = this->clients.GetIterator();
    while (iter.HasNext()) {
        ClusterControllerClient* c = iter.Next();
        if (c == NULL)
            continue;
        c->onClusterUnavailable();
        c->ctrlr = NULL; // implicitly disconnect clients
    }
    this->clients.Clear();
    this->clientsLock.Unlock();
}


/*
 * cluster::ClusterController::Run
 */
DWORD cluster::ClusterController::Run(void* userData) {
    const unsigned int sleepTime = 250;

    while (!this->shouldTerminate()) {

        // update cluster discovery settings
        if (this->cdsNameSlot.IsDirty() || this->cdsPortSlot.IsDirty() || this->cdsRunSlot.IsDirty()) {
            // stop current cluster discovery service
            this->stopDiscoveryService();

            this->clientsLock.Lock();
            vislib::SingleLinkedList<ClusterControllerClient*>::Iterator iter = this->clients.GetIterator();
            while (iter.HasNext()) {
                ClusterControllerClient* c = iter.Next();
                if (c == NULL)
                    continue;
                c->onClusterUnavailable();
            }
            this->clientsLock.Unlock();

            // reset dirty flags of settings
            this->cdsNameSlot.ResetDirty();
            this->cdsPortSlot.ResetDirty();
            this->cdsRunSlot.ResetDirty();

            bool run = this->cdsRunSlot.Param<param::BoolParam>()->Value();
            if (run) {
                try {
                    vislib::StringA name(this->cdsNameSlot.Param<param::StringParam>()->Value().c_str());
                    unsigned short port =
                        static_cast<unsigned short>(this->cdsPortSlot.Param<param::IntParam>()->Value());
                    vislib::SmartPtr<DiscoveryService::DiscoveryConfig> cfg;

                    { // choose first ethernet adapter in UP state
                        NetworkInformation::AdapterList adapters;
                        NetworkInformation::GetAdaptersForType(adapters, NetworkInformation::Adapter::TYPE_ETHERNET);
                        while (adapters.Count() > 0) {
                            if (adapters[0].GetStatus() != NetworkInformation::Adapter::OPERSTATUS_UP) {
                                adapters.RemoveFirst();
                            } else
                                break;
                        }
                        if (adapters.Count() > 0) {
                            NetworkInformation::UnicastAddressList ual = adapters[0].GetUnicastAddresses();
                            for (SIZE_T i = 0; ual.Count(); i++) {
                                if (ual[i].GetAddressFamily() == IPAgnosticAddress::FAMILY_INET) {
                                    cfg = new DiscoveryService::DiscoveryConfig(
                                        IPEndPoint(ual[i].GetAddress(), port), port);
                                    break;
                                }
                            }
                            if (cfg.IsNull() && (ual.Count() > 0)) {
                                cfg =
                                    new DiscoveryService::DiscoveryConfig(IPEndPoint(ual[0].GetAddress(), port), port);
                            }
                        }
                    }

                    if (cfg.IsNull()) {
                        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to choose identifing network adapter.\n");
                        this->cdsRunSlot.Param<param::BoolParam>()->SetValue(false, false);

                    } else {

                        this->discoveryService.Start(name, cfg.operator->(), 1, 0, 0,
                            DiscoveryService::DEFAULT_REQUEST_INTERVAL, DiscoveryService::DEFAULT_REQUEST_INTERVAL / 2,
                            2); // using 2 to ensure that slow systems have enough time

                        // give new theads a chance to start
                        vislib::sys::Thread::Sleep(125);

                        if (this->discoveryService.IsRunning()) {
                            Log::DefaultLog.WriteMsg(
                                Log::LEVEL_INFO, "Cluster \"%s\" discovery service is running.\n", name.PeekBuffer());

                            vislib::sys::AutoLock(this->clientsLock);
                            vislib::SingleLinkedList<ClusterControllerClient*>::Iterator iter =
                                this->clients.GetIterator();
                            while (iter.HasNext()) {
                                ClusterControllerClient* c = iter.Next();
                                if (c == NULL)
                                    continue;
                                c->onClusterAvailable();
                            }

                        } else {
                            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                                "Cluster \"%s\" discovery service is not running after start.\n", name.PeekBuffer());
                            this->cdsRunSlot.Param<param::BoolParam>()->SetValue(false, false);
                        }
                    }

                } catch (vislib::Exception ex) {
                    Log::DefaultLog.WriteMsg(
                        Log::LEVEL_ERROR, "Failed to start cluster discovery service: %s\n", ex.GetMsgA());
                    this->cdsRunSlot.Param<param::BoolParam>()->SetValue(false, false);

                } catch (...) {
                    Log::DefaultLog.WriteMsg(
                        Log::LEVEL_ERROR, "Failed to start cluster discovery service: unknown exception\n");
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
void cluster::ClusterController::OnNodeFound(DiscoveryService& src, const DiscoveryService::PeerHandle& hPeer) throw() {
    Log::DefaultLog.WriteMsg(
        Log::LEVEL_INFO, "Cluster Node found: %s\n", src.GetDiscoveryAddress4(hPeer).ToStringA().PeekBuffer());
    vislib::sys::AutoLock lock(this->clientsLock);
    vislib::SingleLinkedList<ClusterControllerClient*>::Iterator iter = this->clients.GetIterator();
    while (iter.HasNext()) {
        ClusterControllerClient* c = iter.Next();
        if (c == NULL)
            continue;
        c->onNodeFound(hPeer);
    }
}


/*
 * cluster::ClusterController::OnNodeLost
 */
void cluster::ClusterController::OnNodeLost(DiscoveryService& src, const DiscoveryService::PeerHandle& hPeer,
    const DiscoveryListener::NodeLostReason reason) throw() {
    Log::DefaultLog.WriteMsg(
        Log::LEVEL_INFO, "Cluster Node lost: %s\n", src.GetDiscoveryAddress4(hPeer).ToStringA().PeekBuffer());
    vislib::sys::AutoLock lock(this->clientsLock);
    vislib::SingleLinkedList<ClusterControllerClient*>::Iterator iter = this->clients.GetIterator();
    while (iter.HasNext()) {
        ClusterControllerClient* c = iter.Next();
        if (c == NULL)
            continue;
        c->onNodeLost(hPeer);
    }
}


/*
 * cluster::ClusterController::OnUserMessage
 */
void cluster::ClusterController::OnUserMessage(vislib::net::cluster::DiscoveryService& src,
    const vislib::net::cluster::DiscoveryService::PeerHandle& hPeer, const bool isClusterMember, const UINT32 msgType,
    const BYTE* msgBody) throw() {
    try {
        //Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Cluster User Message: from %s\n",
        //    src.GetDiscoveryAddress4(hPeer).ToStringA().PeekBuffer());
        vislib::sys::AutoLock lock(this->clientsLock);
        vislib::SingleLinkedList<ClusterControllerClient*>::Iterator iter = this->clients.GetIterator();
        while (iter.HasNext()) {
            ClusterControllerClient* c = iter.Next();
            if (c == NULL)
                continue;
            c->onUserMsg(hPeer, isClusterMember, msgType, msgBody);
        }
    } catch (vislib::Exception ex) {
        VLTRACE(VISLIB_TRCELVL_ERROR, "Illegal vislib exception in OnUserMessage \"%s\"\n", ex.GetMsgA());
    } catch (...) { VLTRACE(VISLIB_TRCELVL_ERROR, "Illegal exception in OnUserMessage\n"); }
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
    CallRegisterAtController* c = dynamic_cast<CallRegisterAtController*>(&call);
    if ((c == NULL) || (c->Client() == NULL))
        return false;
    vislib::sys::AutoLock lock(this->clientsLock);
    if (this->clients.Find(c->Client()) == NULL) {
        this->clients.Add(c->Client());
        c->Client()->ctrlr = this;
        if (this->discoveryService.IsRunning()) {
            // inform client that cluster is now available
            c->Client()->onClusterAvailable();
        }
    }
    return true;
}


/*
 * cluster::ClusterController::unregisterModule
 */
bool cluster::ClusterController::unregisterModule(Call& call) {
    CallRegisterAtController* c = dynamic_cast<CallRegisterAtController*>(&call);
    if ((c == NULL) || (c->Client() == NULL) || (c->Client()->ctrlr != this))
        return false;
    vislib::sys::AutoLock lock(this->clientsLock);
    c->Client()->ctrlr = NULL;
    this->clients.RemoveAll(c->Client());
    return true;
}


/*
 * cluster::ClusterController::queryStatus
 */
bool cluster::ClusterController::queryStatus(Call& call) {
    CallRegisterAtController* c = dynamic_cast<CallRegisterAtController*>(&call);
    if (c == NULL)
        return false;

    c->SetStatus(this->discoveryService.IsRunning(), static_cast<unsigned int>(this->discoveryService.CountPeers()),
        this->discoveryService.GetName());

    return true;
}


/*
 * cluster::ClusterController::onShutdownCluster
 */
bool cluster::ClusterController::onShutdownCluster(param::ParamSlot& slot) {
    ASSERT(&slot == &this->shutdownClusterSlot);

    for (unsigned int I = 0; I < 4; I++) {
        this->SendUserMsg(ClusterControllerClient::USRMSG_SHUTDOWN, NULL, 0);
        vislib::sys::Thread::Sleep(250);
    }

    return true;
}
