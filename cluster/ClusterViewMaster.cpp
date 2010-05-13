/*
 * ClusterViewMaster.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ClusterViewMaster.h"
#include "AbstractNamedObject.h"
#include "AbstractNamedObjectContainer.h"
#include "CallDescriptionManager.h"
#include "CalleeSlot.h"
#include "CoreInstance.h"
#include "ModuleNamespace.h"
#include "param/StringParam.h"
#include "utility/Configuration.h"
#include "view/AbstractView.h"
#include "view/CallRenderView.h"
#include "vislib/Log.h"
#include "vislib/StringTokeniser.h"
#include "vislib/SystemInformation.h"
#include "vislib/NetworkInformation.h"

using namespace megamol::core;


/*
 * cluster::ClusterViewMaster::ClusterViewMaster
 */
cluster::ClusterViewMaster::ClusterViewMaster(void) : Module(),
        cluster::ClusterControllerClient(),
        viewNameSlot("viewname", "The name of the view to be used"),
        viewSlot("view", "The view to be used (this value is set automatically"),
        commCtrlServer(), commCtrlServerShutdown(false) {

    this->MakeSlotAvailable(&this->registerSlot);
    this->MakeSlotAvailable(&this->ctrlCommAddressSlot);

    this->viewNameSlot << new param::StringParam("");
    this->viewNameSlot.SetUpdateCallback(&ClusterViewMaster::onViewNameChanged);
    this->MakeSlotAvailable(&this->viewNameSlot);

    this->viewSlot.SetCompatibleCall<view::CallRenderViewDescription>();
    // TODO: this->viewSlot.SetVisibility(false);
    this->MakeSlotAvailable(&this->viewSlot);

    this->commCtrlServer.AddListener(this);

    // TODO: Implement

}


/*
 * cluster::ClusterViewMaster::~ClusterViewMaster
 */
cluster::ClusterViewMaster::~ClusterViewMaster(void) {
    this->Release();

    // TODO: Implement

}


/*
 * cluster::ClusterViewMaster::create
 */
bool cluster::ClusterViewMaster::create(void) {
    this->ctrlCommAddressSlot.Param<param::StringParam>()->SetValue(
        this->defaultServerAddress());

    // TODO: Implement

    return true;
}


/*
 * cluster::ClusterViewMaster::release
 */
void cluster::ClusterViewMaster::release(void) {
    if (this->commCtrlServer.IsRunning()) {
        this->commCtrlServerShutdown = true;
        this->commCtrlServer.Terminate(false);
        this->commCtrlServerShutdown = false;
    }
    this->stopCtrlComm();

    // TODO: Implement

}


/*
 * cluster::ClusterViewMaster::OnUserMsg
 */
void cluster::ClusterViewMaster::OnUserMsg(
        const cluster::ClusterController::PeerHandle& hPeer,
        const UINT32 msgType, const BYTE *msgBody) {
    using vislib::sys::Log;

    switch (msgType) {
        case ClusterControllerClient::USRMSG_QUERYHEAD:
            try {
                if (this->commCtrlServer.IsRunning()) {
                    vislib::StringA address(this->commCtrlServer.GetBindAddressA());
                    this->SendUserMsg(hPeer, ClusterControllerClient::USRMSG_HEADHERE,
                        reinterpret_cast<const BYTE *>(address.PeekBuffer()), address.Length() + 1);
                }
            } catch(vislib::Exception ex) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unable to send answer to USRMSG_QUERYHEAD: %s\n", ex.GetMsgA());
            } catch(...) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unable to send answer to USRMSG_QUERYHEAD: unexpected exception\n");
            }
            break;
    }

}


/*
 * cluster::ClusterViewMaster::OnMessageReceived
 */
bool cluster::ClusterViewMaster::OnMessageReceived(
        vislib::net::SimpleMessageDispatcher& src,
        const vislib::net::AbstractSimpleMessage& msg) throw() {

    //if (this->setupState != SETUPSTATE_CONNECTED) {
    //    vislib::sys::AutoLock(this->setupStateLock);
    //    this->setupState = SETUPSTATE_CONNECTED;
    //}

    //TODO: Implement

    return true;
}


/*
 * cluster::ClusterViewMaster::onViewNameChanged
 */
bool cluster::ClusterViewMaster::onViewNameChanged(param::ParamSlot& slot) {
    using vislib::sys::Log;
    if (!this->viewSlot.ConnectCall(NULL)) { // disconnect old call
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
            "Unable to disconnect call from slot \"%s\"\n",
            this->viewSlot.FullName().PeekBuffer());
    }

    CalleeSlot *viewModSlot = NULL;
    vislib::StringA viewName(this->viewNameSlot.Param<param::StringParam>()->Value());
    if (viewName.IsEmpty()) {
        // user just wanted to disconnect
        return true;
    }

    this->LockModuleGraph(false);
    AbstractNamedObject *ano = this->FindNamedObject(viewName);
    view::AbstractView *av = dynamic_cast<view::AbstractView*>(ano);
    if (av == NULL) {
        ModuleNamespace *mn = dynamic_cast<ModuleNamespace*>(ano);
        if (mn != NULL) {
            view::AbstractView *av2;
            AbstractNamedObjectContainer::ChildList::Iterator ci = mn->GetChildIterator();
            while (ci.HasNext()) {
                ano = ci.Next();
                av2 = dynamic_cast<view::AbstractView*>(ano);
                if (av2 != NULL) {
                    if (av != NULL) {
                        av = NULL;
                        break; // too many views
                    } else {
                        av = av2; // if only one view present in children, use it
                    }
                }
            }
        }
    }
    if (av != NULL) {
        viewModSlot = dynamic_cast<CalleeSlot*>(av->FindSlot("render"));
    }
    this->UnlockModuleGraph();

    if (viewModSlot == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "View \"%s\" not found\n",
            viewName.PeekBuffer());
        return true; // this is just for diryt flag reset
    }

    CallDescription *cd = CallDescriptionManager::Instance()
        ->Find(view::CallRenderView::ClassName());
    if (cd == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Cannot find description for call \"%s\"\n",
            view::CallRenderView::ClassName());
        return true; // this is just for diryt flag reset
    }

    Call *c = cd->CreateCall();
    if (c == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Cannot create call \"%s\"\n",
            view::CallRenderView::ClassName());
        return true; // this is just for diryt flag reset
    }

    if (!viewModSlot->ConnectCall(c)) {
        delete c;
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Cannot connect call \"%s\" to inbound-slot \"%s\"\n",
            view::CallRenderView::ClassName(),
            viewModSlot->FullName().PeekBuffer());
        return true; // this is just for diryt flag reset
    }

    if (!this->viewSlot.ConnectCall(c)) {
        delete c;
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Cannot connect call \"%s\" to outbound-slot \"%s\"\n",
            view::CallRenderView::ClassName(),
            this->viewSlot.FullName().PeekBuffer());
        return true; // this is just for diryt flag reset
    }

    // TODO: Implement

    return true;
}


/*
 * cluster::ClusterViewMaster::OnCtrlCommAddressChanged
 */
void cluster::ClusterViewMaster::OnCtrlCommAddressChanged(const vislib::TString& address) {
    if (this->commCtrlServer.IsRunning()) {
        this->commCtrlServerShutdown = true;
        this->commCtrlServer.Terminate(false);
        this->commCtrlServerShutdown = false;
    }
    this->stopCtrlComm();

    vislib::net::IPEndPoint ep;
    float wildness = vislib::net::NetworkInformation::GuessLocalEndPoint(ep, address);

    if (wildness > 0.8f) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Wildness for server end-point \"%s\" from input \"%s\" is too large: %f > 0.8\n",
            ep.ToStringA().PeekBuffer(), vislib::StringA(address).PeekBuffer(), wildness);
        return;
    }

    vislib::SmartRef<vislib::net::TcpCommChannel> c = this->startCtrlCommServer();

    if (!c.IsNull()) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            "Server started at end-point \"%s\" guessed from \"%s\" with wildness %f\n",
            ep.ToStringA().PeekBuffer(), vislib::StringA(address).PeekBuffer(), wildness);
    } else {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Failed to start server at end-point \"%s\" guessed from \"%s\" with wildness %f\n",
            ep.ToStringA().PeekBuffer(), vislib::StringA(address).PeekBuffer(), wildness);
        return;
    }

    this->commCtrlServer.Configure(c.operator->(), ep.ToStringA());
    this->commCtrlServer.Start(NULL);

}


/*
 * cluster::ClusterViewMaster::OnServerError
 */
bool cluster::ClusterViewMaster::OnServerError(const vislib::net::CommServer& src,
        const vislib::Exception& exception) throw() {
    // Downgrading Errors to Infos on shutdown
    vislib::sys::Log::DefaultLog.WriteMsg(
        this->commCtrlServerShutdown ? (vislib::sys::Log::LEVEL_INFO + 50) : vislib::sys::Log::LEVEL_ERROR,
        "Server-Error: %s\n", exception.GetMsgA());
    return true;
}


/*
 * cluster::ClusterViewMaster::OnNewConnection
 */
bool cluster::ClusterViewMaster::OnNewConnection(const vislib::net::CommServer& src,
        vislib::SmartRef<vislib::net::AbstractCommChannel> channel) throw() {
    try {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            "Incoming connection from %s\n", "CURRENTLY UNKNOWN");

        vislib::net::SimpleMessage sm;
        sm.GetHeader().SetMessageID(15);
        sm.GetHeader().SetBodySize(0);
        channel.DynamicCast<vislib::net::AbstractOutboundCommChannel>()->Send(static_cast<void*>(sm), sm.GetMessageSize(), 0, true);
        // TODO: Implement and return true on taking of ownershipf of channel
    } catch(...) {
    }
    return false;
}


/*
 * cluster::ClusterViewMaster::OnServerExited
 */
void cluster::ClusterViewMaster::OnServerExited(const vislib::net::CommServer& src) throw() {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Server exited\n");
}


/*
 * cluster::ClusterViewMaster::OnServerStarted
 */
void cluster::ClusterViewMaster::OnServerStarted(const vislib::net::CommServer& src) throw() {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Server started\n");
}


/*
 * cluster::ClusterViewMaster::defaultPort
 */
UINT16 cluster::ClusterViewMaster::defaultPort(void) const {
    return 17126;
}


/*
 * cluster::ClusterViewMaster::defaultServerAddress
 */
vislib::TString cluster::ClusterViewMaster::defaultServerAddress(void) const {
    using vislib::sys::Log;
    using vislib::net::NetworkInformation;
    const utility::Configuration& cfg
        = this->GetCoreInstance()->Configuration();
    vislib::TString host;
    unsigned short port = this->defaultPort();

    if (cfg.IsConfigValueSet("cmvaddress")) { // host and port
        return cfg.ConfigValue("cmvaddress");
    }
    if (cfg.IsConfigValueSet("cmvhost")) {
        host = cfg.ConfigValue("cmvhost");
    }
    if (cfg.IsConfigValueSet("cmvport")) {
        try {
            port = vislib::CharTraitsW::ParseInt(cfg.ConfigValue("cmvport"));
        } catch(...) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                "Unable to parse configuration value \"cmvport\" as int. "
                "Configuration value ignored.");
        }
    }

    if (host.IsEmpty()) {
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
                    host = ual[i].GetAddress()
#if defined(UNICODE) || defined(_UNICODE)
                        .ToStringW();
#else /* defined(UNICODE) || defined(_UNICODE) */
                        .ToStringA();
#endif /* defined(UNICODE) || defined(_UNICODE) */
                    break;
                }
            }
            if (host.IsEmpty() && (ual.Count() > 0)) {
                    host = ual[0].GetAddress()
#if defined(UNICODE) || defined(_UNICODE)
                        .ToStringW();
#else /* defined(UNICODE) || defined(_UNICODE) */
                        .ToStringA();
#endif /* defined(UNICODE) || defined(_UNICODE) */
                if (ual[0].GetAddressFamily() == vislib::net::IPAgnosticAddress::FAMILY_INET6) {
                    host.Prepend(L"[");
                    host.Append(L"]");
                }
            }
        }
    }
    if (host.IsEmpty()) {
        vislib::sys::SystemInformation::ComputerName(host);
    }

    vislib::TString address;
    address.Format(_T("%s:%u"), host.PeekBuffer(), port);
    return address;
}
