/*
 * ClusterViewMaster.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/ClusterViewMaster.h"
#include "AbstractNamedObject.h"
#include "AbstractNamedObjectContainer.h"
#include "CallDescriptionManager.h"
#include "CalleeSlot.h"
#include "CoreInstance.h"
#include "cluster/NetMessages.h"
#include "ModuleNamespace.h"
#include "param/ButtonParam.h"
#include "param/StringParam.h"
#include "utility/Configuration.h"
#include "view/AbstractView.h"
#include "view/CallRenderView.h"
#include "vislib/assert.h"
#include "vislib/Log.h"
#include "vislib/RawStorage.h"
#include "vislib/StringTokeniser.h"
#include "vislib/SystemInformation.h"
#include "vislib/mathfunctions.h"
#include "vislib/NetworkInformation.h"

using namespace megamol::core;


/*
 * cluster::ClusterViewMaster::ClusterViewMaster
 */
cluster::ClusterViewMaster::ClusterViewMaster(void) : Module(),
        ClusterControllerClient::Listener(), ccc(), ctrlServer(),
        viewNameSlot("viewname", "The name of the view to be used"),
        viewSlot("view", "The view to be used (this value is set automatically"),
        serverAddressSlot("serverAddress", "The TCP/IP address of the server including the port"),
        serverEndPoint(),
        sanityCheckTimeSlot("sanityCheckTime", "Runs a time sync sanity check on all cluster nodes.") {

    this->ccc.AddListener(this);
    this->MakeSlotAvailable(&this->ccc.RegisterSlot());
    this->ctrlServer.AddListener(this);

    this->viewNameSlot << new param::StringParam("");
    this->viewNameSlot.SetUpdateCallback(&ClusterViewMaster::onViewNameChanged);
    this->MakeSlotAvailable(&this->viewNameSlot);

    this->viewSlot.SetCompatibleCall<view::CallRenderViewDescription>();
    // TODO: this->viewSlot.SetVisibility(false);
    this->MakeSlotAvailable(&this->viewSlot);

    this->serverAddressSlot << new param::StringParam("");
    this->serverAddressSlot.SetUpdateCallback(&ClusterViewMaster::onServerAddressChanged);
    this->MakeSlotAvailable(&this->serverAddressSlot);

    this->sanityCheckTimeSlot << new param::ButtonParam();
    this->sanityCheckTimeSlot.SetUpdateCallback(&ClusterViewMaster::onDoSanityCheckTime);
    this->MakeSlotAvailable(&this->sanityCheckTimeSlot);

    // TODO: Implement

}


/*
 * cluster::ClusterViewMaster::~ClusterViewMaster
 */
cluster::ClusterViewMaster::~ClusterViewMaster(void) {
    this->ccc.RemoveListener(this);
    this->ctrlServer.RemoveListener(this);
    this->Release();

    // TODO: Implement

}


/*
 * cluster::ClusterViewMaster::create
 */
bool cluster::ClusterViewMaster::create(void) {
    this->serverAddressSlot.Param<param::StringParam>()->SetValue(this->defaultServerAddress());

    // TODO: Implement

    return true;
}


/*
 * cluster::ClusterViewMaster::release
 */
void cluster::ClusterViewMaster::release(void) {
    this->ctrlServer.Stop();

    // TODO: Implement

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

    vislib::net::SimpleMessage msg;
    msg.GetHeader().SetMessageID(cluster::netmessages::MSG_REQUEST_RESETUP);
    msg.GetHeader().SetBodySize(0);
    msg.AssertBodySize();
    this->ctrlServer.MultiSendMessage(msg);

    return true;
}


/*
 * cluster::ClusterViewMaster::OnClusterUserMessage
 */
void cluster::ClusterViewMaster::OnClusterUserMessage(cluster::ClusterControllerClient& sender,
        const cluster::ClusterController::PeerHandle& hPeer, bool isClusterMember,
        const UINT32 msgType, const BYTE *msgBody) {

    switch (msgType) {
        case ClusterControllerClient::USRMSG_QUERYHEAD:
            if (isClusterMember && this->ctrlServer.IsRunning()) {
                vislib::StringA address = this->serverEndPoint.ToStringA();
                try {
                    sender.SendUserMsg(ClusterControllerClient::USRMSG_HEADHERE,
                        reinterpret_cast<const BYTE*>(address.PeekBuffer()), address.Length() + 1);
                } catch(...) {
                }
            }
            break;
    }

}


/*
 * cluster::ClusterViewMaster::OnControlChannelMessage
 */
void cluster::ClusterViewMaster::OnControlChannelMessage(cluster::ControlChannelServer& server,
        cluster::CommChannel& channel, const vislib::net::AbstractSimpleMessage& msg) {
    using vislib::sys::Log;
    vislib::net::SimpleMessage outMsg;

    switch (msg.GetHeader().GetMessageID()) {
        case cluster::netmessages::MSG_REQUEST_TIMESYNC:
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Time sync started");
            outMsg.GetHeader().SetMessageID(cluster::netmessages::MSG_PING_TIMESYNC);
            outMsg.GetHeader().SetBodySize(sizeof(cluster::netmessages::TimeSyncData));
            outMsg.AssertBodySize();
            ::memset(outMsg.GetBody(), 0, sizeof(cluster::netmessages::TimeSyncData));
            outMsg.GetBodyAs<cluster::netmessages::TimeSyncData>()->srvrTimes[0] = this->GetCoreInstance()->GetInstanceTime();
            channel.SendMessage(outMsg);
            break;

        case cluster::netmessages::MSG_PING_TIMESYNC:
            ASSERT(msg.GetHeader().GetBodySize() == sizeof(cluster::netmessages::TimeSyncData));
            outMsg.GetHeader().SetMessageID(cluster::netmessages::MSG_PING_TIMESYNC);
            outMsg.GetHeader().SetBodySize(sizeof(cluster::netmessages::TimeSyncData));
            outMsg.AssertBodySize();
            ::memcpy(outMsg.GetBody(), msg.GetBody(), sizeof(cluster::netmessages::TimeSyncData));
            {
                UINT32 &trip = outMsg.GetBodyAs<cluster::netmessages::TimeSyncData>()->trip;
                trip++;
                outMsg.GetBodyAs<cluster::netmessages::TimeSyncData>()->srvrTimes[trip] = this->GetCoreInstance()->GetInstanceTime();
                if (trip == cluster::netmessages::MAX_TIME_SYNC_PING) {
                    outMsg.GetHeader().SetMessageID(cluster::netmessages::MSG_DONE_TIMESYNC);
                }
            }
            channel.SendMessage(outMsg);
            break;

        case cluster::netmessages::MSG_WHATSYOURNAME: {
            vislib::StringA myname;
            vislib::sys::SystemInformation::ComputerName(myname);
            outMsg.GetHeader().SetMessageID(cluster::netmessages::MSG_MYNAMEIS);
            outMsg.GetHeader().SetBodySize(myname.Length() + 1);
            outMsg.AssertBodySize();
            ::memcpy(outMsg.GetBody(), myname.PeekBuffer(), myname.Length() + 1);
            channel.SendMessage(outMsg);
        } break;

        case cluster::netmessages::MSG_MYNAMEIS:
            ASSERT(msg.GetHeader().GetBodySize() > 0);
            channel.SetCounterpartName(msg.GetBodyAs<char>());
            break;

        case cluster::netmessages::MSG_TIME_SANITYCHECK:
            ASSERT(msg.GetHeader().GetBodySize() == sizeof(double));
            {
                double remoteTime = *msg.GetBodyAs<double>();
                double localTime = this->GetCoreInstance()->GetInstanceTime();
                double diff = vislib::math::Abs(localTime - remoteTime);
                UINT level = Log::LEVEL_INFO;
                if (diff > 0.1) level = Log::LEVEL_ERROR;
                else if (diff > 0.03) level = Log::LEVEL_WARN;
                Log::DefaultLog.WriteMsg(level, "%s is off by %f seconds\n", channel.CounterpartName().PeekBuffer(), diff);
            }
            break;

        case cluster::netmessages::MSG_REQUEST_GRAPHSETUP: {
            vislib::RawStorage mem;
            this->LockModuleGraph(false);
            try {
                RootModuleNamespace *root = dynamic_cast<RootModuleNamespace*>(this->RootModule());
                if (root != NULL) {
                    root->SerializeGraph(mem);
                }
            } catch(...) {
                mem.EnforceSize(0);
            }
            this->UnlockModuleGraph();
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Sending module graph setup (%d Bytes)", static_cast<int>(mem.GetSize()));
            outMsg.GetHeader().SetMessageID(cluster::netmessages::MSG_GRAPHSETUP);
            outMsg.GetHeader().SetBodySize(mem.GetSize());
            outMsg.AssertBodySize();
            ::memcpy(outMsg.GetBody(), mem, mem.GetSize());
            channel.SendMessage(outMsg);
        } break;

        case cluster::netmessages::MSG_REQUEST_CAMERASETUP: {
            AbstractNamedObject *ano = NULL;
            Call *call = this->viewSlot.CallAs<Call>();
            if (call != NULL) ano = const_cast<CalleeSlot*>(call->PeekCalleeSlot());
            if (ano != NULL) ano = ano->Parent();
            if (ano == NULL) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Cluster View Master not connected to a view. Lazy evaluation NOT implemented");
                break;
            }
            vislib::StringA viewname = ano->FullName();
            outMsg.GetHeader().SetMessageID(cluster::netmessages::MSG_SET_CLUSTERVIEW);
            outMsg.GetHeader().SetBodySize(viewname.Length() + 1);
            outMsg.AssertBodySize();
            ::memcpy(outMsg.GetBody(), viewname.PeekBuffer(), viewname.Length() + 1);
            channel.SendMessage(outMsg);
            // TODO: Implement

        } break;

        default:
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Unhandled message received: %u\n",
                static_cast<unsigned int>(msg.GetHeader().GetMessageID()));
            break;
    }

}


/*
 * cluster::ClusterViewMaster::defaultServerHost
 */
vislib::TString cluster::ClusterViewMaster::defaultServerHost(void) const {
    using vislib::net::NetworkInformation;
    const utility::Configuration& cfg = this->GetCoreInstance()->Configuration();
    if (cfg.IsConfigValueSet("cmvhost")) {
        return cfg.ConfigValue("cmvhost");
    }

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
                return ual[i].GetAddress()
#if defined(UNICODE) || defined(_UNICODE)
                    .ToStringW();
#else /* defined(UNICODE) || defined(_UNICODE) */
                    .ToStringA();
#endif /* defined(UNICODE) || defined(_UNICODE) */
            }
        }
        if (ual.Count() > 0) {
            vislib::TString host(ual[0].GetAddress()
#if defined(UNICODE) || defined(_UNICODE)
                .ToStringW());
#else /* defined(UNICODE) || defined(_UNICODE) */
                .ToStringA());
#endif /* defined(UNICODE) || defined(_UNICODE) */
            if (ual[0].GetAddressFamily() == vislib::net::IPAgnosticAddress::FAMILY_INET6) {
                host.Prepend(_T("["));
                host.Append(_T("]"));
            }
            return host;
        }
    }

#if defined(UNICODE) || defined(_UNICODE)
    return vislib::sys::SystemInformation::ComputerNameW();
#else /* defined(UNICODE) || defined(_UNICODE) */
    return vislib::sys::SystemInformation::ComputerNameA();
#endif /* defined(UNICODE) || defined(_UNICODE) */
}


/*
 * cluster::ClusterViewMaster::defaultServerPort
 */
unsigned short cluster::ClusterViewMaster::defaultServerPort(void) const {
    const utility::Configuration& cfg = this->GetCoreInstance()->Configuration();
    try {
        if (cfg.IsConfigValueSet("cmvport")) {
            return static_cast<unsigned short>(vislib::CharTraitsW::ParseInt(
                cfg.ConfigValue("cmvport")));
        }
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
            "Unable to parse configuration value \"cmvport\" as int. Configuration value ignored.");
    }
    return 17126;
}


/*
 * cluster::ClusterViewMaster::defaultServerAddress
 */
vislib::TString cluster::ClusterViewMaster::defaultServerAddress(void) const {
    const utility::Configuration& cfg
        = this->GetCoreInstance()->Configuration();

    if (cfg.IsConfigValueSet("cmvaddress")) { // host and port
        return cfg.ConfigValue("cmvaddress");
    }

    vislib::TString address;
    address.Format(_T("%s:%u"), this->defaultServerHost().PeekBuffer(), this->defaultServerPort());
    return address;
}


/*
 * cluster::ClusterViewMaster::onServerAddressChanged
 */
bool cluster::ClusterViewMaster::onServerAddressChanged(param::ParamSlot& slot) {
    ASSERT(&slot == &this->serverAddressSlot);
    vislib::StringA address(this->serverAddressSlot.Param<param::StringParam>()->Value());

    if (address.IsEmpty()) {
        if (this->ctrlServer.IsRunning()) {
            this->ctrlServer.Stop();
        }
        return true;
    }

    vislib::net::IPEndPoint ep;
    float wildness = vislib::net::NetworkInformation::GuessLocalEndPoint(ep, address);

    if (ep == this->serverEndPoint) return true;
    this->serverEndPoint = ep;

    if (this->ctrlServer.IsRunning()) {
        this->ctrlServer.Stop();
    }

    if (wildness > 0.8) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Guessed server end point \"%s\" from \"%s\" with too high wildness: %f\n",
            ep.ToStringA().PeekBuffer(), address.PeekBuffer(), wildness);
        return true;
    }

    vislib::sys::Log::DefaultLog.WriteMsg((wildness > 0.3) ? vislib::sys::Log::LEVEL_WARN : vislib::sys::Log::LEVEL_INFO,
        "Starting server on \"%s\" guessed from \"%s\" with wildness: %f\n",
        ep.ToStringA().PeekBuffer(), address.PeekBuffer(), wildness);

    this->ctrlServer.Start(this->serverEndPoint);

    return true;
}


/*
 * cluster::ClusterViewMaster::onDoSanityCheckTime
 */
bool cluster::ClusterViewMaster::onDoSanityCheckTime(param::ParamSlot& slot) {
    ASSERT(&slot == &this->sanityCheckTimeSlot);
    vislib::net::SimpleMessage msg;
    msg.GetHeader().SetMessageID(cluster::netmessages::MSG_TIME_SANITYCHECK);
    msg.GetHeader().SetBodySize(0);
    msg.AssertBodySize();
    this->ctrlServer.MultiSendMessage(msg);
    return true;
}
