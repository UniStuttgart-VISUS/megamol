/*
 * ClusterViewMaster.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/cluster/ClusterViewMaster.h"
#include "mmcore/AbstractNamedObject.h"
#include "mmcore/AbstractNamedObjectContainer.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/cluster/NetMessages.h"
#include "mmcore/ModuleNamespace.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/Configuration.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/CallRenderView.h"
#include "vislib/assert.h"
#include "vislib/sys/Log.h"
#include "vislib/RawStorage.h"
#include "vislib/RawStorageSerialiser.h"
#include "vislib/net/ShallowSimpleMessage.h"
#include "vislib/net/SimpleMessageHeaderData.h"
#include "vislib/StringTokeniser.h"
#include "vislib/sys/SystemInformation.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/net/NetworkInformation.h"
#include "vislib/UTF8Encoder.h"

using namespace megamol::core;


/*
 * cluster::ClusterViewMaster::ClusterViewMaster
 */
cluster::ClusterViewMaster::ClusterViewMaster(void) : Module(),
        ClusterControllerClient::Listener(), CommChannelServer::Listener(),
        param::ParamUpdateListener(), ccc(), ctrlServer(),
        viewNameSlot("viewname", "The name of the view to be used"),
        viewSlot("view", "The view to be used (this value is set automatically"),
        serverAddressSlot("serverAddress", "The TCP/IP address of the server including the port"),
        serverEndPoint(),
        sanityCheckTimeSlot("RemoteView::sanityCheckTime", "Runs a time sync sanity check on all cluster nodes."),
        camUpdateThread(&ClusterViewMaster::cameraUpdateThread),
        pauseRemoteViewSlot("RemoteView::Pause", "Enters remote view pause mode"),
        resumeRemoteViewSlot("RemoteView::Resume", "Resumes from remote view pause mode"),
        forceNetVSyncOnSlot("RemoteView::NetVSyncOn", "Forces network v-sync on"),
        forceNetVSyncOffSlot("RemoteView::NetVSyncOff", "Forces network v-sync off") {

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

    this->pauseRemoteViewSlot << new param::ButtonParam();
    this->pauseRemoteViewSlot.SetUpdateCallback(&ClusterViewMaster::onPauseRemoteViewClicked);
    this->MakeSlotAvailable(&this->pauseRemoteViewSlot);

    this->resumeRemoteViewSlot << new param::ButtonParam();
    this->resumeRemoteViewSlot.SetUpdateCallback(&ClusterViewMaster::onResumeRemoteViewClicked);
    this->MakeSlotAvailable(&this->resumeRemoteViewSlot);

    this->forceNetVSyncOnSlot << new param::ButtonParam();
    this->forceNetVSyncOnSlot.SetUpdateCallback(&ClusterViewMaster::onForceNetVSyncOnClicked);
    this->MakeSlotAvailable(&this->forceNetVSyncOnSlot);

    this->forceNetVSyncOffSlot << new param::ButtonParam();
    this->forceNetVSyncOffSlot.SetUpdateCallback(&ClusterViewMaster::onForceNetVSyncOffClicked);
    this->MakeSlotAvailable(&this->forceNetVSyncOffSlot);

    // TODO: Implement

}


/*
 * cluster::ClusterViewMaster::~ClusterViewMaster
 */
cluster::ClusterViewMaster::~ClusterViewMaster(void) {
    this->ccc.RemoveListener(this);
    this->ctrlServer.RemoveListener(this);
    if (this->camUpdateThread.IsRunning()) {
        this->ModuleGraphLock().LockExclusive();
        this->viewSlot.ConnectCall(NULL);
        this->ModuleGraphLock().UnlockExclusive();
        this->camUpdateThread.Join();
    }
    this->Release();

    // TODO: Implement

}


/*
 * cluster::ClusterViewMaster::create
 */
bool cluster::ClusterViewMaster::create(void) {
    this->serverAddressSlot.Param<param::StringParam>()->SetValue(this->defaultServerAddress());
    this->GetCoreInstance()->RegisterParamUpdateListener(this);

    // TODO: Implement

    return true;
}


/*
 * cluster::ClusterViewMaster::release
 */
void cluster::ClusterViewMaster::release(void) {
    this->GetCoreInstance()->UnregisterParamUpdateListener(this);
    this->ctrlServer.Stop();
    if (this->camUpdateThread.IsRunning()) {
        this->ModuleGraphLock().LockExclusive();
        this->viewSlot.ConnectCall(NULL);
        this->ModuleGraphLock().UnlockExclusive();
        this->camUpdateThread.Join();
    }

    // TODO: Implement

}


/*
 * cluster::ClusterViewMaster::onViewNameChanged
 */
bool cluster::ClusterViewMaster::onViewNameChanged(param::ParamSlot& slot) {
    using vislib::sys::Log;
    this->ModuleGraphLock().LockExclusive();
    if (!this->viewSlot.ConnectCall(NULL)) { // disconnect old call
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
            "Unable to disconnect call from slot \"%s\"\n",
            this->viewSlot.FullName().PeekBuffer());
    }
    this->ModuleGraphLock().UnlockExclusive();
    if (this->camUpdateThread.IsRunning()) {
        this->camUpdateThread.Join();
    }

    CalleeSlot *viewModSlot = NULL;
    vislib::StringA viewName(this->viewNameSlot.Param<param::StringParam>()->Value());
    if (viewName.IsEmpty()) {
        // user just wanted to disconnect
        return true;
    }

    this->ModuleGraphLock().LockExclusive();
    AbstractNamedObject::ptr_type ano = this->FindNamedObject(viewName);
    view::AbstractView *av = dynamic_cast<view::AbstractView*>(ano.get());
    if (av == NULL) {
        ModuleNamespace *mn = dynamic_cast<ModuleNamespace*>(ano.get());
        if (mn != NULL) {
            view::AbstractView *av2;
            child_list_type::iterator ci, cie;
            ci = mn->ChildList_Begin();
            cie = mn->ChildList_End();
            for (; ci != cie; ++ci) {
                av2 = dynamic_cast<view::AbstractView*>((*ci).get());
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
    this->ModuleGraphLock().UnlockExclusive();

    if (viewModSlot == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "View \"%s\" not found\n",
            viewName.PeekBuffer());
        return true; // this is just for diryt flag reset
    }

    factories::CallDescription::ptr cd = this->GetCoreInstance()->GetCallDescriptionManager().Find(view::CallRenderView::ClassName());
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

    this->camUpdateThread.Start(static_cast<void*>(this));

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
 * cluster::ClusterViewMaster::OnCommChannelMessage
 */
void cluster::ClusterViewMaster::OnCommChannelMessage(cluster::CommChannelServer& server,
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
            outMsg.GetBodyAs<cluster::netmessages::TimeSyncData>()->srvrTimes[0] = this->GetCoreInstance()->GetCoreInstanceTime();
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
                outMsg.GetBodyAs<cluster::netmessages::TimeSyncData>()->srvrTimes[trip] = this->GetCoreInstance()->GetCoreInstanceTime();
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
                double localTime = this->GetCoreInstance()->GetCoreInstanceTime();
                double diff = vislib::math::Abs(localTime - remoteTime);
                UINT level = Log::LEVEL_INFO;
                if (diff > 0.1) level = Log::LEVEL_ERROR;
                else if (diff > 0.03) level = Log::LEVEL_WARN;
                Log::DefaultLog.WriteMsg(level, "%s is off by %f seconds\n", channel.CounterpartName().PeekBuffer(), diff);
            }
            break;

        case cluster::netmessages::MSG_REQUEST_GRAPHSETUP: {
            vislib::RawStorage mem;
            this->ModuleGraphLock().LockExclusive();
            try {
                AbstractNamedObject::ptr_type root_ptr = this->RootModule();
                RootModuleNamespace *root = dynamic_cast<RootModuleNamespace*>(root_ptr.get());
                if (root != NULL) {
                    root->SerializeGraph(mem);
                }
            } catch(...) {
                mem.EnforceSize(0);
            }
            this->ModuleGraphLock().UnlockExclusive();
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Sending module graph setup (%d Bytes)", static_cast<int>(mem.GetSize()));
            outMsg.GetHeader().SetMessageID(cluster::netmessages::MSG_GRAPHSETUP);
            outMsg.GetHeader().SetBodySize(static_cast<vislib::net::SimpleMessageSize>(mem.GetSize()));
            outMsg.AssertBodySize();
            ::memcpy(outMsg.GetBody(), mem, mem.GetSize());
            channel.SendMessage(outMsg);
        } break;

        case cluster::netmessages::MSG_REQUEST_CAMERASETUP: {
            AbstractNamedObject::ptr_type ano;
            Call *call = this->viewSlot.CallAs<Call>();
            if (call != NULL) ano = AbstractNamedObject::ptr_type(const_cast<CalleeSlot*>(call->PeekCalleeSlot()));
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
        } break;

        case cluster::netmessages::MSG_REQUEST_CAMERAVALUES: {
            Call *call = this->viewSlot.CallAs<Call>();
            this->ModuleGraphLock().LockExclusive();
            AbstractNamedObject::const_ptr_type avp;
            const view::AbstractView *av = NULL;
            if ((call != NULL) && (call->PeekCalleeSlot() != NULL)) {
                avp = call->PeekCalleeSlot()->Parent();
                av = dynamic_cast<const view::AbstractView*>(avp.get());
            }
            this->ModuleGraphLock().UnlockExclusive();
            if (av != NULL) {
                vislib::RawStorage mem;
                mem.AssertSize(sizeof(vislib::net::SimpleMessageHeaderData));
                vislib::RawStorageSerialiser serialiser(&mem, sizeof(vislib::net::SimpleMessageHeaderData));
                vislib::net::ShallowSimpleMessage cmsg(mem);
                serialiser.SetOffset(sizeof(vislib::net::SimpleMessageHeaderData));
                av->SerialiseCamera(serialiser);

                cmsg.SetStorage(mem, mem.GetSize());
                cmsg.GetHeader().SetMessageID(cluster::netmessages::MSG_SET_CAMERAVALUES);
                cmsg.GetHeader().SetBodySize(static_cast<vislib::net::SimpleMessageSize>(
                    mem.GetSize() - sizeof(vislib::net::SimpleMessageHeaderData)));

                channel.SendMessage(cmsg);
            }

        } break;

        case cluster::netmessages::MSG_NETVSYNC_JOIN: {
        } break;

        default:
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Unhandled message received: %u\n",
                static_cast<unsigned int>(msg.GetHeader().GetMessageID()));
            break;
    }

}


/*
 * cluster::ClusterViewMaster::ParamUpdated
 */
void cluster::ClusterViewMaster::ParamUpdated(param::ParamSlot& slot) {
    if (!this->ctrlServer.IsRunning()) return;

    vislib::net::SimpleMessage msg;
    msg.GetHeader().SetMessageID(cluster::netmessages::MSG_SET_PARAMVALUE);
    vislib::StringA name = slot.FullName();
    vislib::StringA::Size nameL = name.Length() + 1;
    vislib::StringA value;
    vislib::StringA::Size valueL;
    vislib::UTF8Encoder::Encode(value, slot.Parameter()->ValueString());
    valueL = value.Length() + 1;

    msg.GetHeader().SetBodySize(nameL + valueL);
    msg.AssertBodySize();
    ::memcpy(msg.GetBody(), name.PeekBuffer(), nameL);
    ::memcpy(msg.GetBodyAsAt<void>(nameL), value.PeekBuffer(), valueL);

    this->ctrlServer.MultiSendMessage(msg);

}


/*
 * cluster::ClusterViewMaster::cameraUpdateThread
 */
DWORD cluster::ClusterViewMaster::cameraUpdateThread(void *userData) {
    AbstractNamedObject::const_ptr_type avp;
    const view::AbstractView *av = NULL;
    ClusterViewMaster *This = static_cast<ClusterViewMaster *>(userData);
    unsigned int syncnumber = static_cast<unsigned int>(-1);
    Call *call = NULL;
    unsigned int csn = 0;
    vislib::RawStorage mem;
    mem.AssertSize(sizeof(vislib::net::SimpleMessageHeaderData));
    vislib::RawStorageSerialiser serialiser(&mem, sizeof(vislib::net::SimpleMessageHeaderData));
    vislib::net::ShallowSimpleMessage msg(mem);

    while (true) {
        This->ModuleGraphLock().LockExclusive();
        av = NULL;
        call = This->viewSlot.CallAs<Call>();
        if ((call != NULL) && (call->PeekCalleeSlot() != NULL) && (call->PeekCalleeSlot()->Parent() != NULL)) {
            avp = call->PeekCalleeSlot()->Parent();
            av = dynamic_cast<const view::AbstractView*>(avp.get());
        }
        This->ModuleGraphLock().UnlockExclusive();
        if (av == NULL) break;

        csn = av->GetCameraSyncNumber();
        if (csn != syncnumber) {
            syncnumber = csn;
            serialiser.SetOffset(sizeof(vislib::net::SimpleMessageHeaderData));
            av->SerialiseCamera(serialiser);

            msg.SetStorage(mem, mem.GetSize());
            msg.GetHeader().SetMessageID(cluster::netmessages::MSG_SET_CAMERAVALUES);
            msg.GetHeader().SetBodySize(static_cast<vislib::net::SimpleMessageSize>(
                mem.GetSize() - sizeof(vislib::net::SimpleMessageHeaderData)));

            // TODO: Better use another server
            This->ctrlServer.MultiSendMessage(msg);

        }

        vislib::sys::Thread::Sleep(1000 / 60); // ~60 fps
    }

    return 0;
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
 * cluster::ClusterViewMaster::defaultVSyncServerAddress
 */
vislib::TString cluster::ClusterViewMaster::defaultVSyncServerAddress(void) const {
    const utility::Configuration& cfg
        = this->GetCoreInstance()->Configuration();

    if (cfg.IsConfigValueSet("cmvvsyncaddress")) { // host and port
        return cfg.ConfigValue("cmvvsyncaddress");
    }

    unsigned short netysyncport = 17226;
    try {
        if (cfg.IsConfigValueSet("cmvvsyncport")) {
            netysyncport = static_cast<unsigned short>(vislib::CharTraitsW::ParseInt(
                cfg.ConfigValue("cmvvsyncport")));
        }
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
            "Unable to parse configuration value \"cmvnetysyncportport\" as int. Configuration value ignored.");
    }

    vislib::TString address;
    address.Format(_T("%s:%u"), this->defaultServerHost().PeekBuffer(), netysyncport);
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


/*
 * cluster::ClusterViewMaster::onPauseRemoteViewClicked
 */
bool cluster::ClusterViewMaster::onPauseRemoteViewClicked(param::ParamSlot& slot) {
    ASSERT(&slot == &this->pauseRemoteViewSlot);
    vislib::net::SimpleMessage msg;
    msg.GetHeader().SetMessageID(cluster::netmessages::MSG_REMOTEVIEW_PAUSE);
    msg.GetHeader().SetBodySize(1);
    msg.AssertBodySize();
    *msg.GetBodyAs<char>() = 1;
    this->ctrlServer.MultiSendMessage(msg);
    return true;
}


/*
 * cluster::ClusterViewMaster::onResumeRemoteViewClicked
 */
bool cluster::ClusterViewMaster::onResumeRemoteViewClicked(param::ParamSlot& slot) {
    ASSERT(&slot == &this->resumeRemoteViewSlot);
    vislib::net::SimpleMessage msg;
    msg.GetHeader().SetMessageID(cluster::netmessages::MSG_REMOTEVIEW_PAUSE);
    msg.GetHeader().SetBodySize(1);
    msg.AssertBodySize();
    *msg.GetBodyAs<char>() = 0;
    this->ctrlServer.MultiSendMessage(msg);
    return true;
}


/*
 * cluster::ClusterViewMaster::onForceNetVSyncOnClicked
 */
bool cluster::ClusterViewMaster::onForceNetVSyncOnClicked(param::ParamSlot& slot) {
    ASSERT(&slot == &this->forceNetVSyncOnSlot);
    vislib::net::SimpleMessage msg;
    msg.GetHeader().SetMessageID(cluster::netmessages::MSG_FORCENETVSYNC);
    msg.GetHeader().SetBodySize(1);
    msg.AssertBodySize();
    *msg.GetBodyAs<char>() = 1;
    this->ctrlServer.MultiSendMessage(msg);
    return true;
}


/*
 * cluster::ClusterViewMaster::onForceNetVSyncOffClicked
 */
bool cluster::ClusterViewMaster::onForceNetVSyncOffClicked(param::ParamSlot& slot) {
    ASSERT(&slot == &this->forceNetVSyncOffSlot);
    vislib::net::SimpleMessage msg;
    msg.GetHeader().SetMessageID(cluster::netmessages::MSG_FORCENETVSYNC);
    msg.GetHeader().SetBodySize(1);
    msg.AssertBodySize();
    *msg.GetBodyAs<char>() = 0;
    this->ctrlServer.MultiSendMessage(msg);
    return true;
}
