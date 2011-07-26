/*
 * SimpleClusterServer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/SimpleClusterServer.h"
//#include "cluster/SimpleClusterClientViewRegistration.h"
#include "cluster/SimpleClusterCommUtil.h"
//#include "cluster/SimpleClusterView.h"
#include "CallDescriptionManager.h"
#include "CoreInstance.h"
#include "param/BoolParam.h"
#include "param/ButtonParam.h"
#include "param/IntParam.h"
#include "param/StringParam.h"
#include "RootModuleNamespace.h"
#include "view/AbstractView.h"
#include "view/CallRenderView.h"
#include "vislib/assert.h"
#include "vislib/AutoLock.h"
#include "vislib/IPAddress.h"
#include "vislib/IPCommEndPoint.h"
#include "vislib/Log.h"
#include "vislib/NetworkInformation.h"
#include "vislib/RawStorage.h"
#include "vislib/RawStorageSerialiser.h"
#include "vislib/ShallowSimpleMessage.h"
#include "vislib/Socket.h"
#include "vislib/SystemInformation.h"
#include "vislib/TcpCommChannel.h"
#include "vislib/Trace.h"
#include "vislib/UTF8Encoder.h"
//#include "vislib/SocketException.h"
//#include "AbstractNamedObject.h"
//#include <GL/gl.h>
//#include "vislib/Thread.h"


using namespace megamol::core;

//============================================================================

/*
 * cluster::SimpleClusterServer::Client::Client
 */
cluster::SimpleClusterServer::Client::Client(SimpleClusterServer& parent, vislib::SmartRef<vislib::net::AbstractCommChannel> channel)
        : vislib::net::SimpleMessageDispatchListener(), parent(parent), dispatcher(), terminationImminent(false),
        wantCamUpdates(false) {
    this->dispatcher.AddListener(this);
    ///* *HAZARD* This has to be exactly this cast! */
    //vislib::net::AbstractCommChannel *cc = dynamic_cast<vislib::net::AbstractCommChannel *>(channel.operator ->());
    vislib::net::SimpleMessageDispatcher::Configuration cfg(channel);
    this->dispatcher.Start(&cfg);
}


/*
 * cluster::SimpleClusterServer::Client::~Client
 */
cluster::SimpleClusterServer::Client::~Client(void) {
    if (this->dispatcher.IsRunning()) {
        this->terminationImminent = true;
        this->dispatcher.Terminate();
        //this->dispatcher.Join(); // blocking bullshit!
    }
}


/*
 * cluster::SimpleClusterServer::Client::Close
 */
void cluster::SimpleClusterServer::Client::Close(void) {
    if (this->dispatcher.IsRunning()) {
        this->terminationImminent = true;
        this->dispatcher.Terminate();
        this->dispatcher.Join();
    }
}


/*
 * cluster::SimpleClusterServer::Client::OnCommunicationError
 */
bool cluster::SimpleClusterServer::Client::OnCommunicationError(
        vislib::net::SimpleMessageDispatcher& src, const vislib::Exception& exception) throw() {
    if (!this->terminationImminent) {
        vislib::sys::Log::DefaultLog.WriteWarn("Server: Communication error: %s", exception.GetMsgA());
    }
    return false; // everything is lost anyway
}


/*
 * cluster::SimpleClusterServer::Client::OnDispatcherExited
 */
void cluster::SimpleClusterServer::Client::OnDispatcherExited(
        vislib::net::SimpleMessageDispatcher& src) throw() {
    vislib::sys::Log::DefaultLog.WriteInfo("Server: Client Connection %s", this->terminationImminent ? "closed" : "lost");
    // parent.clients will be updated as sfx as soon as the receiver thread terminates
    //vislib::sys::AutoLock(this->parent.clientsLock);
    //this->parent.clients.RemoveAll(this); // delete this as sfx was problematic
    // TODO: Implement
}


/*
 * cluster::SimpleClusterServer::Client::OnDispatcherStarted
 */
void cluster::SimpleClusterServer::Client::OnDispatcherStarted(
        vislib::net::SimpleMessageDispatcher& src) throw() {
    vislib::sys::Log::DefaultLog.WriteInfo("Server: Client Connection Accepted; Receiver Thread started.");
}


/*
 * cluster::SimpleClusterServer::Client::OnMessageReceived
 */
bool cluster::SimpleClusterServer::Client::OnMessageReceived(
        vislib::net::SimpleMessageDispatcher& src, const vislib::net::AbstractSimpleMessage& msg) throw() {
    using vislib::sys::Log;
    vislib::net::SimpleMessage answer;

    switch (msg.GetHeader().GetMessageID()) {
        case MSG_HANDSHAKE_INIT:
            this->name = vislib::StringA(msg.GetBodyAs<char>(), static_cast<int>(msg.GetHeader().GetBodySize()));
            Log::DefaultLog.WriteInfo("Server: Handshake with render node \"%s\" initialized\n", this->name.PeekBuffer());
            answer.GetHeader().SetMessageID(MSG_HANDSHAKE_BACK);
            this->send(answer);
            break;
        case MSG_HANDSHAKE_FORTH:
            answer.GetHeader().SetMessageID(MSG_HANDSHAKE_DONE);
            this->send(answer);
            Log::DefaultLog.WriteInfo("Server: Handshake with render node \"%s\" complete\n", this->name.PeekBuffer());
            break;
        case MSG_TIMESYNC: {
            answer = msg;
            TimeSyncData *tsd = answer.GetBodyAs<TimeSyncData>();
            if (tsd->cnt < TIMESYNCDATACOUNT) {
                tsd->time[tsd->cnt++] = this->parent.GetCoreInstance()->GetCoreInstanceTime();
            }
            this->send(answer);
        } break;
        case MSG_MODULGRAPH: {
            vislib::RawStorage mem;
            RootModuleNamespace *rmns = dynamic_cast<RootModuleNamespace*>(this->parent.RootModule());
            rmns->LockModuleGraph(false);
            rmns->SerializeGraph(mem);
            rmns->UnlockModuleGraph();
            answer.GetHeader().SetMessageID(MSG_MODULGRAPH);
            answer.SetBody(mem, mem.GetSize());
            this->send(answer);
        } break;
        case MSG_VIEWCONNECT: {
            vislib::StringA toname = this->parent.viewSlot.CallAs<Call>()->PeekCalleeSlot()->FullName();
            answer.GetHeader().SetMessageID(MSG_VIEWCONNECT);
            answer.SetBody(toname.PeekBuffer(), toname.Length());
            this->send(answer);
            this->parent.camUpdateThreadForce = true;

/*          ** does not really work
            const view::AbstractView *av = NULL;
            Call *call = NULL;
            vislib::RawStorage mem;
            mem.AssertSize(sizeof(vislib::net::SimpleMessageHeaderData));
            vislib::RawStorageSerialiser serialiser(&mem, sizeof(vislib::net::SimpleMessageHeaderData));
            vislib::net::ShallowSimpleMessage smsg(mem);

            call = this->parent.viewSlot.CallAs<Call>();
            if ((call != NULL) && (call->PeekCalleeSlot() != NULL) && (call->PeekCalleeSlot()->Parent() != NULL)) {
                av = dynamic_cast<const view::AbstractView*>(call->PeekCalleeSlot()->Parent());
            }
            if (av != NULL) {

                serialiser.SetOffset(sizeof(vislib::net::SimpleMessageHeaderData));
                av->SerialiseCamera(serialiser);

                smsg.SetStorage(mem, mem.GetSize());
                smsg.GetHeader().SetMessageID(MSG_CAMERAUPDATE);
                smsg.GetHeader().SetBodySize(static_cast<vislib::net::SimpleMessageSize>(
                    mem.GetSize() - sizeof(vislib::net::SimpleMessageHeaderData)));

                this->send(smsg);
            }
*/
        } break;
        case MSG_CAMERAUPDATE:
            this->parent.camUpdateThreadForce = true;
            break;
        case MSG_WANTCAMERAUPDATE: {
            if (msg.GetHeader().GetBodySize() < 1) break;
            this->wantCamUpdates = (msg.GetBodyAs<unsigned char>()[0] != 0);
            this->parent.camUpdateThreadForce |= this->wantCamUpdates;
            Log::DefaultLog.WriteInfo("Client %s %s camera updates", this->name.PeekBuffer(),
                (this->wantCamUpdates ? "requests" : "declines"));
        } break;
        default:
            Log::DefaultLog.WriteInfo("Server: TCP Message %d received\n", static_cast<int>(msg.GetHeader().GetMessageID()));
            break;
    }
    // TODO: Implement

    return true; // continue
}


/*
 * cluster::SimpleClusterServer::Client::send
 */
void cluster::SimpleClusterServer::Client::send(const vislib::net::AbstractSimpleMessage& msg) {
    using vislib::sys::Log;
    try {
        this->dispatcher.GetChannel()->Send(msg, msg.GetMessageSize(), vislib::net::AbstractCommChannel::TIMEOUT_INFINITE, true);
    } catch(vislib::Exception ex) {
        Log::DefaultLog.WriteError("Failed to send simple TCP message: %s\n", ex.GetMsgA());
    } catch(...) {
        Log::DefaultLog.WriteError("Failed to send simple TCP message: unexpected exception\n");
    }
}

//============================================================================

/*
 * cluster::SimpleClusterServer::SimpleClusterServer
 */
cluster::SimpleClusterServer::SimpleClusterServer(void) : Module(),
        viewnameSlot("viewname", "The parameter slot holding the name of the view module to be use"),
        viewConStatus(0), viewSlot("view", "The view to be used"),
        udpTargetSlot("udptarget", "The udp target"),
        udpTargetPortSlot("udptargetport", "The port used for udp communication"),
        udpTarget(), udpSocket(),
        clusterShutdownBtnSlot("shutdownCluster", "shutdown rendering node instances"),
        clusterNameSlot("clusterName", "The name of the cluster"),
        serverRunningSlot("server::Running", "The server running flag"),
        serverStartSlot("server::Start", "Start the server"),
        serverStopSlot("server::Stop", "Stop the server"),
        serverPortSlot("server::Port", "The server endpoint port slot"), 
        serverReconnectSlot("server::Reconnect", "Send the clients a reconnect message"),
        serverRestartSlot("server::Restart", "Restarts the TCP server"),
        serverNameSlot("server::Name", "The name for this server"),
        serverThread(), clientsLock(), clients(),
        camUpdateThread(&SimpleClusterServer::cameraUpdateThread), camUpdateThreadForce(false) {
    vislib::net::Socket::Startup();
    this->udpTarget.SetPort(0); // marks illegal endpoint

    this->clusterNameSlot << new param::StringParam("MM04SC");
    this->MakeSlotAvailable(&this->clusterNameSlot);

    this->viewnameSlot << new param::StringParam("");
    this->viewnameSlot.SetUpdateCallback(&SimpleClusterServer::onViewNameUpdated);
    this->MakeSlotAvailable(&this->viewnameSlot);

    this->viewSlot.SetCompatibleCall<view::CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->viewSlot);

    this->udpTargetSlot << new param::StringParam("");
    this->udpTargetSlot.SetUpdateCallback(&SimpleClusterServer::onUdpTargetUpdated);
    this->MakeSlotAvailable(&this->udpTargetSlot);

    this->udpTargetPortSlot << new param::IntParam(GetDatagramPort(), 1 /* 49152 */, 65535);
    this->udpTargetPortSlot.SetUpdateCallback(&SimpleClusterServer::onUdpTargetUpdated);
    this->MakeSlotAvailable(&this->udpTargetPortSlot);

    this->clusterShutdownBtnSlot << new param::ButtonParam();
    this->clusterShutdownBtnSlot.SetUpdateCallback(&SimpleClusterServer::onShutdownClusterClicked);
    this->MakeSlotAvailable(&this->clusterShutdownBtnSlot);

    this->serverRunningSlot << new param::BoolParam(false);
    this->serverRunningSlot.SetUpdateCallback(&SimpleClusterServer::onServerRunningChanged);
    this->MakeSlotAvailable(&this->serverRunningSlot);

    this->serverPortSlot << new param::IntParam(GetStreamPort(), 1 /* 49152 */, 65535);
    this->serverPortSlot.SetUpdateCallback(&SimpleClusterServer::onServerEndPointChanged);
    this->MakeSlotAvailable(&this->serverPortSlot);

    this->serverReconnectSlot << new param::ButtonParam();
    this->serverReconnectSlot.SetUpdateCallback(&SimpleClusterServer::onServerReconnectClicked);
    this->MakeSlotAvailable(&this->serverReconnectSlot);

    this->serverRestartSlot << new param::ButtonParam();
    this->serverRestartSlot.SetUpdateCallback(&SimpleClusterServer::onServerRestartClicked);
    this->MakeSlotAvailable(&this->serverRestartSlot);

    this->serverStartSlot << new param::ButtonParam();
    this->serverStartSlot.SetUpdateCallback(&SimpleClusterServer::onServerStartStopClicked);
    this->MakeSlotAvailable(&this->serverStartSlot);

    this->serverStopSlot << new param::ButtonParam();
    this->serverStopSlot.SetUpdateCallback(&SimpleClusterServer::onServerStartStopClicked);
    this->MakeSlotAvailable(&this->serverStopSlot);

    this->serverNameSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->serverNameSlot);

    this->serverThread.AddListener(this);
}


/*
 * cluster::SimpleClusterServer::~SimpleClusterServer
 */
cluster::SimpleClusterServer::~SimpleClusterServer(void) {
    this->Release();
    ASSERT(this->clients.IsEmpty());
    vislib::net::Socket::Cleanup();
}


/*
 * cluster::SimpleClusterServer::create
 */
bool cluster::SimpleClusterServer::create(void) {
    ASSERT(this->instance() != NULL);

    if (this->instance()->Configuration().IsConfigValueSet("scname")) {
        this->clusterNameSlot.Param<param::StringParam>()->SetValue(
            this->instance()->Configuration().ConfigValue("scname"), false);
    }

    if (this->instance()->Configuration().IsConfigValueSet("scservername")) {
        this->serverNameSlot.Param<param::StringParam>()->SetValue(
            this->instance()->Configuration().ConfigValue("scservername"), false);
    }

    this->udpTargetPortSlot.Param<param::IntParam>()->SetValue(GetDatagramPort(&this->instance()->Configuration()));
    this->udpTargetPortSlot.ResetDirty();
    if (this->instance()->Configuration().IsConfigValueSet("scsudptarget")) {
        this->udpTargetSlot.Param<param::StringParam>()->SetValue(
            this->instance()->Configuration().ConfigValue("scsudptarget"), false);
    }
    this->udpTargetSlot.ResetDirty();
    this->onUdpTargetUpdated(this->udpTargetSlot);

    this->serverPortSlot.Param<param::IntParam>()->SetValue(GetStreamPort(&this->instance()->Configuration()));

    if (this->instance()->Configuration().IsConfigValueSet("scsrun")) {
        try {
            bool run = vislib::CharTraitsW::ParseBool(
                this->instance()->Configuration().ConfigValue("scsrun"));
            this->serverRunningSlot.Param<param::BoolParam>()->SetValue(run);
        } catch(...) {
            vislib::sys::Log::DefaultLog.WriteError("Unable to parse configuration value scsrun");
        }
    }

    this->GetCoreInstance()->RegisterParamUpdateListener(this);

    return true;
}


/*
 * cluster::SimpleClusterServer::release
 */
void cluster::SimpleClusterServer::release(void) {
    this->GetCoreInstance()->UnregisterParamUpdateListener(this);
    this->disconnectView();
    if (this->camUpdateThread.IsRunning()) {
        this->camUpdateThread.Join();
    }
    this->stopServer(); // also disconnects clients
    this->udpSocket.Close();
}


/*
 * cluster::SimpleClusterServer::IsRunning
 */
bool cluster::SimpleClusterServer::IsRunning(void) const {
    return (this->viewConStatus >= 0);
}


/*
 * cluster::SimpleClusterServer::Start
 */
bool cluster::SimpleClusterServer::Start(void) {
    return true;
}


/*
 * cluster::SimpleClusterServer::Terminate
 */
bool cluster::SimpleClusterServer::Terminate(void) {
    this->disconnectView();
    this->stopServer(); // also disconnects clients
    return true;
}


/*
 * cluster::SimpleClusterServer::OnNewConnection
 */
bool cluster::SimpleClusterServer::OnNewConnection(const vislib::net::CommServer& src,
        vislib::SmartRef<vislib::net::AbstractCommChannel> channel) throw() {
    vislib::sys::Log::DefaultLog.WriteInfo("Incoming TCP connection");
    vislib::sys::AutoLock(this->clientsLock);
    this->clients.Add(new Client(*this, channel));
    return true;
}


/*
 * cluster::SimpleClusterServer::ParamUpdated
 */
void cluster::SimpleClusterServer::ParamUpdated(param::ParamSlot& slot) {
    vislib::net::SimpleMessage msg;
    msg.GetHeader().SetMessageID(MSG_PARAMUPDATE);
    vislib::StringA name = slot.FullName();
    vislib::StringA value;
    vislib::UTF8Encoder::Encode(value, slot.Param<param::AbstractParam>()->ValueString());
    name.Append("=");
    name.Append(value);
    msg.SetBody(name, name.Length());
    for (SIZE_T i = 0; i < this->clients.Count(); i++) {
        if (!this->clients[i]->IsRunning()) continue;
        this->clients[i]->Send(msg);
    }
}


/*
 * cluster::SimpleClusterServer::onShutdownClusterClicked
 */
bool cluster::SimpleClusterServer::onShutdownClusterClicked(param::ParamSlot& slot) {
    ASSERT(&slot == &this->clusterShutdownBtnSlot);
    SimpleClusterDatagram datagram;
    datagram.msg = MSG_SHUTDOWN;
    vislib::StringA cn(this->clusterNameSlot.Param<param::StringParam>()->Value());
    if (cn.Length() > 127) cn.Truncate(127);
    datagram.payload.Strings.len1 = static_cast<unsigned char>(cn.Length());
    ::memcpy(datagram.payload.Strings.str1, cn.PeekBuffer(), datagram.payload.Strings.len1);
    this->sendUDPDiagram(datagram);
    return true;
}


/*
 * cluster::SimpleClusterServer::onUdpTargetUpdated
 */
bool cluster::SimpleClusterServer::onUdpTargetUpdated(param::ParamSlot& slot) {
    ASSERT((&slot == &this->udpTargetSlot) || (&slot == &this->udpTargetPortSlot));
    this->udpSocket.Close(); // will be lazy initialized the next time required

    vislib::StringA host(this->udpTargetSlot.Param<param::StringParam>()->Value());
    int port = this->udpTargetPortSlot.Param<param::IntParam>()->Value();
    try {
        this->udpTarget.SetPort(0); // marks illegal endpoint

        vislib::net::IPAddress addr;
        if (addr.Lookup(host)) {
            this->udpTarget.SetIPAddress(addr); // TODO: makes no sense must be best available broadcast address
            this->udpTarget.SetPort(port);
            vislib::sys::Log::DefaultLog.WriteInfo("UDPTarget set to %s\n",
                this->udpTarget.ToStringA().PeekBuffer());
        } else {
            vislib::sys::Log::DefaultLog.WriteError("Unable to set new udp target: %s is not a valid IPv4 host\n", host.PeekBuffer());
        }

    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to set new udp target: %s\n", ex.GetMsgA());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to set new udp target: unexpected exception\n");
    }

    return true;
}


/*
 * cluster::SimpleClusterServer::onViewNameUpdated
 */
bool cluster::SimpleClusterServer::onViewNameUpdated(param::ParamSlot& slot) {
    ASSERT(&slot == &this->viewnameSlot);
    this->disconnectView();
    vislib::StringA viewmodname(this->viewnameSlot.Param<param::StringParam>()->Value());
    this->LockModuleGraph(false);
    megamol::core::view::AbstractView *av = dynamic_cast<megamol::core::view::AbstractView *>(this->FindNamedObject(viewmodname, true));
    this->UnlockModuleGraph();
    if (av != NULL) {
        if (this->instance()->InstantiateCall(this->viewSlot.FullName(), av->FullName() + "::render", 
                CallDescriptionManager::Instance()->Find(view::CallRenderView::ClassName())) != NULL) {
            this->newViewConnected();
        } else {
            av = NULL; // DO NOT DELETE
        }
    }
    if (av == NULL){
        vislib::sys::Log::DefaultLog.WriteError(
            "Unable to connect SimpleClusterServer to view \"%s\": not found or incompatible type\n",
            viewmodname.PeekBuffer());
    }

    return true;
}


/*
 * cluster::SimpleClusterServer::disconnectView
 */
void cluster::SimpleClusterServer::disconnectView(void) {
    if (this->viewConStatus == 1) {
        this->viewSlot.DisconnectCalls();
        this->viewSlot.ConnectCall(NULL);
        this->viewConStatus = -1; // disconnected
        vislib::sys::Log::DefaultLog.WriteInfo("SCS: View Disconnected");
        this->stopServer();
        if (this->camUpdateThread.IsRunning()) {
            this->camUpdateThread.Join();
        }
    }
}


/*
 * cluster::SimpleClusterServer::newViewConnected
 */
void cluster::SimpleClusterServer::newViewConnected(void) {
    ASSERT(this->viewConStatus != 1);
    ASSERT(this->viewSlot.CallAs<view::CallRenderView>() != NULL);
    ASSERT(this->viewSlot.CallAs<view::CallRenderView>()->PeekCalleeSlot() != NULL);
    ASSERT(this->viewSlot.CallAs<view::CallRenderView>()->PeekCalleeSlot()->Parent() != NULL);
    vislib::sys::Log::DefaultLog.WriteInfo("SCS: View \"%s\" Connected",
        this->viewSlot.CallAs<view::CallRenderView>()->PeekCalleeSlot()->Parent()->FullName().PeekBuffer());
    this->viewConStatus = 1;
    this->onServerReconnectClicked(this->serverReconnectSlot);
    this->camUpdateThread.Start(this);
}


/*
 * cluster::SimpleClusterServer::sendUDPDiagram
 */
void cluster::SimpleClusterServer::sendUDPDiagram(cluster::SimpleClusterDatagram& datagram) {
    try {
        if (!this->udpSocket.IsValid()) {
            this->udpSocket.Create(vislib::net::Socket::FAMILY_INET, vislib::net::Socket::TYPE_DGRAM, vislib::net::Socket::PROTOCOL_UDP);
        }
        datagram.cntEchoed = 0;
        if (this->udpTarget.GetIPAddress4()[3] != 0) { // assume it's not a broadcast address
            datagram.cntEchoed++;
        }
        if (this->udpTarget.GetPort() != 0) {
            this->udpSocket.Send(this->udpTarget, &datagram, sizeof(cluster::SimpleClusterDatagram));
            VLTRACE(VISLIB_TRCELVL_INFO, "Server >>> UDP Datagram sent to %s\n", this->udpTarget.ToStringA().PeekBuffer());
        } else {
            vislib::sys::Log::DefaultLog.WriteWarn("Not udp target set to send the message");
        }

    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteError("SCS: Unable to send udp message: %s\n", ex.GetMsgA());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteError("SCS: Unable to send udp message: unexpected exception\n");
    }
}


/*
 * cluster::SimpleClusterServer::stopServer
 */
void cluster::SimpleClusterServer::stopServer(void) {
    if (this->serverThread.IsRunning()) {
        vislib::sys::Thread::Sleep(2000); // *HAZARD* Terminating a server which has not reached listen state results in undefined behaviour
        this->serverThread.Terminate();
        this->serverThread.Join();
    }

    this->clientsLock.Lock();
    while (this->clients.Count() > 0) {
        while (!this->clients.IsEmpty() && !this->clients[0]->IsRunning()) {
            this->clients.RemoveFirst();
        }
        for (SIZE_T i = 0; i < this->clients.Count(); i++) {
            if (this->clients[i]->IsRunning()) {
                this->clients[i]->Close();
            }
        }
        vislib::sys::Thread::Sleep(100);
    }
    this->clientsLock.Unlock();

}


/*
 * cluster::SimpleClusterServer::onServerRunningChanged
 */
bool cluster::SimpleClusterServer::onServerRunningChanged(param::ParamSlot& slot) {
    bool tarVal = this->serverRunningSlot.Param<param::BoolParam>()->Value();
    if (this->serverThread.IsRunning() != tarVal) {
        if (tarVal) {
            this->onServerRestartClicked(slot);
        } else {
            this->stopServer();
        }
    }
    return true;
}


/*
 * cluster::SimpleClusterServer::onServerEndPointChanged
 */
bool cluster::SimpleClusterServer::onServerEndPointChanged(param::ParamSlot& slot) {
    this->stopServer();
    this->onServerRestartClicked(slot);
    return true;
}


/*
 * cluster::SimpleClusterServer::onServerReconnectClicked
 */
bool cluster::SimpleClusterServer::onServerReconnectClicked(param::ParamSlot& slot) {
    if (!this->serverThread.IsRunning()) {
        if (&slot == &this->serverReconnectSlot) vislib::sys::Log::DefaultLog.WriteWarn("TCP-Server is not running");
        return true;
    }
    if (this->viewSlot.CallAs<Call>() == NULL) {
        if (&slot == &this->serverReconnectSlot) vislib::sys::Log::DefaultLog.WriteWarn("No view connected");
        return true;
    }
    SimpleClusterDatagram datagram;
    datagram.msg = MSG_CONNECTTOSERVER;

    vislib::StringA cn(this->clusterNameSlot.Param<param::StringParam>()->Value());
    if (cn.Length() > 127) cn.Truncate(127);
    datagram.payload.Strings.len1 = static_cast<unsigned char>(cn.Length());
    ::memcpy(datagram.payload.Strings.str1, cn.PeekBuffer(), datagram.payload.Strings.len1);

    vislib::StringA ses, compName;//(this->serverThread.GetBindAddressA());
    compName = this->serverNameSlot.Param<param::StringParam>()->Value();
    if (compName.IsEmpty()) {
        vislib::sys::SystemInformation::ComputerName(compName);
    }
    //compName = "129.69.205.29";
    //compName = "127.0.0.1";
    ses.Format("%s:%d", compName.PeekBuffer(), this->serverPortSlot.Param<param::IntParam>()->Value());
    if (ses.Length() > 127) ses.Truncate(127);
    datagram.payload.Strings.len2 = static_cast<unsigned char>(ses.Length());
    ::memcpy(datagram.payload.Strings.str2, ses.PeekBuffer(), datagram.payload.Strings.len2);

    this->sendUDPDiagram(datagram);

    return true;
}


/*
 * cluster::SimpleClusterServer::onServerRestartClicked
 */
bool cluster::SimpleClusterServer::onServerRestartClicked(param::ParamSlot& slot) {
    using vislib::sys::Log;
    this->stopServer();

    if (!this->serverRunningSlot.Param<param::BoolParam>()->Value()) {
        if (&slot == &this->serverRestartSlot) {
            Log::DefaultLog.WriteWarn("TCP-Server not started: server disabled");
        }
        return true;
    }

    vislib::net::IPEndPoint ep;
    ep.SetIPAddress(vislib::net::IPAddress::ANY);
    ep.SetPort(this->serverPortSlot.Param<param::IntParam>()->Value());

    vislib::net::CommServer::Configuration cfg(vislib::net::TcpCommChannel::Create(vislib::net::TcpCommChannel::FLAG_NODELAY
        | vislib::net::TcpCommChannel::FLAG_REUSE_ADDRESS), vislib::net::IPCommEndPoint::Create(ep));
    this->serverThread.Start(&cfg);
    vislib::sys::Thread::Sleep(500);

    if (this->serverThread.IsRunning()) {
        Log::DefaultLog.WriteInfo("TCP-Server started on %s\n", ep.ToStringA().PeekBuffer());
        this->onServerReconnectClicked(slot);
    }
    return true;
}


/*
 * cluster::SimpleClusterServer::onServerStartStopClicked
 */
bool cluster::SimpleClusterServer::onServerStartStopClicked(param::ParamSlot& slot) {
    this->serverRunningSlot.Param<param::BoolParam>()->SetValue(&slot == &this->serverStartSlot);
    //this->onServerRunningChanged(slot);
    return true;
}


/*
 * cluster::SimpleClusterServer::cameraUpdateThread
 */
DWORD cluster::SimpleClusterServer::cameraUpdateThread(void *userData) {
    const view::AbstractView *av = NULL;
    SimpleClusterServer *This = static_cast<SimpleClusterServer *>(userData);
    unsigned int syncnumber = static_cast<unsigned int>(-1);
    Call *call = NULL;
    unsigned int csn = 0;
    vislib::RawStorage mem;
    mem.AssertSize(sizeof(vislib::net::SimpleMessageHeaderData));
    vislib::RawStorageSerialiser serialiser(&mem, sizeof(vislib::net::SimpleMessageHeaderData));
    vislib::net::ShallowSimpleMessage msg(mem);

    while (true) {
        This->LockModuleGraph(false);
        av = NULL;
        call = This->viewSlot.CallAs<Call>();
        if ((call != NULL) && (call->PeekCalleeSlot() != NULL) && (call->PeekCalleeSlot()->Parent() != NULL)) {
            av = dynamic_cast<const view::AbstractView*>(call->PeekCalleeSlot()->Parent());
        }
        This->UnlockModuleGraph();
        if (av == NULL) break;

        csn = av->GetCameraSyncNumber();
        if ((csn != syncnumber) || This->camUpdateThreadForce) {
            syncnumber = csn;
            This->camUpdateThreadForce = false;
            serialiser.SetOffset(sizeof(vislib::net::SimpleMessageHeaderData));
            av->SerialiseCamera(serialiser);

            msg.SetStorage(mem, mem.GetSize());
            msg.GetHeader().SetMessageID(MSG_CAMERAUPDATE);
            msg.GetHeader().SetBodySize(static_cast<vislib::net::SimpleMessageSize>(
                mem.GetSize() - sizeof(vislib::net::SimpleMessageHeaderData)));

            // Better use another server
            This->clientsLock.Lock();
            for (SIZE_T i = 0; i < This->clients.Count(); i++) {
                if (This->clients[i]->IsRunning() && This->clients[i]->WantCameraUpdates()) {
                    This->clients[i]->Send(msg);
                }
            }
            This->clientsLock.Unlock();

        }

        vislib::sys::Thread::Sleep(1000 / 60); // ~60 fps
    }

    return 0;
}
