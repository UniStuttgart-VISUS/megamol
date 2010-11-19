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
#include "view/AbstractView.h"
#include "view/CallRenderView.h"
#include "vislib/assert.h"
#include "vislib/IPAddress.h"
#include "vislib/Log.h"
#include "vislib/NetworkInformation.h"
#include "vislib/Socket.h"
#include "vislib/SystemInformation.h"
#include "vislib/TcpCommChannel.h"
//#include "vislib/SocketException.h"
//#include "AbstractNamedObject.h"
//#include <GL/gl.h>
//#include "vislib/Thread.h"


using namespace megamol::core;


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
        serverEndPointAddrSlot("server::EndPointAddr", "The server endpoint address slot"), 
        serverEndPointPortSlot("server::EndPointPort", "The server endpoint port slot"), 
        serverReconnectSlot("server::Reconnect", "Send the clients a reconnect message"),
        serverRestartSlot("server::Restart", "Restarts the TCP server"),
        serverThread() {
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

    vislib::TString compName;
    vislib::sys::SystemInformation::ComputerName(compName);
    this->serverEndPointAddrSlot << new param::StringParam(compName);
    this->serverEndPointAddrSlot.SetUpdateCallback(&SimpleClusterServer::onServerEndPointChanged);
    this->MakeSlotAvailable(&this->serverEndPointAddrSlot);

    this->serverEndPointPortSlot << new param::IntParam(GetStreamPort(), 1 /* 49152 */, 65535);
    this->serverEndPointPortSlot.SetUpdateCallback(&SimpleClusterServer::onServerEndPointChanged);
    this->MakeSlotAvailable(&this->serverEndPointPortSlot);

    this->serverReconnectSlot << new param::ButtonParam();
    this->serverReconnectSlot.SetUpdateCallback(&SimpleClusterServer::onServerReconnectClicked);
    this->MakeSlotAvailable(&this->serverReconnectSlot);

    this->serverRestartSlot << new param::ButtonParam();
    this->serverRestartSlot.SetUpdateCallback(&SimpleClusterServer::onServerRestartClicked);
    this->MakeSlotAvailable(&this->serverRestartSlot);

    this->serverThread.AddListener(this);

    //this->registerViewSlot.SetCallback(
    //    SimpleClusterClientViewRegistration::ClassName(),
    //    SimpleClusterClientViewRegistration::FunctionName(0),
    //    &SimpleClusterClient::onViewRegisters);
    //this->MakeSlotAvailable(&this->registerViewSlot);

    //this->udpPortSlot << new param::IntParam(
    //    GetDatagramPort(), 1 /* 49152 */, 65535);
    //this->udpPortSlot.SetUpdateCallback(&SimpleClusterClient::onUdpPortChanged);
    //this->MakeSlotAvailable(&this->udpPortSlot);
}


/*
 * cluster::SimpleClusterServer::~SimpleClusterServer
 */
cluster::SimpleClusterServer::~SimpleClusterServer(void) {
    this->Release();
    //ASSERT(!this->udpReceiver.IsRunning());
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

    if (this->instance()->Configuration().IsConfigValueSet("scsrun")) {
        try {
            bool run = vislib::TCharTraits::ParseBool(
            this->instance()->Configuration().ConfigValue("scsrun"));
            this->serverRunningSlot.Param<param::BoolParam>()->SetValue(run);
        } catch(...) {
            vislib::sys::Log::DefaultLog.WriteError("Unable to parse configuration value scsrun");
        }
    }

    this->udpTargetPortSlot.Param<param::IntParam>()->SetValue(GetDatagramPort(&this->instance()->Configuration()));
    this->udpTargetPortSlot.ResetDirty();
    if (this->instance()->Configuration().IsConfigValueSet("scsudptarget")) {
        this->udpTargetSlot.Param<param::StringParam>()->SetValue(
            this->instance()->Configuration().ConfigValue("scsudptarget"), false);
    }
    this->udpTargetSlot.ResetDirty();
    this->onUdpTargetUpdated(this->udpTargetSlot);

    this->serverEndPointPortSlot.Param<param::IntParam>()->SetValue(GetStreamPort(&this->instance()->Configuration()));
    this->serverEndPointPortSlot.ResetDirty();
    if (this->instance()->Configuration().IsConfigValueSet("sctcpaddr")) {
        this->serverEndPointAddrSlot.Param<param::StringParam>()->SetValue(
            this->instance()->Configuration().ConfigValue("sctcpaddr"), false);
    }
    this->onServerEndPointChanged(this->serverEndPointAddrSlot);

    return true;
}


/*
 * cluster::SimpleClusterServer::release
 */
void cluster::SimpleClusterServer::release(void) {
    this->disconnectView();
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
    vislib::sys::Log::DefaultLog.WriteError("Incoming TCP connection rejected");

    // TODO: Implement

    return false;
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
            this->udpTarget.SetIPAddress(addr);
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
        this->viewConStatus = -1; // disconnected
        vislib::sys::Log::DefaultLog.WriteInfo("SCS: View Disconnected");
    }
    this->stopServer();
}


/*
 * cluster::SimpleClusterServer::newViewConnected
 */
void cluster::SimpleClusterServer::newViewConnected(void) {
    ASSERT(this->viewConStatus != 1);
    // TODO: ASSERT no TCP-Connections
    ASSERT(this->viewSlot.CallAs<view::CallRenderView>() != NULL);
    ASSERT(this->viewSlot.CallAs<view::CallRenderView>()->PeekCalleeSlot() != NULL);
    ASSERT(this->viewSlot.CallAs<view::CallRenderView>()->PeekCalleeSlot()->Parent() != NULL);
    vislib::sys::Log::DefaultLog.WriteInfo("SCS: View \"%s\" Connected",
        this->viewSlot.CallAs<view::CallRenderView>()->PeekCalleeSlot()->Parent()->FullName().PeekBuffer());
    this->viewConStatus = 1;
    this->stopServer();
    this->onServerRestartClicked(this->serverReconnectSlot);
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
        this->serverThread.Terminate();
    }

    // TODO: Implement
    // Disconnects TCP clients

}


/*
 * cluster::SimpleClusterServer::onServerRunningChanged
 */
bool cluster::SimpleClusterServer::onServerRunningChanged(param::ParamSlot& slot) {
    this->stopServer();
    if (this->serverRunningSlot.Param<param::BoolParam>()->Value()) {
        this->onServerRestartClicked(slot);
    }
    return true;
}


/*
 * cluster::SimpleClusterServer::onServerEndPointChanged
 */
bool cluster::SimpleClusterServer::onServerEndPointChanged(param::ParamSlot& slot) {
    this->stopServer();
    vislib::net::IPEndPoint ep;
    if (this->getServerEndPoint(ep)) {
        this->onServerRestartClicked(slot);
    }
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

    vislib::StringA ses(this->serverThread.GetBindAddressA());
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
        Log::DefaultLog.WriteWarn("TCP-Server not started: server disabled");
        return true;
    }

    vislib::net::IPEndPoint ep;
    if (!this->getServerEndPoint(ep)) {
        Log::DefaultLog.WriteWarn("Unable to start TCP-Server: End Point is not valid");
        return true;
    }

    this->serverThread.Configure(
        new vislib::net::TcpCommChannel(vislib::net::TcpCommChannel::FLAG_NODELAY),
        ep.ToStringA());
    this->serverThread.Start();

    if (this->serverThread.IsRunning()) {
        Log::DefaultLog.WriteInfo("TCP-Server started on %s\n", ep.ToStringA());
        this->onServerReconnectClicked(slot);
    }
    return true;
}


/*
 * cluster::SimpleClusterServer::getServerEndPoint
 */
bool cluster::SimpleClusterServer::getServerEndPoint(vislib::net::IPEndPoint& outEP) {
    int port = this->serverEndPointPortSlot.Param<param::IntParam>()->Value();
    vislib::StringA addr(this->serverEndPointAddrSlot.Param<param::StringParam>()->Value());
    vislib::net::IPEndPoint lep;

    float guess = vislib::net::NetworkInformation::GuessLocalEndPoint(lep, addr,
        vislib::net::IPAgnosticAddress::FAMILY_INET /* TODO: Configurable */
        );
    lep.SetPort(port);
    if (guess > 0.8) {
        vislib::sys::Log::DefaultLog.WriteError("Guessed local server end point %s from input %s:%d with wildness %f: Too wild!\n",
            lep.ToStringA().PeekBuffer(), addr.PeekBuffer(), port, guess);
        return false;
    }
    vislib::sys::Log::DefaultLog.WriteMsg(
        (guess > 0.2) ? vislib::sys::Log::LEVEL_WARN : vislib::sys::Log::LEVEL_INFO,
        "Guessed local server end point %s from input %s:%d with wildness %f\n",
        lep.ToStringA().PeekBuffer(), addr.PeekBuffer(), port, guess);
    outEP = lep;
    return true;
}
