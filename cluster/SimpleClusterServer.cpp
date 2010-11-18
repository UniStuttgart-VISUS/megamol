/*
 * SimpleClusterServer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/SimpleClusterServer.h"
//#include "cluster/SimpleClusterClientViewRegistration.h"
#include "cluster/SimpleClusterDatagram.h"
//#include "cluster/SimpleClusterView.h"
#include "CallDescriptionManager.h"
#include "CoreInstance.h"
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
        clusterShutdownBtnSlot("shutdownCluster", "shutdown rendering node instances") {
    vislib::net::Socket::Startup();
    this->udpTarget.SetPort(0); // marks illegal endpoint

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

    this->udpTargetPortSlot.Param<param::IntParam>()->SetValue(GetDatagramPort(&this->instance()->Configuration()));
    this->udpTargetPortSlot.ResetDirty();
    if (this->instance()->Configuration().IsConfigValueSet("scsudptarget")) {
        this->udpTargetSlot.Param<param::StringParam>()->SetValue(
            this->instance()->Configuration().ConfigValue("scsudptarget"));
    }
    this->udpTargetSlot.ResetDirty();
    this->onUdpTargetUpdated(this->udpTargetSlot);

    return true;
}


/*
 * cluster::SimpleClusterServer::release
 */
void cluster::SimpleClusterServer::release(void) {
    this->disconnectView();

    //vislib::Array<SimpleClusterView*> scv(this->views);
    //this->views.Clear();
    //for (unsigned int i = 0; i < scv.Count(); i++) {
    //    scv[i]->Unregister(this);
    //}
    //this->udpInSocket.Close();
    //if (this->udpReceiver.IsRunning()) {
    //    this->udpReceiver.Join();
    //}

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

    // TODO: Implement

    return true;
}


/*
 * cluster::SimpleClusterServer::Terminate
 */
bool cluster::SimpleClusterServer::Terminate(void) {
    this->disconnectView();

    // TODO: Implement

    return true;
}


/*
 * cluster::SimpleClusterServer::onShutdownClusterClicked
 */
bool cluster::SimpleClusterServer::onShutdownClusterClicked(param::ParamSlot& slot) {
    ASSERT(&slot == &this->clusterShutdownBtnSlot);
    SimpleClusterDatagram datagram;
    datagram.msg = MSG_SHUTDOWN;
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

    // TODO: Implement

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

    SimpleClusterDatagram datagram;
    datagram.msg = MSG_CONNECTTOSERVER;
    strcpy(datagram.payload.data, "unknown TODO fixme");

    // TODO: Implement

    this->sendUDPDiagram(datagram);
}


/*
 * cluster::SimpleClusterServer::sendUDPDiagram
 */
void cluster::SimpleClusterServer::sendUDPDiagram(cluster::SimpleClusterDatagram& datagram) {
    try {
        if (!this->udpSocket.IsValid()) {
            this->udpSocket.Create(vislib::net::Socket::FAMILY_INET, vislib::net::Socket::TYPE_DGRAM, vislib::net::Socket::PROTOCOL_UDP);
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
