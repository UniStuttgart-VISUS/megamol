/*
 * SimpleClusterClient.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/SimpleClusterClient.h"
#include "cluster/SimpleClusterClientViewRegistration.h"
#include "cluster/SimpleClusterDatagram.h"
#include "cluster/SimpleClusterView.h"
#include "CoreInstance.h"
#include "param/IntParam.h"
#include "vislib/assert.h"
#include "vislib/Log.h"
#include "vislib/Socket.h"
#include "vislib/SocketException.h"
//#include "AbstractNamedObject.h"
//#include <GL/gl.h>
//#include "vislib/Thread.h"


using namespace megamol::core;


/*
 * cluster::SimpleClusterClient::SimpleClusterClient
 */
cluster::SimpleClusterClient::SimpleClusterClient(void) : Module(),
        registerViewSlot("registerView", "The slot views may register at"),
        views(),
        udpPortSlot("udpport", "The port used for udp communication"),
        udpInSocket(), udpReceiver(&SimpleClusterClient::udpReceiverLoop) {
    vislib::net::Socket::Startup();

    this->registerViewSlot.SetCallback(
        SimpleClusterClientViewRegistration::ClassName(),
        SimpleClusterClientViewRegistration::FunctionName(0),
        &SimpleClusterClient::onViewRegisters);
    this->MakeSlotAvailable(&this->registerViewSlot);

    this->udpPortSlot << new param::IntParam(
        GetDatagramPort(), 1 /* 49152 */, 65535);
    this->udpPortSlot.SetUpdateCallback(&SimpleClusterClient::onUdpPortChanged);
    this->MakeSlotAvailable(&this->udpPortSlot);
}


/*
 * cluster::SimpleClusterClient::~SimpleClusterClient
 */
cluster::SimpleClusterClient::~SimpleClusterClient(void) {
    this->Release();
    ASSERT(!this->udpReceiver.IsRunning());
    vislib::net::Socket::Cleanup();
}


/*
 * cluster::SimpleClusterClient::Unregister
 */
void cluster::SimpleClusterClient::Unregister(cluster::SimpleClusterView *view) {
    if (view == NULL) return;
    if (this->views.Contains(view)) {
        this->views.RemoveAll(view);
        view->Unregister(this);
    }
}


/*
 * cluster::SimpleClusterClient::create
 */
bool cluster::SimpleClusterClient::create(void) {
    this->udpPortSlot.Param<param::IntParam>()->SetValue(
        GetDatagramPort(&this->instance()->Configuration()));
    this->udpPortSlot.ResetDirty();
    this->onUdpPortChanged(this->udpPortSlot);

    return true;
}


/*
 * cluster::SimpleClusterClient::release
 */
void cluster::SimpleClusterClient::release(void) {
    vislib::Array<SimpleClusterView*> scv(this->views);
    this->views.Clear();
    for (unsigned int i = 0; i < scv.Count(); i++) {
        scv[i]->Unregister(this);
    }
    this->udpInSocket.Close();
    if (this->udpReceiver.IsRunning()) {
        this->udpReceiver.Join();
    }

}


/*
 * cluster::SimpleClusterClient::udpReceiverLoop
 */
DWORD cluster::SimpleClusterClient::udpReceiverLoop(void *ctxt) {
    SimpleClusterClient *that = reinterpret_cast<SimpleClusterClient *>(ctxt);
    SimpleClusterDatagram datagram;
    vislib::sys::Log::DefaultLog.WriteInfo("UDP Receiver started\n");
    try {
        while (that->udpInSocket.IsValid()) {
            that->udpInSocket.Receive(&datagram, sizeof(datagram));

            // TODO: Work with datagram
            vislib::sys::Log::DefaultLog.WriteInfo("UDP Receiver: datagram %u received\n", datagram.msg);

        }
    } catch(vislib::net::SocketException sex) {
        DWORD errc = sex.GetErrorCode();
        if (errc != 995) {
            vislib::sys::Log::DefaultLog.WriteError("UDP Receive error: (%u) %s\n", errc, sex.GetMsgA());
        }
    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteError("UDP Receive error: %s\n", ex.GetMsgA());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteError("UDP Receive error: unexpected exception\n");
    }
    vislib::sys::Log::DefaultLog.WriteInfo("UDP Receiver stopped\n");
    return 0;
}


/*
 * cluster::SimpleClusterClient::onViewRegisters
 */
bool cluster::SimpleClusterClient::onViewRegisters(Call& call) {
    SimpleClusterClientViewRegistration *sccvr = dynamic_cast<SimpleClusterClientViewRegistration*>(&call);
    if (sccvr == NULL) return false;
    sccvr->SetClient(this);
    if (!this->views.Contains(sccvr->GetView())) {
        this->views.Add(sccvr->GetView());
    }
    return true;
}


/*
 * cluster::SimpleClusterClient::onUdpPortChanged
 */
bool cluster::SimpleClusterClient::onUdpPortChanged(param::ParamSlot& slot) {
    using vislib::sys::Log;
    ASSERT(&slot == &this->udpPortSlot);

    try {
        this->udpInSocket.Close();
        if (this->udpReceiver.IsRunning()) {
            this->udpReceiver.Join();
        }
        this->udpInSocket.Create(vislib::net::Socket::FAMILY_INET,
            vislib::net::Socket::TYPE_DGRAM,
            vislib::net::Socket::PROTOCOL_UDP);
        this->udpInSocket.Bind(vislib::net::IPEndPoint(vislib::net::IPAddress::ANY,
            this->udpPortSlot.Param<param::IntParam>()->Value()));
        this->udpReceiver.Start(reinterpret_cast<void*>(this));

    } catch(vislib::Exception ex) {
        Log::DefaultLog.WriteError("Failed to start UDP: %s\n", ex.GetMsgA());
    } catch(...) {
        Log::DefaultLog.WriteError("Failed to start UDP: unexpected exception\n");
    }
    return true;
}
