/*
 * SimpleClusterClient.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/SimpleClusterClient.h"
#include "cluster/SimpleClusterClientViewRegistration.h"
#include "cluster/SimpleClusterCommUtil.h"
#include "cluster/SimpleClusterView.h"
#include "CoreInstance.h"
#include "param/IntParam.h"
#include "param/StringParam.h"
#include "vislib/assert.h"
#include "vislib/Log.h"
#include "vislib/NetworkInformation.h"
#include "vislib/Socket.h"
#include "vislib/SocketException.h"
#include <signal.h>
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
        udpInSocket(), udpReceiver(&SimpleClusterClient::udpReceiverLoop),
        clusterNameSlot("clusterName", "The name of the cluster"),
        udpEchoBAddrSlot("udpechoaddr", "The address to echo broadcast udp messages"),
        tcpChan(NULL), tcpSan(), conServerAddr("") {
    vislib::net::Socket::Startup();

    this->clusterNameSlot << new param::StringParam("MM04SC");
    this->MakeSlotAvailable(&this->clusterNameSlot);

    this->registerViewSlot.SetCallback(
        SimpleClusterClientViewRegistration::ClassName(),
        SimpleClusterClientViewRegistration::FunctionName(0),
        &SimpleClusterClient::onViewRegisters);
    this->MakeSlotAvailable(&this->registerViewSlot);

    this->udpPortSlot << new param::IntParam(GetDatagramPort(), 1 /* 49152 */, 65535);
    this->udpPortSlot.SetUpdateCallback(&SimpleClusterClient::onUdpPortChanged);
    this->MakeSlotAvailable(&this->udpPortSlot);

    this->udpEchoBAddrSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->udpEchoBAddrSlot);

    this->tcpSan.AddListener(this);

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


/**
 * Selects adapters capable of UDP broadcast
 *
 * @param adapter The adapter to tesh
 * @param userContext not used
 *
 * @return True if the adapter is capable of UDP broadcast
 */
bool udpBroadcastAdapters(const vislib::net::NetworkInformation::Adapter& adapter, void *userContext) {
    vislib::net::NetworkInformation::Confidence conf;
    vislib::net::NetworkInformation::Adapter::OperStatus opStat = adapter.GetStatus(&conf);
    if (conf != vislib::net::NetworkInformation::INVALID) {
        if (opStat != vislib::net::NetworkInformation::Adapter::OPERSTATUS_UP) {
            return false;
        }
    }
    adapter.GetBroadcastAddress(&conf);
    return (conf != vislib::net::NetworkInformation::INVALID);
}


/*
 * cluster::SimpleClusterClient::create
 */
bool cluster::SimpleClusterClient::create(void) {

    if (this->instance()->Configuration().IsConfigValueSet("scname")) {
        this->clusterNameSlot.Param<param::StringParam>()->SetValue(
            this->instance()->Configuration().ConfigValue("scname"));
    }

    if (this->instance()->Configuration().IsConfigValueSet("sccudpechobaddr")) {
        this->udpEchoBAddrSlot.Param<param::StringParam>()->SetValue(
            this->instance()->Configuration().ConfigValue("sccudpechobaddr"));
    } else {
        vislib::net::NetworkInformation::AdapterList adapters;
        vislib::net::NetworkInformation::GetAdaptersForPredicate(adapters, &udpBroadcastAdapters);
        if (adapters.Count() > 0) {
            vislib::net::NetworkInformation::Confidence bcac;
            vislib::net::IPAddress bca;
            bool bcaset = false;
            for (SIZE_T i = 0; i < adapters.Count(); i++) {
                bca = adapters[i].GetBroadcastAddress(&bcac);
                if (bcac == vislib::net::NetworkInformation::VALID) {
                    this->udpEchoBAddrSlot.Param<param::StringParam>()->SetValue(bca.ToStringA());
                    bcaset = true;
                    break;
                }
            }
            if (!bcaset) for (SIZE_T i = 0; i < adapters.Count(); i++) {
                bca = adapters[i].GetBroadcastAddress(&bcac);
                if (bcac != vislib::net::NetworkInformation::INVALID) {
                    this->udpEchoBAddrSlot.Param<param::StringParam>()->SetValue(bca.ToStringA());
                    break;
                }
            }
        }
    }

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
    if (this->tcpChan != NULL) {
        this->tcpChan->Close();
        this->tcpChan->Release();
        this->tcpChan = NULL;
    }
    if (this->tcpSan.IsRunning()) {
        this->tcpSan.Terminate();
        this->tcpSan.Join();
    }

}


/*
 * cluster::SimpleClusterClient::OnMessageReceived
 */
bool cluster::SimpleClusterClient::OnMessageReceived(vislib::net::SimpleMessageDispatcher& src,
        const vislib::net::AbstractSimpleMessage& msg) throw() {

    vislib::sys::Log::DefaultLog.WriteInfo("Client: TCP Message %d received\n",
        static_cast<int>(msg.GetHeader().GetMessageID()));

    // TODO: Implement

    return true;
}

/*
 * cluster::SimpleClusterClient::OnCommunicationError
 */
bool cluster::SimpleClusterClient::OnCommunicationError(vislib::net::SimpleMessageDispatcher& src,
            const vislib::Exception& exception) throw() {

    vislib::sys::Log::DefaultLog.WriteInfo("Client: Receiver failed: %s\n",
        exception.GetMsgA());

    return false;
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
            //vislib::net::IPEndPoint fromEP;
            that->udpInSocket.Receive(/*fromEP, */&datagram, sizeof(datagram));

            if (datagram.cntEchoed > 0) {
                datagram.cntEchoed--;
                // echo broadcast
                vislib::StringA baddr(that->udpEchoBAddrSlot.Param<param::StringParam>()->Value());
                vislib::net::IPAddress addr;
                if (addr.Lookup(baddr)) {
                    int bport = that->udpPortSlot.Param<param::IntParam>()->Value();
                    vislib::net::IPEndPoint ep(addr, bport);
                    that->udpInSocket.Send(ep, &datagram, sizeof(datagram));
                }
            }

            switch (datagram.msg) {
                case MSG_CONNECTTOSERVER: {
                    vislib::StringA mcn(that->clusterNameSlot.Param<param::StringParam>()->Value());
                    vislib::StringA rcn(datagram.payload.Strings.str1, datagram.payload.Strings.len1);
                    if (rcn.IsEmpty() || rcn.Equals(mcn)) {
                        vislib::StringA srv(datagram.payload.Strings.str2, datagram.payload.Strings.len2);
                        if (srv.Equals(that->conServerAddr)) {
                            vislib::sys::Log::DefaultLog.WriteInfo("Already connected to server \"%s\"", srv.PeekBuffer());
                        } else {
                            that->conServerAddr = srv;
                            vislib::sys::Log::DefaultLog.WriteInfo("Trying connect to new server \"%s\"", srv.PeekBuffer());
                            if (that->tcpChan != NULL) {
                                that->tcpChan->Close();
                                that->tcpChan->Release();
                                that->tcpChan = NULL;
                            }
                            if (that->tcpSan.IsRunning()) {
                                that->tcpSan.Terminate();
                                that->tcpSan.Join();
                            }

                            that->tcpChan = new vislib::net::TcpCommChannel(vislib::net::TcpCommChannel::FLAG_NODELAY);
                            try {
                                that->tcpChan->Connect(srv);
                                that->tcpSan.Start(that->tcpChan); // cast?
                                vislib::sys::Thread::Sleep(500);
                                vislib::sys::Log::DefaultLog.WriteInfo("TCP Connection started to \"%s\"", srv.PeekBuffer());
                                //vislib::net::SimpleMessage sm;
                                //sm.GetHeader().SetMessageID(MSG_HANDSHAKE_INIT);
                                //that->tcpChan->Send(sm, sm.GetMessageSize());

                            } catch(vislib::Exception ex) {
                                vislib::sys::Log::DefaultLog.WriteError("Failed to connect: %s\n", ex.GetMsgA());
                                that->tcpChan->Close();
                                that->tcpChan->Release();
                                that->tcpChan = NULL;
                                if (that->tcpSan.IsRunning()) {
                                    that->tcpSan.Terminate();
                                    that->tcpSan.Join();
                                }
                            } catch(...) {
                                vislib::sys::Log::DefaultLog.WriteError("Failed to connect: unexpected exception\n");
                                that->tcpChan->Close();
                                that->tcpChan->Release();
                                that->tcpChan = NULL;
                                if (that->tcpSan.IsRunning()) {
                                    that->tcpSan.Terminate();
                                    that->tcpSan.Join();
                                }
                            }
                        }
                    } else {
                        vislib::sys::Log::DefaultLog.WriteInfo("Server Connect Message for other cluster \"%s\" ignored\n", rcn.PeekBuffer());
                    }
                } break;
                case MSG_SHUTDOWN: {
                    vislib::StringA mcn(that->clusterNameSlot.Param<param::StringParam>()->Value());
                    vislib::StringA rcn(datagram.payload.Strings.str1, datagram.payload.Strings.len1);
                    if (rcn.IsEmpty() || rcn.Equals(mcn)) {
                        // somehow tell teh application to terminate ... :-/
                        vislib::sys::Log::DefaultLog.WriteInfo("Sending interrupt signal to frontend");
                        ::raise(SIGINT); // because I known console frontend will respond correctly
                    } else {
                        vislib::sys::Log::DefaultLog.WriteInfo("Shutdown Message for other cluster \"%s\" ignored\n", rcn.PeekBuffer());
                    }
                } break;
                default:
                    vislib::sys::Log::DefaultLog.WriteInfo("UDP Receiver: datagram %u received\n", datagram.msg);
            }

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
