/*
 * ClusterControllerClient.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/ClusterControllerClient.h"
#include "cluster/CallRegisterAtController.h"
#include "param/StringParam.h"
#include "vislib/Log.h"
#include "vislib/IPEndPoint.h"
#include "vislib/NetworkInformation.h"

using namespace megamol::core;


/*
 * cluster::ClusterControllerClient::ClusterControllerClient
 */
cluster::ClusterControllerClient::ClusterControllerClient(void)
        : registerSlot("register", "The slot to register the client at the controller"),
        ctrlCommAddressSlot("ctrlCommAddress", "Specifies the address string (including port) used to communicate control commands"),
        ctrlr(NULL), ctrlChannel(NULL) {

    this->registerSlot.SetCompatibleCall<CallRegisterAtControllerDescription>();
    this->registerSlot.AddListener(this);
    // must be published in derived class to avoid diamond-inheritance

    this->ctrlCommAddressSlot << new param::StringParam("");
    this->ctrlCommAddressSlot.SetUpdateCallback(&ClusterControllerClient::onCtrlCommAddressChanged);
    // must be published in derived class to avoid diamond-inheritance

}


/*
 * cluster::ClusterControllerClient::~ClusterControllerClient
 */
cluster::ClusterControllerClient::~ClusterControllerClient(void) {
    this->ctrlr = NULL; // DO NOT DELETE
    this->stopCtrlComm(); // just to be dead sure
}


/*
 * cluster::ClusterControllerClient::OnClusterAvailable
 */
void cluster::ClusterControllerClient::OnClusterAvailable(void) {
    // intentionally empty
}


/*
 * cluster::ClusterControllerClient::OnClusterUnavailable
 */
void cluster::ClusterControllerClient::OnClusterUnavailable(void) {
    // intentionally empty
}


/*
 * cluster::ClusterControllerClient::OnNodeFound
 */
void cluster::ClusterControllerClient::OnNodeFound(
        const cluster::ClusterController::PeerHandle& hPeer) {
    // intentionally empty
}


/*
 * cluster::ClusterControllerClient::OnNodeLost
 */
void cluster::ClusterControllerClient::OnNodeLost(
        const cluster::ClusterController::PeerHandle& hPeer) {
    // intentionally empty
}


/*
 * cluster::ClusterControllerClient::SendUserMsg
 */
void cluster::ClusterControllerClient::SendUserMsg(
        const UINT32 msgType, const BYTE *msgBody, const SIZE_T msgSize) {
    if (this->ctrlr == NULL) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Cannot 'SendUserMsg[%d]': ClusterControllerClient is not connected to the controller\n",
            __LINE__);
        return;
    }
    this->ctrlr->SendUserMsg(msgType, msgBody, msgSize);
}


/*
 * cluster::ClusterControllerClient::SendUserMsg
 */
void cluster::ClusterControllerClient::SendUserMsg(
        const cluster::ClusterController::PeerHandle& hPeer,
        const UINT32 msgType, const BYTE *msgBody, const SIZE_T msgSize) {
    if (this->ctrlr == NULL) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Cannot 'SendUserMsg[%d]': ClusterControllerClient is not connected to the controller\n",
            __LINE__);
        return;
    }
    this->ctrlr->SendUserMsg(hPeer, msgType, msgBody, msgSize);
}


/*
 * cluster::ClusterControllerClient::OnConnect
 */
void cluster::ClusterControllerClient::OnConnect(AbstractSlot& slot) {
    if (&slot != &this->registerSlot) return;
    CallRegisterAtController *crac
        = this->registerSlot.CallAs<CallRegisterAtController>();
    if (crac != NULL) {
        crac->SetClient(this);
        (*crac)(CallRegisterAtController::CALL_REGISTER);
    }
}


/*
 * cluster::ClusterControllerClient::OnDisconnect
 */
void cluster::ClusterControllerClient::OnDisconnect(AbstractSlot& slot) {
    if (&slot != &this->registerSlot) return;
    CallRegisterAtController *crac
        = this->registerSlot.CallAs<CallRegisterAtController>();
    if (crac != NULL) {
        crac->SetClient(this);
        (*crac)(CallRegisterAtController::CALL_UNREGISTER);
    }
}


/*
 * cluster::ClusterControllerClient::stopCtrlComm
 */
void cluster::ClusterControllerClient::stopCtrlComm(void) {
    if (!this->ctrlChannel.IsNull()) {
        try {
            this->ctrlChannel->Close();
            this->ctrlChannel.Release();
        } catch(...) {
            this->ctrlChannel.Release();
        }
        this->ctrlChannel = NULL;
    }
}


/*
 * cluster::ClusterControllerClient::startCtrlCommServer
 */
vislib::SmartRef<vislib::net::TcpCommChannel> cluster::ClusterControllerClient::startCtrlCommServer(void) {
    this->stopCtrlComm();
    this->ctrlChannel = new vislib::net::TcpCommChannel();
    // rest has to be done in derived server/master class
    return this->ctrlChannel;
}


/*
 * cluster::ClusterControllerClient::startCtrlCommClient
 */
vislib::net::AbstractCommChannel * cluster::ClusterControllerClient::startCtrlCommClient(const vislib::net::IPEndPoint& address) {
    try {
        this->stopCtrlComm();
        this->ctrlChannel = new vislib::net::TcpCommChannel();
        this->ctrlChannel->Connect(address);
        return dynamic_cast<vislib::net::AbstractCommChannel *>(this->ctrlChannel.operator ->());

    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to connect to server: %s\n", ex.GetMsgA());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to connect to server: unexpected exception\n");
    }
    return NULL;
}


/*
 * cluster::ClusterControllerClient::isCtrlCommConnectedTo
 */
bool cluster::ClusterControllerClient::isCtrlCommConnectedTo(const char *address) const {
    vislib::net::IPEndPoint ep1, ep2;
    if (address == NULL) return false;
    if (this->ctrlChannel.IsNull()) return false;
    vislib::net::NetworkInformation::GuessRemoteEndPoint(ep1, address);
    vislib::net::NetworkInformation::GuessRemoteEndPoint(ep2, this->ctrlCommAddressSlot.Param<param::StringParam>()->Value());
    return (ep1 == ep2);
}


/*
 * cluster::ClusterControllerClient::onCtrlCommAddressChanged
 */
bool cluster::ClusterControllerClient::onCtrlCommAddressChanged(param::ParamSlot& slot) {
    ASSERT(&slot == &this->ctrlCommAddressSlot);
    this->OnCtrlCommAddressChanged(slot.Param<param::StringParam>()->Value());
    return true;
}
