/*
 * ClusterControllerClient.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/ClusterControllerClient.h"
#include "cluster/CallRegisterAtController.h"
#include "vislib/Log.h"

using namespace megamol::core;


/*
 * cluster::ClusterControllerClient::ClusterControllerClient
 */
cluster::ClusterControllerClient::ClusterControllerClient(void)
        : registerSlot("register", "The slot to register the client at the controller"),
        ctrlr(NULL) {

    this->registerSlot.SetCompatibleCall<CallRegisterAtControllerDescription>();
    this->registerSlot.AddListener(this);
    // must be published in derived class to avoid diamond-inheritance
}


/*
 * cluster::ClusterControllerClient::~ClusterControllerClient
 */
cluster::ClusterControllerClient::~ClusterControllerClient(void) {
    this->ctrlr = NULL; // DO NOT DELETE
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
 * cluster::ClusterControllerClient::OnUserMsg
 */
void cluster::ClusterControllerClient::OnUserMsg(
        const cluster::ClusterController::PeerHandle& hPeer,
        const UINT32 msgType, const BYTE *msgBody) {
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
