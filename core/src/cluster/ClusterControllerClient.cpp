/*
 * ClusterControllerClient.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/cluster/ClusterControllerClient.h"
#include "mmcore/cluster/CallRegisterAtController.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/net/IPEndPoint.h"
#include "vislib/net/NetworkInformation.h"

using namespace megamol::core;


/*
 * cluster::ClusterControllerClient::ClusterControllerClient
 */
cluster::ClusterControllerClient::ClusterControllerClient(void)
        : AbstractSlot::Listener()
        , vislib::Listenable<ClusterControllerClient>()
        , registerSlot("register", "The slot to register the client at the controller")
        , ctrlr(NULL) {

    this->registerSlot.SetCompatibleCall<CallRegisterAtControllerDescription>();
    this->registerSlot.AddListener(this);
    // must be published in derived class to avoid diamond-inheritance
}


/*
 * cluster::ClusterControllerClient::~ClusterControllerClient
 */
cluster::ClusterControllerClient::~ClusterControllerClient(void) {}


/*
 * cluster::ClusterControllerClient::SendUserMsg
 */
void cluster::ClusterControllerClient::SendUserMsg(const UINT32 msgType, const BYTE* msgBody, const SIZE_T msgSize) {
    if (this->ctrlr == NULL) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Cannot 'SendUserMsg[%d]': ClusterControllerClient is not connected to the controller\n", __LINE__);
        return;
    }
    this->ctrlr->SendUserMsg(msgType, msgBody, msgSize);
}


/*
 * cluster::ClusterControllerClient::SendUserMsg
 */
void cluster::ClusterControllerClient::SendUserMsg(const cluster::ClusterController::PeerHandle& hPeer,
    const UINT32 msgType, const BYTE* msgBody, const SIZE_T msgSize) {
    if (this->ctrlr == NULL) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Cannot 'SendUserMsg[%d]': ClusterControllerClient is not connected to the controller\n", __LINE__);
        return;
    }
    this->ctrlr->SendUserMsg(hPeer, msgType, msgBody, msgSize);
}


/*
 * cluster::ClusterControllerClient::OnConnect
 */
void cluster::ClusterControllerClient::OnConnect(AbstractSlot& slot) {
    if (&slot != &this->registerSlot)
        return;
    CallRegisterAtController* crac = this->registerSlot.CallAs<CallRegisterAtController>();
    if (crac != NULL) {
        crac->SetClient(this);
        (*crac)(CallRegisterAtController::CALL_REGISTER);
    }
}


/*
 * cluster::ClusterControllerClient::OnDisconnect
 */
void cluster::ClusterControllerClient::OnDisconnect(AbstractSlot& slot) {
    if (&slot != &this->registerSlot)
        return;
    CallRegisterAtController* crac = this->registerSlot.CallAs<CallRegisterAtController>();
    if (crac != NULL) {
        crac->SetClient(this);
        (*crac)(CallRegisterAtController::CALL_UNREGISTER);
    }
}


/*
 * cluster::ClusterControllerClient::onClusterAvailable
 */
void cluster::ClusterControllerClient::onClusterAvailable(void) {
    ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener* l = dynamic_cast<Listener*>(iter.Next());
        if (l == NULL)
            continue;
        l->OnClusterDiscoveryAvailable(*this);
    }
}


/*
 * cluster::ClusterControllerClient::onClusterUnavailable
 */
void cluster::ClusterControllerClient::onClusterUnavailable(void) {
    ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener* l = dynamic_cast<Listener*>(iter.Next());
        if (l == NULL)
            continue;
        l->OnClusterDiscoveryUnavailable(*this);
    }
}


/*
 * cluster::ClusterControllerClient::onNodeFound
 */
void cluster::ClusterControllerClient::onNodeFound(const cluster::ClusterController::PeerHandle& hPeer) {
    ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener* l = dynamic_cast<Listener*>(iter.Next());
        if (l == NULL)
            continue;
        l->OnClusterNodeFound(*this, hPeer);
    }
}


/*
 * cluster::ClusterControllerClient::onNodeLost
 */
void cluster::ClusterControllerClient::onNodeLost(const cluster::ClusterController::PeerHandle& hPeer) {
    ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener* l = dynamic_cast<Listener*>(iter.Next());
        if (l == NULL)
            continue;
        l->OnClusterNodeLost(*this, hPeer);
    }
}


/*
 * cluster::ClusterControllerClient::onUserMsg
 */
void cluster::ClusterControllerClient::onUserMsg(
    const ClusterController::PeerHandle& hPeer, bool isClusterMember, const UINT32 msgType, const BYTE* msgBody) {
    ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener* l = dynamic_cast<Listener*>(iter.Next());
        if (l == NULL)
            continue;
        l->OnClusterUserMessage(*this, hPeer, isClusterMember, msgType, msgBody);
    }
}
