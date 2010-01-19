/*
 * ClusterControllerClient.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ClusterControllerClient.h"

using namespace megamol::core;


/*
 * special::ClusterControllerClient::ClusterControllerClient
 */
special::ClusterControllerClient::ClusterControllerClient(void)
        : ctrlr(NULL) {
    // intentionally empty
}


/*
 * special::ClusterControllerClient::~ClusterControllerClient
 */
special::ClusterControllerClient::~ClusterControllerClient(void) {
    this->ctrlr = NULL; // DO NOT DELETE
}


/*
 * special::ClusterControllerClient::OnClusterAvailable
 */
void special::ClusterControllerClient::OnClusterAvailable(void) {
    // intentionally empty
}


/*
 * special::ClusterControllerClient::OnClusterUnavailable
 */
void special::ClusterControllerClient::OnClusterUnavailable(void) {
    // intentionally empty
}


/*
 * special::ClusterControllerClient::OnNodeFound
 */
void special::ClusterControllerClient::OnNodeFound(
        const special::ClusterController::PeerHandle& hPeer) {
    // intentionally empty
}


/*
 * special::ClusterControllerClient::OnNodeLost
 */
void special::ClusterControllerClient::OnNodeLost(
        const special::ClusterController::PeerHandle& hPeer) {
    // intentionally empty
}


/*
 * special::ClusterControllerClient::OnUserMsg
 */
void special::ClusterControllerClient::OnUserMsg(
        const special::ClusterController::PeerHandle& hPeer,
        const UINT32 msgType, const BYTE *msgBody) {
    // intentionally empty
}


/*
 * special::ClusterControllerClient::SendUserMsg
 */
void special::ClusterControllerClient::SendUserMsg(
        const UINT32 msgType, const BYTE *msgBody, const SIZE_T msgSize) {

    // TODO: Implement

}


/*
 * special::ClusterControllerClient::SendUserMsg
 */
void special::ClusterControllerClient::SendUserMsg(
        const special::ClusterController::PeerHandle& hPeer,
        const UINT32 msgType, const BYTE *msgBody, const SIZE_T msgSize) {

    // TODO: Implement

}
