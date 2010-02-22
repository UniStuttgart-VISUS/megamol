/*
 * ClusterControllerClient.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/ClusterControllerClient.h"

using namespace megamol::core;


/*
 * cluster::ClusterControllerClient::ClusterControllerClient
 */
cluster::ClusterControllerClient::ClusterControllerClient(void)
        : ctrlr(NULL) {
    // intentionally empty
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

    // TODO: Implement

}


/*
 * cluster::ClusterControllerClient::SendUserMsg
 */
void cluster::ClusterControllerClient::SendUserMsg(
        const cluster::ClusterController::PeerHandle& hPeer,
        const UINT32 msgType, const BYTE *msgBody, const SIZE_T msgSize) {

    // TODO: Implement

}
