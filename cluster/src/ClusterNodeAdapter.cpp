/*
 * ClusterNodeAdapter.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/ClusterNodeAdapter.h"

#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::net::cluster::ClusterNodeAdapter::~ClusterNodeAdapter
 */
vislib::net::cluster::ClusterNodeAdapter::~ClusterNodeAdapter(void) {
    // TODO: Implement
}


/*
 * vislib::net::cluster::ClusterNodeAdapter::Initialise
 */
void vislib::net::cluster::ClusterNodeAdapter::Initialise(
        sys::CmdLineProviderA& inOutCmdLine) {
}


/*
 * vislib::net::cluster::ClusterNodeAdapter::Initialise
 */
void vislib::net::cluster::ClusterNodeAdapter::Initialise(
        sys::CmdLineProviderW& inOutCmdLine) {
}


/*
 * vislib::net::cluster::ClusterNodeAdapter::ClusterNodeAdapter
 */
vislib::net::cluster::ClusterNodeAdapter::ClusterNodeAdapter(void) 
        : Super() {
}

/*
 * vislib::net::cluster::ClusterNodeAdapter::ClusterNodeAdapter
 */
vislib::net::cluster::ClusterNodeAdapter::ClusterNodeAdapter(
        const ClusterNodeAdapter& rhs) : Super(rhs) {
}


/*
 * vislib::net::cluster::ClusterNodeAdapter::sendToEachPeer
 */
SIZE_T vislib::net::cluster::ClusterNodeAdapter::sendToEachPeer(
        const BYTE *data, const SIZE_T cntData) {
    SendToPeerCtx context;
    context.Data = data;
    context.CntData = cntData;

    return this->forEachPeer(SendToPeerFunc, &context);
}


/*
 * vislib::net::cluster::ClusterNodeAdapter::operator =
 */
vislib::net::cluster::ClusterNodeAdapter& 
vislib::net::cluster::ClusterNodeAdapter::operator =(
        const ClusterNodeAdapter& rhs) {
    Super::operator=(rhs);
    return *this;
}


/*
 * vislib::net::cluster::ClusterNodeAdapter::SendToPeerFunc
 */
bool vislib::net::cluster::ClusterNodeAdapter::SendToPeerFunc(
        AbstractClusterNode *thisPtr, Socket& peerSocket, void *context) {
    SendToPeerCtx *ctx = static_cast<SendToPeerCtx *>(context);
    peerSocket.Send(ctx->Data, ctx->CntData, Socket::TIMEOUT_INFINITE, 0, true);
    return true;
}
