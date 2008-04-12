/*
 * AbstractClusterNode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractClusterNode.h"

#include "vislib/MissingImplementationException.h"


/*
 * vislib::net::cluster::AbstractClusterNode::~AbstractClusterNode
 */
vislib::net::cluster::AbstractClusterNode::~AbstractClusterNode(void) {
}


/*
 * vislib::net::cluster::AbstractClusterNode::Initialise
 */
void vislib::net::cluster::AbstractClusterNode::Initialise(
        sys::CmdLineProviderA& inOutCmdLine) {
    throw MissingImplementationException("AbstractClusterNode::Initialise",
        __FILE__, __LINE__);
}


/*
 * vislib::net::cluster::AbstractClusterNode::Initialise
 */
void vislib::net::cluster::AbstractClusterNode::Initialise(
        sys::CmdLineProviderW& inOutCmdLine) {
    throw MissingImplementationException("AbstractClusterNode::Initialise",
        __FILE__, __LINE__);
}


/*
 * vislib::net::cluster::AbstractClusterNode::Run
 */
DWORD vislib::net::cluster::AbstractClusterNode::Run(void) {
    throw MissingImplementationException("AbstractClusterNode::Run",
        __FILE__, __LINE__);
}


/*
 * vislib::net::cluster::AbstractClusterNode::AbstractClusterNode
 */
vislib::net::cluster::AbstractClusterNode::AbstractClusterNode(void) {
}


/*
 * vislib::net::cluster::AbstractClusterNode::AbstractClusterNode
 */
vislib::net::cluster::AbstractClusterNode::AbstractClusterNode(
        const AbstractClusterNode& rhs) {
}


/*
 * vislib::net::cluster::AbstractClusterNode::countPeers
 */
SIZE_T vislib::net::cluster::AbstractClusterNode::countPeers(void) const {
    throw MissingImplementationException("AbstractClusterNode::countPeers",
        __FILE__, __LINE__);
}


/*
 * vislib::net::cluster::AbstractClusterNode::forEachPeer
 */
SIZE_T vislib::net::cluster::AbstractClusterNode::forEachPeer(
        ForeachPeerFunc func, void *context) {
    throw MissingImplementationException("AbstractClusterNode::forEachPeer",
        __FILE__, __LINE__);
}


/*
 * vislib::net::cluster::AbstractClusterNode::onMessageReceived
 */
void vislib::net::cluster::AbstractClusterNode::onMessageReceived(
        const Socket& src, const UINT msgId, const BYTE *body, 
        const SIZE_T cntBody) {
    // Do nothing. Implementing subclasses override this method if they are
    // interested in messages.
}


/*
 * vislib::net::cluster::AbstractClusterNode::sendToEachPeer
 */
SIZE_T vislib::net::cluster::AbstractClusterNode::sendToEachPeer(
        const BYTE *data, const SIZE_T cntData) {
    SendToPeerCtx context;
    context.Data = data;
    context.CntData = cntData;

    return this->forEachPeer(sendToPeerFunc, &context);
}


/*
 * vislib::net::cluster::AbstractClusterNode::operator =
 */
vislib::net::cluster::AbstractClusterNode& 
vislib::net::cluster::AbstractClusterNode::operator =(
        const AbstractClusterNode& rhs) {
    return *this;
}


/*
 * vislib::net::cluster::AbstractClusterNode::sendToPeerFunc
 */
bool vislib::net::cluster::AbstractClusterNode::sendToPeerFunc(
        AbstractClusterNode *thisPtr, const PeerIdentifier& peerId,
        Socket& peerSocket, void *context) {
    SendToPeerCtx *ctx = static_cast<SendToPeerCtx *>(context);
    peerSocket.Send(ctx->Data, ctx->CntData, Socket::TIMEOUT_INFINITE, 0, true);
    return true;
}
