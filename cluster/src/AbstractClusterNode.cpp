/*
 * AbstractClusterNode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractClusterNode.h"

#include "vislib/MissingImplementationException.h"
#include "vislib/Trace.h"

#include "messagereceiver.h"


/*
 * vislib::net::cluster::AbstractClusterNode::~AbstractClusterNode
 */
vislib::net::cluster::AbstractClusterNode::~AbstractClusterNode(void) {
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
 * vislib::net::cluster::AbstractClusterNode::onMessageReceived
 */
void vislib::net::cluster::AbstractClusterNode::onMessageReceived(
        const Socket& src, const UINT msgId, const BYTE *body, 
        const SIZE_T cntBody) {
    // Do nothing. Implementing subclasses override this method if they are
    // interested in messages.
}


/*
 * vislib::net::cluster::AbstractClusterNode::onMessageReceiverExiting
 */
void vislib::net::cluster::AbstractClusterNode::onMessageReceiverExiting(
        vislib::net::Socket& socket, PReceiveMessagesCtx rmc) {
    TRACE(Trace::LEVEL_VL_INFO, "AbstractClusterNode::onMessageReceiverExiting "
        "releasing receive context ...\n");
    FreeRecvMsgCtx(rmc);
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
