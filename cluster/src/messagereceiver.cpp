/*
 * messagereceiver.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "messagereceiver.h"

#include "vislib/assert.h"
#include "vislib/clustermessages.h"
#include "vislib/IllegalParamException.h"
#include "vislib/memutils.h"
#include "vislib/RawStorage.h"
#include "vislib/SocketException.h"
#include "vislib/Trace.h"


/*
 * vislib::net::cluster::AllocateRecvMsgCtx
 */
vislib::net::cluster::ReceiveMessagesCtx *
vislib::net::cluster::AllocateRecvMsgCtx(AbstractClusterNode *receiver,
        Socket *socket) {
    ReceiveMessagesCtx *retval = new ReceiveMessagesCtx;
    retval->Receiver = receiver;
    retval->Socket = socket;

    return retval;
}


/*
 * vislib::net::cluster::FreeRecvMsgCtx
 */
void vislib::net::cluster::FreeRecvMsgCtx(ReceiveMessagesCtx *& ctx) {
    SAFE_DELETE(ctx);
}


/*
 * vislib::net::cluster::ReceiveMessages
 */
DWORD vislib::net::cluster::ReceiveMessages(void *receiveMessagesCtx) {
    ReceiveMessagesCtx *ctx = static_cast<ReceiveMessagesCtx *>(
        receiveMessagesCtx);            // Receive context.
    const MessageHeader *msgHdr = NULL; // Pointer to header area of 'recvBuf'.
    const BlockHeader *blkHdr = NULL;   // Pointer to a block header.
    SIZE_T msgSize = 0;                 // Total message size (header + body).
    RawStorage recvBuf;                 // Receives data from network.
    DWORD retval = 0;                   // Function return value.
    AbstractClusterNode::PeerIdentifier peerId; // Peer address of socket.

    /* Sanity checks. */
    ASSERT(ctx != NULL);
    ASSERT(ctx->Receiver != NULL);
    ASSERT(ctx->Socket != NULL);
    if ((ctx == NULL) || (ctx->Receiver == NULL) || (ctx->Socket == NULL)) {
        throw IllegalParamException("receiveMessagesCtx", __FILE__, __LINE__);
    }

    try {
        peerId = ctx->Socket->GetPeerEndPoint();
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Message receiver thread could not "
            "retrieve identifier of peer node: %s\n", e.GetMsgA());
    }

    try {

        VLTRACE(Trace::LEVEL_VL_INFO, "The cluster message receiver thread is "
            "starting ...\n");

        while (true) {

            /* Receive a message header into 'recvBuf'. */
            recvBuf.AssertSize(sizeof(MessageHeader));
            msgHdr = recvBuf.As<MessageHeader>();
            if (ctx->Socket->Receive(static_cast<void *>(recvBuf), 
                    sizeof(MessageHeader), Socket::TIMEOUT_INFINITE, 
                    0, true) == 0) {
                VLTRACE(Trace::LEVEL_VL_INFO, "vislib::net::cluster::"
                    "ReceiveMessages exits because of graceful disconnect.\n");
            }

            /* Sanity check. */
            ASSERT(recvBuf.GetSize() >= sizeof(MessageHeader));
            //ASSERT(msgHdr->MagicNumber == MAGIC_NUMBER);
            if (msgHdr->MagicNumber != MAGIC_NUMBER) {
                VLTRACE(Trace::LEVEL_WARN, "Discarding data packet without "
                    "valid magic number. Expected %u, but received %u.\n",
                    MAGIC_NUMBER, msgHdr->MagicNumber);
                break;
            }

            /* Receive the rest of the message after the header in 'recvBuf'. */
            msgSize = sizeof(MessageHeader) + msgHdr->Header.BlockLength;
            recvBuf.AssertSize(msgSize, true);
            msgHdr = recvBuf.As<MessageHeader>();
            if (ctx->Socket->Receive(recvBuf.As<BYTE>() + sizeof(MessageHeader),
                    msgHdr->Header.BlockLength, Socket::TIMEOUT_INFINITE, 0, 
                    true) == 0) {
                VLTRACE(Trace::LEVEL_VL_INFO, "vislib::net::cluster::"
                    "ReceiveMessages exits because of graceful disconnect.\n");
            }

            /* Call the handler method to process the message. */
            if (msgHdr->Header.BlockId == MSGID_MULTIPLE) {
                /* Received a compound message, so split it. */
                VLTRACE(Trace::LEVEL_VL_INFO, "Splitting compond message ...\n");
                INT remBody = static_cast<INT>(msgHdr->Header.BlockLength);
                const BYTE *d = recvBuf.As<BYTE>() + sizeof(MessageHeader);

                while (remBody > 0) {
                    blkHdr = reinterpret_cast<const BlockHeader *>(d);
                    const BYTE *body = d + sizeof(BlockHeader);

                    VLTRACE(Trace::LEVEL_VL_INFO, "Received message %u.\n", 
                        blkHdr->BlockId);
                    ctx->Receiver->onMessageReceived(*ctx->Socket,
                        blkHdr->BlockId, 
                        (blkHdr->BlockLength > 0) ? body : NULL,
                        blkHdr->BlockLength);

                    d += (sizeof(BlockHeader)+ blkHdr->BlockLength);
                    remBody -= (sizeof(BlockHeader)+ blkHdr->BlockLength);
                }

            } else {
                /* Receive single message. */
                const BlockHeader *blkHdr = &msgHdr->Header;
                const BYTE *body = recvBuf.As<BYTE>() + sizeof(MessageHeader);

                VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Cluster service "
                    "received message %u.\n", blkHdr->BlockId);
                ctx->Receiver->onMessageReceived(*ctx->Socket,
                    blkHdr->BlockId, 
                    (blkHdr->BlockLength > 0) ? body : NULL,
                    blkHdr->BlockLength);
            }
        } /* end while (true) */
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "vislib::net::cluster::ReceiveMessages "
            "exits because of communication error: %s\n", e.GetMsgA());
        // TODO: Remove HOTFIX
        //ctx->Receiver->onCommunicationError(peerId, 
        //    AbstractClusterNode::RECEIVE_COMMUNICATION_ERROR, e);
        retval = e.GetErrorCode();
    } catch (...) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Unexpected exception caught in "
            "vislib::net::cluster::ReceiveMessages.\n");
        retval = -1;
    }

    ctx->Receiver->onMessageReceiverExiting(*ctx->Socket, ctx);
    return retval;
}
