/*
 * AbstractSyncMsgUser.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractSyncMsgUser.h"

#include "vislib/Trace.h"


/*
 * vislib::net::AbstractSyncMsgUser::~AbstractSyncMsgUser
 */
vislib::net::AbstractSyncMsgUser::~AbstractSyncMsgUser(void) {
    VLSTACKTRACE("AbstractSyncMsgUser::~AbstractSyncMsgUser", __FILE__, 
        __LINE__);
    // Nothing to do.
}


/*
 * vislib::net::AbstractSyncMsgUser::AbstractSyncMsgUser
 */
vislib::net::AbstractSyncMsgUser::AbstractSyncMsgUser(void) {
    VLSTACKTRACE("AbstractSyncMsgUser::AbstractSyncMsgUser", __FILE__, 
        __LINE__);
    // Nothing to do.
}


/*
 * vislib::net::AbstractSyncMsgUser::receiveViaMsgBuffer
 */
const vislib::net::SimpleMessage& 
vislib::net::AbstractSyncMsgUser::receiveViaMsgBuffer(
        SmartRef<AbstractCommClientChannel> channel,
        const UINT timeout) {
    VLSTACKTRACE("AbstractSyncMsgUser::receiveViaMsgBuffer", __FILE__, 
        __LINE__);
    
    SimpleMessageSize headerSize = this->msgBuffer.GetHeader().GetHeaderSize();
    channel->Receive(static_cast<void *>(this->msgBuffer), headerSize, timeout,
        true);
    SimpleMessageSize bodySize = this->msgBuffer.GetHeader().GetBodySize();

    if (bodySize > 0) {
        this->msgBuffer.AssertBodySize();
        channel->Receive(this->msgBuffer.GetBody(), bodySize, timeout, true);
    }

    return this->msgBuffer;
}

/*
 * vislib::net::AbstractSyncMsgUser::sendViaMsgBuffer
 */
void vislib::net::AbstractSyncMsgUser::sendViaMsgBuffer(
        SmartRef<AbstractCommClientChannel> channel,
        const SimpleMessageID msgID, 
        const void *body, 
        const unsigned int bodySize,
        const UINT timeout) {
    VLSTACKTRACE("AbstractSyncMsgUser::sendViaMsgBuffer", __FILE__, __LINE__);

    this->msgBuffer.GetHeader().SetMessageID(msgID);
    this->msgBuffer.SetBody(body, bodySize);
    
    VLTRACE(Trace::LEVEL_VL_VERBOSE, "Sending message %u via buffer ...\n",
        this->msgBuffer.GetHeader().GetMessageID());
    channel->Send(static_cast<const void *>(this->msgBuffer),
        this->msgBuffer.GetMessageSize(), 
        timeout,
        true);
    VLTRACE(Trace::LEVEL_VL_VERBOSE, "Message %u sent via buffer.\n", 
        this->msgBuffer.GetHeader().GetMessageID());
}
