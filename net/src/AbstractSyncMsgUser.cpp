/*
 * AbstractSyncMsgUser.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractSyncMsgUser.h"

#include "the/trace.h"


/*
 * vislib::net::AbstractSyncMsgUser::~AbstractSyncMsgUser
 */
vislib::net::AbstractSyncMsgUser::~AbstractSyncMsgUser(void) {
    THE_STACK_TRACE;
    // Nothing to do.
}


/*
 * vislib::net::AbstractSyncMsgUser::AbstractSyncMsgUser
 */
vislib::net::AbstractSyncMsgUser::AbstractSyncMsgUser(void) {
    THE_STACK_TRACE;
    // Nothing to do.
}


/*
 * vislib::net::AbstractSyncMsgUser::receiveViaMsgBuffer
 */
const vislib::net::SimpleMessage& 
vislib::net::AbstractSyncMsgUser::receiveViaMsgBuffer(
        SmartRef<AbstractCommClientChannel> channel,
        const UINT timeout) {
    THE_STACK_TRACE;
    
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
    THE_STACK_TRACE;

    this->msgBuffer.GetHeader().SetMessageID(msgID);
    this->msgBuffer.SetBody(body, bodySize);
    
    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Sending message %u via buffer ...\n",
        this->msgBuffer.GetHeader().GetMessageID());
    channel->Send(static_cast<const void *>(this->msgBuffer),
        this->msgBuffer.GetMessageSize(), 
        timeout,
        true);
    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Message %u sent via buffer.\n", 
        this->msgBuffer.GetHeader().GetMessageID());
}
