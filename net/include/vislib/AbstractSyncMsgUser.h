/*
 * AbstractSyncMsgUser.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTSYNCMSGUSER_H_INCLUDED
#define VISLIB_ABSTRACTSYNCMSGUSER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/IPAddress.h"           // Must be first!
#include "vislib/AbstractCommChannel.h"
#include "vislib/SimpleMessage.h"
#include "vislib/SmartRef.h"
#include "vislib/StackTrace.h"
#include "vislib/TcpCommChannel.h"
#include "vislib/UnexpectedMessageException.h"


namespace vislib {
namespace net {


    /**
     * This class provides an inheritable implementation for sending and 
     * receiving messages synchronously. The implementation provides storage 
     * for a single message that is used for both, sending and receiving.
     *
     * Please note, that the methods provided by this class are 
     * NOT THREAD-SAFE AT ALL!
     *
     * Please note that this class is not Ludewig-conformant ...
     */
    class AbstractSyncMsgUser {

    public:

        /** Dtor. */
        virtual ~AbstractSyncMsgUser(void);

    protected:

        /** Ctor. */
        AbstractSyncMsgUser(void);

        /** 
         * Copy ctor, which does not copy the actual message data. Each copy
         * will have a new buffer.
         *
         * @param rhs The object to be cloned.
         */
        inline AbstractSyncMsgUser(const AbstractSyncMsgUser& rhs) {
            VLSTACKTRACE("AbstractSyncMsgUser::AbstractSyncMsgUser", __FILE__, 
                __LINE__);
            *this = rhs;
        }

        /**
         * Perform a message round-trip using the message buffer. This includes
         * sending a message with ID 'requestMsgID' and a body containing
         * 'requestBody' via 'channel' and afterwards receiving a simple
         * SimpleMessage.
         *
         * @param channel         The communication channel to use for the 
         *                        round-trip.
         * @param requestMsgID    The ID of the request message.
         * @param requestBody     The request data, which must be 
         *                        'requestBodySize' bytes.
         * @param requestBodySize Size of the data in bytes designated 
         *                        'requestBody'.
         * @param timeout         Timeout for send and receive operations. If
         *                        each operation cannot be fulfiled in the
         *                        given amount of time, the method fails with
         *                        an exception.
         *
         * @return A reference to the response message received after the 
         *         request was sent successfully. This will be valid until the 
         *         next message is sent or received.
         *
         * @throws Exception Or derived in case of a communication error.
         */
        inline const SimpleMessage& requestViaMsgBuffer(
                SmartRef<AbstractCommClientChannel> channel,
                const SimpleMessageID requestMsgID, 
                const void *requestBody, 
                const unsigned int requestBodySize,
                const UINT timeout = AbstractCommChannel::TIMEOUT_INFINITE) {
            VLSTACKTRACE("AbstractSyncMsgUser::requestViaMsgBuffer", __FILE__,
                __LINE__);
            this->sendViaMsgBuffer(channel, requestMsgID, requestBody, 
                requestBodySize, timeout);
            return this->receiveViaMsgBuffer(channel, timeout);
        }

        /**
         * Perform a message round-trip using the message buffer. This includes
         * sending a message with ID 'requestMsgID' and a body containing
         * 'requestBody' via 'channel' and afterwards receiving a simple
         * SimpleMessage.
         *
         * @param channel         The communication channel to use for the 
         *                        round-trip.
         * @param requestMsgID    The ID of the request message.
         * @param requestBody     The request data, which must be 
         *                        'requestBodySize' bytes.
         * @param requestBodySize Size of the data in bytes designated 
         *                        'requestBody'.
         * @param timeout         Timeout for send and receive operations. If
         *                        each operation cannot be fulfiled in the
         *                        given amount of time, the method fails with
         *                        an exception.
         *
         * @return A reference to the response message received after the 
         *         request was sent successfully. This will be valid until the 
         *         next message is sent or received.
         *
         * @throws Exception Or derived in case of a communication error.
         */
        inline const SimpleMessage& requestViaMsgBuffer(
                SmartRef<TcpCommChannel> channel,
                const SimpleMessageID requestMsgID, 
                const void *requestBody, 
                const unsigned int requestBodySize,
                const UINT timeout = AbstractCommChannel::TIMEOUT_INFINITE) {
            VLSTACKTRACE("AbstractSyncMsgUser::requestViaMsgBuffer", __FILE__,
                __LINE__);
            return this->requestViaMsgBuffer(
                channel.DynamicCast<vislib::net::AbstractCommClientChannel>(), 
                requestMsgID, requestBody, requestBodySize, timeout);
        }

        /**
         * Perform a message round-trip using the message buffer. This includes
         * sending a message with ID 'requestMsgID' and a body containing
         * 'requestBody' via 'channel' and afterwards receiving a simple
         * SimpleMessage.
         *
         * The user is responsible for choosing only valid body data for 'T', 
         * i. e. structures which do not contain pointers etc. and therefore
         * can be sent directly.
         *
         * @param channel         The communication channel to use for the 
         *                        round-trip.
         * @param requestMsgID    The ID of the request message.
         * @param requestBody     The request data.
         * @param timeout         Timeout for send and receive operations. If
         *                        each operation cannot be fulfiled in the
         *                        given amount of time, the method fails with
         *                        an exception.
         *
         * @return A reference to the response message received after the 
         *         request was sent successfully. This will be valid until the 
         *         next message is sent or received.
         *
         * @throws Exception Or derived in case of a communication error.
         */
        template<class T> inline const SimpleMessage& requestViaMsgBuffer(
                SmartRef<AbstractCommClientChannel> channel,
                const SimpleMessageID requestMsgID, 
                const T& requestBody,
                const UINT timeout = AbstractCommChannel::TIMEOUT_INFINITE) {
            VLSTACKTRACE("AbstractSyncMsgUser::requestViaMsgBuffer", __FILE__, 
                __LINE__);
            return this->requestViaMsgBuffer(channel, requestMsgID, 
                &requestBody, sizeof(T), timeout);
        }

        /**
         * Perform a message round-trip using the message buffer. This includes
         * sending a message with ID 'requestMsgID' and a body containing
         * 'requestBody' via 'channel' and afterwards receiving a simple
         * SimpleMessage.
         *
         * The user is responsible for choosing only valid body data for 'T', 
         * i. e. structures which do not contain pointers etc. and therefore
         * can be sent directly.
         *
         * @param channel         The communication channel to use for the 
         *                        round-trip.
         * @param requestMsgID    The ID of the request message.
         * @param requestBody     The request data.
         * @param timeout         Timeout for send and receive operations. If
         *                        each operation cannot be fulfiled in the
         *                        given amount of time, the method fails with
         *                        an exception.
         *
         * @return A reference to the response message received after the 
         *         request was sent successfully. This will be valid until the 
         *         next message is sent or received.
         *
         * @throws Exception Or derived in case of a communication error.
         */
        template<class T> inline const SimpleMessage& requestViaMsgBuffer(
                SmartRef<TcpCommChannel> channel,
                const SimpleMessageID requestMsgID, 
                const T& requestBody,
                const UINT timeout = AbstractCommChannel::TIMEOUT_INFINITE) {
            VLSTACKTRACE("AbstractSyncMsgUser::requestViaMsgBuffer", __FILE__, 
                __LINE__);
            return this->requestViaMsgBuffer(channel, requestMsgID, 
                &requestBody, sizeof(T), timeout);
        }

        /**
         * Perform a message round-trip using the message buffer. This includes
         * sending a message with ID 'requestMsgID' and a body containing
         * 'requestBody' via 'channel' and afterwards receiving a simple
         * SimpleMessage.
         *
         * The user is responsible for choosing the correct template parameters
         * explicitly. These are:
         * I:   The request structure, which must be serialisable by memcpy.
         * Iid: The message ID for the request. This will be packed into the
         *      message header.
         * O:   The response structure, which must be serialisable by memcpy.
         * Oid: The message ID the response must have. If another ID is 
         *      received, the operation will fail.
         *
         * @param channel            The communication channel to use for the
         *                           round-trip.
         * @param requestBody        The request data.
         * @param outAdditionalBytes If not NULL, the method writes the number
         *                           of bytes in the response that is not part
                                     of 'O' into this variable.
         * @param timeout            Timeout for send and receive operations. 
         *                           If each operation cannot be fulfiled in
         *                           the given amount of time, the method 
         *                           fails with an exception.
         * 
         * @return A reference to the response message received after the 
         *         request was sent successfully. This will be valid until the 
         *         next message is sent or received.
         *
         * @throws Exception Or derived in case of a communication error.
         * @throws UnexpectedMessageException In case the response received does
         *                                    not have the message ID 'Oid'.
         */
        template<class I, SimpleMessageID Iid, class O, SimpleMessageID Oid>
        const O *requestViaMsgBuffer(SmartRef<AbstractCommClientChannel> channel,
            const I& requestBody,
            SIZE_T *outAdditionalBytes = NULL,
            const UINT timeout = AbstractCommChannel::TIMEOUT_INFINITE);

        /**
         * Perform a message round-trip using the message buffer. This includes
         * sending a message with ID 'requestMsgID' and a body containing
         * 'requestBody' via 'channel' and afterwards receiving a simple
         * SimpleMessage.
         *
         * The user is responsible for choosing the correct template parameters
         * explicitly. These are:
         * I:   The request structure, which must be serialisable by memcpy.
         * Iid: The message ID for the request. This will be packed into the
         *      message header.
         * O:   The response structure, which must be serialisable by memcpy.
         * Oid: The message ID the response must have. If another ID is 
         *      received, the operation will fail.
         *
         * @param channel            The communication channel to use for the
         *                           round-trip.
         * @param requestBody        The request data.
         * @param outAdditionalBytes If not NULL, the method writes the number
         *                           of bytes in the response that is not part
                                     of 'O' into this variable.
         * @param timeout            Timeout for send and receive operations. 
         *                           If each operation cannot be fulfiled in
         *                           the given amount of time, the method 
         *                           fails with an exception.
         *
         * @return A reference to the response message received after the 
         *         request was sent successfully. This will be valid until the 
         *         next message is sent or received.
         *
         * @throws Exception Or derived in case of a communication error.
         * @throws UnexpectedMessageException In case the response received does
         *                                    not have the message ID 'Oid'.
         */
        template<class I, SimpleMessageID Iid, class O, SimpleMessageID Oid>
        inline const O *requestViaMsgBuffer(
                SmartRef<TcpCommChannel> channel,
                const I& requestBody,
                SIZE_T *outAdditionalBytes = NULL,
                const UINT timeout = AbstractCommChannel::TIMEOUT_INFINITE) {
            VLSTACKTRACE("AbstractSyncMsgUser::requestViaMsgBuffer", __FILE__, 
                __LINE__);
            return this->requestViaMsgBuffer<I, Iid, O, Oid>(
                channel.DynamicCast<AbstractCommClientChannel>(), 
                requestBody, outAdditionalBytes, timeout);
        }

        /**
         * Receive a SimpleMessage into the local message buffer from the given 
         * 'channel'.
         *
         * @param channel The channel to receive the message from
         * @param timeout A timeout for receiving the message. If the message
         *                cannot be received in time, the operation will fail
         *                with an exception.
         *
         * @return A reference to the message received. This will be valid 
         *         until the next message is sent or received.
         *
         * @throws Exception Or derived in case of a communication error.
         */
        // TODO: WHO NEEDS THIS? PLEASE REPORT TO MUELLER...
        VLDEPRECATED inline const SimpleMessage& receiveViaMsgBuffer(
                SmartRef<AbstractCommChannel> channel,
                const UINT timeout = AbstractCommChannel::TIMEOUT_INFINITE) {
            VLSTACKTRACE("AbstractSyncMsgUser::receiveViaMsgBuffer", __FILE__,
                __LINE__);
            return this->receiveViaMsgBuffer(
                channel.DynamicCast<AbstractCommClientChannel>(), timeout);
        }

        /**
         * Receive a SimpleMessage into the local message buffer from the given 
         * 'channel'.
         *
         * @param channel The channel to receive the message from
         * @param timeout A timeout for receiving the message. If the message
         *                cannot be received in time, the operation will fail
         *                with an exception.
         *
         * @return A reference to the message received. This will be valid 
         *         until the next message is sent or received.
         *
         * @throws Exception Or derived in case of a communication error.
         */
        const SimpleMessage& receiveViaMsgBuffer(
            SmartRef<AbstractCommClientChannel> channel,
            const UINT timeout = AbstractCommChannel::TIMEOUT_INFINITE);

        /**
         * Receive a SimpleMessage into the local message buffer from the given 
         * 'channel'.
         *
         * @param channel The channel to receive the message from
         * @param timeout A timeout for receiving the message. If the message
         *                cannot be received in time, the operation will fail
         *                with an exception.
         *
         * @return A reference to the message received. This will be valid 
         *         until the next message is sent or received.
         *
         * @throws Exception Or derived in case of a communication error.
         */
        inline const SimpleMessage& receiveViaMsgBuffer(
                SmartRef<TcpCommChannel> channel,
                const UINT timeout = AbstractCommChannel::TIMEOUT_INFINITE) {
            VLSTACKTRACE("AbstractSyncMsgUser::receiveViaMsgBuffer", __FILE__,
                __LINE__);
            return this->receiveViaMsgBuffer(
                channel.DynamicCast<AbstractCommClientChannel>(), timeout);
        }

        /**
         * Pack a message with the given ID and body data into the local message
         * buffer and send it via the given 'channel'.
         *
         * @param channel  The communication channel to use for sending.
         * @param msgID    The ID of the message that will be added in the 
         *                 message header.
         * @param body     Pointer to 'bodySize' bytes of message body data.
         * @param bodySize Size of the body in bytes.
         * @param timeout  A timeout for sending the message. If the message
         *                 cannot be send in time, the operation will fail
         *                 with an exception.
         *
         * @throws Exception Or derived in case of a communication error.
         */
        // TODO: WHO NEEDS THIS? PLEASE REPORT TO MUELLER...
        VLDEPRECATED inline void sendViaMsgBuffer(SmartRef<AbstractCommChannel> channel,
                const SimpleMessageID msgID, 
                const void *body, 
                const unsigned int bodySize,
                const UINT timeout = AbstractCommChannel::TIMEOUT_INFINITE) {
            VLSTACKTRACE("AbstractSyncMsgUser::sendViaMsgBuffer", __FILE__, 
                __LINE__);
            this->sendViaMsgBuffer(
                channel.DynamicCast<AbstractCommClientChannel>(),
                msgID, body, bodySize, timeout);
        }

        /**
         * Pack a message with the given ID and body data into the local message
         * buffer and send it via the given 'channel'.
         *
         * @param channel  The communication channel to use for sending.
         * @param msgID    The ID of the message that will be added in the 
         *                 message header.
         * @param body     Pointer to 'bodySize' bytes of message body data.
         * @param bodySize Size of the body in bytes.
         * @param timeout  A timeout for sending the message. If the message
         *                 cannot be send in time, the operation will fail
         *                 with an exception.
         *
         * @throws Exception Or derived in case of a communication error.
         */
        void sendViaMsgBuffer(SmartRef<AbstractCommClientChannel> channel,
            const SimpleMessageID msgID, 
            const void *body, 
            const unsigned int bodySize,
            const UINT timeout = AbstractCommChannel::TIMEOUT_INFINITE);

        /**
         * Pack a message with the given ID and body data into the local message
         * buffer and send it via the given 'channel'.
         *
         * @param channel  The communication channel to use for sending.
         * @param msgID    The ID of the message that will be added in the 
         *                 message header.
         * @param body     Pointer to 'bodySize' bytes of message body data.
         * @param bodySize Size of the body in bytes.
         * @param timeout  A timeout for sending the message. If the message
         *                 cannot be send in time, the operation will fail
         *                 with an exception.
         *
         * @throws Exception Or derived in case of a communication error.
         */
        inline void sendViaMsgBuffer(SmartRef<TcpCommChannel> channel,
                const SimpleMessageID msgID, 
                const void *body, 
                const unsigned int bodySize,
                const UINT timeout = AbstractCommChannel::TIMEOUT_INFINITE) {
            VLSTACKTRACE("AbstractSyncMsgUser::sendViaMsgBuffer", __FILE__, 
                __LINE__);
            this->sendViaMsgBuffer(
                channel.DynamicCast<AbstractCommClientChannel>(),
                msgID, 
                body, 
                bodySize,
                timeout);
        }

        /**
         * Pack a structure of type 'T' into a message with the given ID and 
         * send it via the given 'channel'.
         *
         * The user is responsible for choosing only valid body data for 'T', 
         * i. e. structures which do not contain pointers etc. and therefore
         * can be sent directly.
         *
         * @param channel  The communication channel to use for sending.
         * @param msgID    The ID of the message that will be added in the 
         *                 message header.
         * @param body     The message body.
         * @param timeout  A timeout for sending the message. If the message
         *                 cannot be send in time, the operation will fail
         *                 with an exception.
         *
         * @throws Exception Or derived in case of a communication error.
         */
        template<class T> 
        inline void sendViaMsgBuffer(
                SmartRef<AbstractCommClientChannel> channel,
                const SimpleMessageID msgID, 
                const T& body) {
            VLSTACKTRACE("AbstractSyncMsgUser::sendViaMsgBuffer", __FILE__, 
                __LINE__);
            this->sendViaMsgBuffer(channel, msgID, &body, sizeof(T));
        }

        /**
         * Pack a structure of type 'T' into a message with the given ID and 
         * send it via the given 'channel'.
         *
         * The user is responsible for choosing only valid body data for 'T', 
         * i. e. structures which do not contain pointers etc. and therefore
         * can be sent directly.
         *
         * @param channel  The communication channel to use for sending.
         * @param msgID    The ID of the message that will be added in the 
         *                 message header.
         * @param body     The message body.
         * @param timeout  A timeout for sending the message. If the message
         *                 cannot be send in time, the operation will fail
         *                 with an exception.
         *
         * @throws Exception Or derived in case of a communication error.
         */
        template<class T> 
        inline void sendViaMsgBuffer(SmartRef<TcpCommChannel> channel,
                const SimpleMessageID msgID, 
                const T& body) {
            VLSTACKTRACE("AbstractSyncMsgUser::sendViaMsgBuffer", __FILE__, 
                __LINE__);
            this->sendViaMsgBuffer(channel, msgID, &body, sizeof(T));
        }

        /**
         * Assignment operator that does nothing.
         *
         * The message buffer will not be copied, i. e. each copy will have its
         * own buffer.
         *
         * @param rhs The right hand side operand.
         *
         * @return (*this).
         */
        inline AbstractSyncMsgUser& operator =(const AbstractSyncMsgUser& rhs) {
            VLSTACKTRACE("AbstractSyncMsgUser::operator =", __FILE__, __LINE__);
            return (*this);
        }

    private:

        /**
         * The resizable message buffer. 
         * The implementation assumes that there can be at most one message
         * "on the wire", i. e. sending and receiving messages is a strictly
         * sequential process for the node.
         */
        SimpleMessage msgBuffer;

    };


    /*
     * AbstractSyncMsgUser::requestViaMsgBuffer
     */
    template<class I, SimpleMessageID Iid, class O, SimpleMessageID Oid> 
    const O *AbstractSyncMsgUser::requestViaMsgBuffer(
            SmartRef<AbstractCommClientChannel> channel,
            const I& requestBody,
            SIZE_T *outAdditionalBytes,
            const UINT timeout) {
        VLSTACKTRACE("AbstractSyncMsgUser::requestViaMsgBuffer", __FILE__, 
            __LINE__);

        /* Send request and receive response. */
        const SimpleMessage& response = this->requestViaMsgBuffer(channel, 
            Iid, requestBody);

        /* Check response ID. */
        SimpleMessageID responseID = response.GetHeader().GetMessageID();
        if (responseID != Oid) {
            throw UnexpectedMessageException(responseID, Oid, __FILE__, 
                __LINE__);
        }

        /* Get overflow data length if requested. */
        if (outAdditionalBytes != NULL) {
            *outAdditionalBytes = response.GetHeader().GetBodySize() 
                - sizeof(O);
        }

        /* Get body. */
        const O *retval = static_cast<const O*>(response.GetBody());
        return retval;
    }
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTSYNCMSGUSER_H_INCLUDED */

