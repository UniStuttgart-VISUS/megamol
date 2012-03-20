/*
 * UdpCommChannel.h
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_UDPCOMMCHANNEL_H_INCLUDED
#define VISLIB_UDPCOMMCHANNEL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Socket.h"                      // Must be first!
#include "vislib/AbstractCommChannel.h"
#include "vislib/StackTrace.h"


namespace vislib {
namespace net {


    /**
     * This class implements a datagram communication channel based on a UDP
     * socket. 
     */
    class UdpCommChannel : public AbstractCommChannel {

    public:

        /**
         * Create a new channel.
         *
         * @param flags The flags for the channel.
         */
        static inline SmartRef<UdpCommChannel> Create(const UINT64 flags = 0) {
            VLSTACKTRACE("UdpCommChannel::Create", __FILE__, __LINE__);
            return SmartRef<UdpCommChannel>(new UdpCommChannel(flags), false);
        }

        /**
         * This flag enables transmission and receipt of broadcast messages 
         * using the channel
         */
        static const UINT64 FLAG_BROADCAST;

        /**
         * This flag enables or disables the reuse of addresses already bound.
         * Setting the flag has an effect on the communication channel itself 
         * as well as on the child channels created in server mode.
         */
        static const UINT64 FLAG_REUSE_ADDRESS;

        /**
         * Permit incoming connection attempt on the communication channel.
         *
         * IMPORTANT NOTE: This operation is not supported on datagram sockets
         * and will therefore always fail!
         *
         * @return The client connection.
         *
         * @throws SocketException In case the operation fails.
         */
        virtual SmartRef<AbstractCommClientChannel> Accept(void);

        /**
         * Binds the server to a specified end point address.
         *
         * @param endPoint The end point address to bind to.
         *
         * @throws IllegalParamException If the specified end point is no 
         *                               IP end point.
         * @throws SocketException If the socket could not be bound to the
         *                         specified end point address.
         */
        virtual void Bind(SmartRef<AbstractCommEndPoint> endPoint);

        /**
         * Close the underlying socket and reset the communication channel to
         * its initialisation state.
         *
         * @throws SocketException In case the operation fails.
         */
        virtual void Close(void);

        /**
         * Sets the default target address which to all datagram packets are 
         * sent if no explicit target address is specified.
         *
         * @param endPoint The remote end point to connect to.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual void Connect(SmartRef<AbstractCommEndPoint> endPoint);

        /**
         * Get the underlying socket.
         *
         * @return The underlying socket.
         */
        inline Socket& GetSocket(void) {
            VLSTACKTRACE("UdpCommChannel::GetSocket", __FILE__, __LINE__);
            return this->socket;
        }

        /**
         * Answer the address the channel is using locally.
         *
         * The object returned needs not necessarily to be identical with the
         * address and end point has been bound or connected. Subclasses must,
         * however, guarantee that the returned end point is equal wrt. to the
         * == operator of the end point object.
         *
         * @return The address of the local end point.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual SmartRef<AbstractCommEndPoint> GetLocalEndPoint(void) const;

        /**
         * Answer the address the remote peer of this channel is using.
         *
         * The object returned needs not necessarily to be identical with the
         * address and end point has been bound or connected. Subclasses must,
         * however, guarantee that the returned end point is equal wrt. to the
         * == operator of the end point object.
         *
         * @return The address of the remote end point.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual SmartRef<AbstractCommEndPoint> GetRemoteEndPoint(void) const;

        /**
         * Answer whether the channel is enabled for broadcast use.
         *
         * @return true if broadcast is enabled, false otherwise.
         */
        inline bool IsSetBroadcast(void) const {
            VLSTACKTRACE("UdpCommChannel::IsSetBroadcast", __FILE__, __LINE__);
            return ((this->flags & FLAG_BROADCAST) != 0);
        }

        /**
         * Answer whether address reuse (SO_REUSEADDR) is enabled or not.
         *
         * @return true if address reuse is enabled, false otherwise.
         */
        inline bool IsSetReuseAddress(void) const {
            VLSTACKTRACE("UdpCommChannel::IsSetReuseAddress", __FILE__, 
                __LINE__);
            return ((this->flags & FLAG_REUSE_ADDRESS) != 0);
        }

        /**
         * Place the communication channel in a state in which it is listening 
         * for an incoming connection.
         *
         * IMPORTANT NOTE: This operation is not supported on datagram sockets
         * and will therefore always fail!
         *
         * @param backlog Maximum length of the queue of pending connections.
         *
         * @throws SocketException In case the operation fails.
         */
        virtual void Listen(const int backlog = SOMAXCONN);

        /**
         * Receives 'cntBytes' over the communication channel and saves them to 
         * the memory designated by 'outData'. 'outData' must be large enough to 
         * receive at least 'cntBytes'.
         *
         * IMPORTANT NOTE: You must set the default peer address using Connect()
         * before you can use this method.
         *
         * @param outData      The buffer to receive the data. The caller must
         *                     allocate this memory and remains owner.
         * @param cntBytes     The number of bytes to receive.
         * @param timeout      A timeout in milliseconds. 
         *                     Use AbstractCommChannel::TIMEOUT_INFINITE to 
         *                     specify an infinite timeout. If the operation 
         *                     timeouts, an exception will be thrown.
         * @param forceReceive If the data block cannot be received as a single 
         *                     packet, repeat the operation until all of 
         *                     'cntBytes' is received or a step fails.
         *
         * @return The number of bytes acutally received.
         *
         * @throws PeerDisconnectedException In case the peer node did 
         *                                   disconnect gracefully.
         * @throws SocketException In case the operation fails.
         */
        virtual SIZE_T Receive(void *outData, const SIZE_T cntBytes,
            const UINT timeout = TIMEOUT_INFINITE, 
            const bool forceReceive = true);

        /**
         * Send 'cntBytes' from the location designated by 'data' over the
         * communication channel.
         *
         * IMPORTANT NOTE: You must set the default peer address using Connect()
         * before you can use this method.
         *
         * @param data      The data to be sent. The caller remains owner of the
         *                  memory.
         * @param cntBytes  The number of bytes to be sent. 'data' must contain
         *                  at least this number of bytes.
         * @param timeout   A timeout in milliseconds. 
         *                  Use AbstractCommChannel::TIMEOUT_INFINITE to 
         *                  specify an infinite timeout. If the operation 
         *                  timeouts, an exception will be thrown.
         * @param forceSend If the data block cannot be sent as a single packet,
         *                  repeat the operation until all of 'cntBytes' is sent
         *                  or a step fails.
         *
         * @return The number of bytes acutally sent.
         *
         * @throws SocketException In case the operation fails.
         */
        virtual SIZE_T Send(const void *data, const SIZE_T cntBytes,
            const UINT timeout = TIMEOUT_INFINITE, 
            const bool forceSend = true);

    private:

        /** Superclass typedef. */
        typedef AbstractCommChannel Super;

        /**
         * Ctor.
         *
         * @param flags The flags for the channel.
         */
        explicit UdpCommChannel(const UINT64 flags);

        /**
         * Create a communication channel from an existing socket.
         *
         * @param socket The socket to be used.
         */
        UdpCommChannel(Socket& socket, const UINT64 flags);

        /** Dtor. */
        virtual ~UdpCommChannel(void);

        /**
         * Creates or re-creates the underlying socket.
         *
         * If the socket already exists, it is destroyed and re-created.
         *
         * The current flags are applied to the newly created socket.
         *
         * @param endPoint This end point is used to determine the address
         *                 family of the new socket.
         *
         * @throws SocketException In case the operation fails.
         */
        void createSocket(const IPEndPoint& endPoint);

        /** Behaviour flags for the channel. */
        UINT64 flags;

        /** The socket that performs the actual work. */
        Socket socket;

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_UDPCOMMCHANNEL_H_INCLUDED */
