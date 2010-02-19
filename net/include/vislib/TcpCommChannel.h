/*
 * TcpCommChannel.h
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_TCPCOMMCHANNEL_H_INCLUDED
#define VISLIB_TCPCOMMCHANNEL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/Socket.h"          // Must be first.
#include "vislib/AbstractBidiCommChannel.h"
#include "vislib/AbstractClientEndpoint.h"
#include "vislib/AbstractServerEndpoint.h"


namespace vislib {
namespace net {


    /**
     * This class implements a communication channel based on TCP/IP sockets.
     * The TCP/IP version supports AbstractClientEndpoint and
     * AbstractServerEndpoint as well as AbstractBidiCommChannel.
     *
     * Q: Why does TcpCommChannel not support vislib::net::SocketAddress?
     * A: TcpCommChannel is designed to support IPv6 from the beginning. 
     *    SocketAddress is only for backward compatibility and should not be used
     *    for new programs.
     */
    class TcpCommChannel : public AbstractBidiCommChannel,
            public AbstractClientEndpoint, 
            public AbstractServerEndpoint {

    public:

         /**
          * This behaviour flag disables the Nagle algorithm for send 
          * coalescing. Setting the flag has an effect on the communication
          * channel itself as well as on the child channels created in server
          * mode.
          */
        static const UINT64 FLAG_NODELAY;

        /**
         * Ctor.
         *
         * @param flags The flags for the channel.
         */
        TcpCommChannel(const UINT64 flags = 0);

        /**
         * Create a communication channel from an existing socket.
         *
         * @param socket The socket to be used.
         */
        explicit TcpCommChannel(Socket& socket);

        /**
         * Permit incoming connection attempt on the communication channel.
         *
         * @return The client connection.
         *
         * @throws SocketException In case the operation fails.
         */
        virtual SmartRef<AbstractCommChannel> Accept(void);

        /**
         * Increment the reference count.
         *
         * @return The new value of the reference counter.
         */
        //UINT32 AddRef(void);

        /**
         * Binds the server to a specified address.
         *
         * @param address The address to bind to.
         *
         * @throws SocketException In case the operation fails.
         */
        virtual void Bind(const char *address);

        /**
         * Binds the server to a specified address.
         *
         * @param address The address to bind to.
         *
         * @throws SocketException In case the operation fails.
         */
        virtual void Bind(const wchar_t *address);

        /**
         * Binds the server to a specified address.
         *
         * @param address The address to bind to.
         *
         * @throws SocketException In case the operation fails.
         */
        virtual void Bind(const IPEndPoint address);

        /**
         * Terminate the open connection if any and reset the communication
         * channel to initialisation state.
         *
         * @throws SocketException In case the operation fails.
         */
        virtual void Close(void);

        /**
         * Connects the end point to the peer node at the specified address.
         *
         * The method tries to use IPv6 if possible.
         *
         * @param address The address to connect to.
         *
         * @throws SocketException In case the operation fails.
         * @throws IllegalParamException In case the address could not be 
         *                               parsed.
         */
        virtual void Connect(const char *address);

        /**
         * Connects the end point to the peer node at the specified address.
         *
         * The method tries to use IPv6 if possible.
         *
         * @param address The address to connect to.
         *
         * @throws SocketException In case the operation fails.
         * @throws IllegalParamException In case the address could not be 
         *                               parsed.
         */
        virtual void Connect(const wchar_t *address);

        /**
         * Connects the end point to the peer node at the specified address.
         *
         * @param address The end point to connect to.
         *
         * @throws SocketException In case the operation fails.
         */
        virtual void Connect(const IPEndPoint& address);

        /**
         * Get the underlying socket.
         *
         * @return The underlying socket.
         */
        inline Socket& GetSocket(void) {
            return this->socket;
        }

        /**
         * Answer whether the Nagle algorihm is disabled on the socket.
         *
         * @return true if the Nagle algorithm is disabled, false otherwise.
         */
        inline bool IsSetNoDelay(void) const {
            return ((this->flags & FLAG_NODELAY) != 0);
        }

        /**
         * Place the communication channel in a state in which it is listening 
         * for an incoming connection.
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
         * @param outData      The buffer to receive the data. The caller must
         *                     allocate this memory and remains owner.
         * @param cntBytes     The number of bytes to receive.
         * @param timeout      A timeout in milliseconds. A value less than 1 
         *                     specifies an infinite timeout. If the operation 
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
            const INT timeout = TIMEOUT_INFINITE, 
            const bool forceReceive = true);

        /**
         * Decrement the reference count. If the reference count reaches zero,
         * the object is released using the allocator A.
         *
         * @return The new value of the reference counter.
         */
        //UINT32 Release(void);

        /**
         * Send 'cntBytes' from the location designated by 'data' over the
         * communication channel.
         *
         * @param data      The data to be sent. The caller remains owner of the
         *                  memory.
         * @param cntBytes  The number of bytes to be sent. 'data' must contain
         *                  at least this number of bytes.
         * @param timeout   A timeout in milliseconds. A value less than 1 
         *                  specifies an infinite timeout. If the operation 
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
            const INT timeout = TIMEOUT_INFINITE, 
            const bool forceSend = true);

    protected:

        /** Dtor. */
        virtual ~TcpCommChannel(void);

    private:

        /**
         * Disallow copies as we want to handle that via reference counting.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        TcpCommChannel(const TcpCommChannel& rhs);

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
#endif /* VISLIB_TCPCOMMCHANNEL_H_INCLUDED */
