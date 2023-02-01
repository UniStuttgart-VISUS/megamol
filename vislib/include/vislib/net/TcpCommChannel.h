/*
 * TcpCommChannel.h
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/net/AbstractCommChannel.h"
#include "vislib/net/Socket.h"


namespace vislib::net {


/**
 * This class implements a communication channel based on TCP/IP sockets.
 * The TCP/IP version supports client as well as server end point behaviour.
 */
class TcpCommChannel : public AbstractCommChannel {

public:
    /**
     * Create a new channel.
     *
     * @param flags The flags for the channel.
     */
    static inline SmartRef<TcpCommChannel> Create(const UINT64 flags = 0) {
        return SmartRef<TcpCommChannel>(new TcpCommChannel(flags), false);
    }

    /**
     * This behaviour flag disables the Nagle algorithm for send
     * coalescing. Setting the flag has an effect on the communication
     * channel itself as well as on the child channels created in server
     * mode.
     */
    static const UINT64 FLAG_NODELAY;

    /**
     * This behaviour flag sets the send buffer of the underlying socket to
     * zero. Setting the flag has an effect on the communication
     * channel itself as well as on the child channels created in server
     * mode.
     */
    static const UINT64 FLAG_NOSENDBUFFER;

    /**
     * This flag enables or disables the reuse of addresses already bound.
     * Setting the flag has an effect on the communication channel itself
     * as well as on the child channels created in server mode.
     */
    static const UINT64 FLAG_REUSE_ADDRESS;

    /**
     * Permit incoming connection attempt on the communication channel.
     *
     * @return The client connection.
     *
     * @throws SocketException In case the operation fails.
     */
    SmartRef<AbstractCommClientChannel> Accept() override;

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
    void Bind(SmartRef<AbstractCommEndPoint> endPoint) override;

    /**
     * Terminate the open connection if any and reset the communication
     * channel to initialisation state.
     *
     * @throws SocketException In case the operation fails.
     */
    void Close() override;

    /**
     * Connects the channel to the peer node at the specified end
     * point address.
     *
     * @param endPoint The remote end point to connect to.
     *
     * @throws Exception Or derived in case the operation fails.
     */
    void Connect(SmartRef<AbstractCommEndPoint> endPoint) override;

    /**
     * Get the underlying socket.
     *
     * @return The underlying socket.
     */
    inline Socket& GetSocket() {
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
    SmartRef<AbstractCommEndPoint> GetLocalEndPoint() const override;

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
    SmartRef<AbstractCommEndPoint> GetRemoteEndPoint() const override;

    /**
     * Answer whether the Nagle algorihm is disabled on the socket.
     *
     * @return true if the Nagle algorithm is disabled, false otherwise.
     */
    inline bool IsSetNoDelay() const {
        return ((this->flags & FLAG_NODELAY) != 0);
    }

    /**
     * Answer whether the send buffer of this socket is set to zero.
     *
     * @return true if the send buffer is zero, false otherwise.
     */
    inline bool IsSetNoSendBuffer() const {
        return ((this->flags & FLAG_NOSENDBUFFER) != 0);
    }


    /**
     * Answer whether address reuse (SO_REUSEADDR) is enabled or not.
     *
     * @return true if address reuse is enabled, false otherwise.
     */
    inline bool IsSetReuseAddress() const {
        return ((this->flags & FLAG_REUSE_ADDRESS) != 0);
    }

    /**
     * Place the communication channel in a state in which it is listening
     * for an incoming connection.
     *
     * @param backlog Maximum length of the queue of pending connections.
     *
     * @throws SocketException In case the operation fails.
     */
    void Listen(const int backlog = SOMAXCONN) override;

    /**
     * Receives 'cntBytes' over the communication channel and saves them to
     * the memory designated by 'outData'. 'outData' must be large enough to
     * receive at least 'cntBytes'.
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
    SIZE_T Receive(void* outData, const SIZE_T cntBytes, const UINT timeout = TIMEOUT_INFINITE,
        const bool forceReceive = true) override;

    /**
     * Send 'cntBytes' from the location designated by 'data' over the
     * communication channel.
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
    SIZE_T Send(const void* data, const SIZE_T cntBytes, const UINT timeout = TIMEOUT_INFINITE,
        const bool forceSend = true) override;

private:
    /** Superclass typedef. */
    typedef AbstractCommChannel Super;

    /**
     * Ctor.
     *
     * @param flags The flags for the channel.
     */
    explicit TcpCommChannel(const UINT64 flags);

    /**
     * Create a communication channel from an existing socket.
     *
     * @param socket The socket to be used.
     */
    TcpCommChannel(Socket& socket, const UINT64 flags);

    /**
     * Disallow copies as we want to handle that via reference counting.
     *
     * @param rhs The object to be cloned.
     *
     * @throws UnsupportedOperationException Unconditionally.
     */
    TcpCommChannel(const TcpCommChannel& rhs);

    /** Dtor. */
    ~TcpCommChannel() override;

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

} // namespace vislib::net

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
