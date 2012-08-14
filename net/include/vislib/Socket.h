/*
 * Socket.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). All rights reserved.
 */

#ifndef VISLIB_SOCKET_H_INCLUDED
#define VISLIB_SOCKET_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <Winsock2.h>
#else /* _WIN32 */
#include <arpa/inet.h>
#include <netinet/tcp.h>

#define SOCKET int
#define INVALID_SOCKET (-1)
#endif /* !_WIN32 */

#ifdef _MSC_VER
#pragma comment(lib, "Ws2_32")
#endif /* _MSC_VER */


#include "vislib/IPEndPoint.h"
#include "vislib/SocketAddress.h"
#include "vislib/types.h"


namespace vislib {
namespace net {

    /**
     * This is a raw socket wrapper class that allows platform-independent use
     * of socket functionality on Windows and Linux systems.
     *
     * @author Christoph Mueller
     */
    class Socket {

    public:

        /** The supported protocols. */
        enum Protocol {
            PROTOCOL_IP = IPPROTO_IP,               //!< IPv4.
            PROTOCOL_HOPOPTS = IPPROTO_HOPOPTS,     //!< IPv6 hop-by-hop options.
            PROTOCOL_ICMP = IPPROTO_ICMP,           //!< Control message protocol.
            PROTOCOL_IGMP = IPPROTO_IGMP,           //!< Internet group mgmnt.
            //PROTOCOL_GGP = IPPROTO_GGP,             // Gateway^2 (deprecated).
#ifdef _WIN32
            PROTOCOL_IPV4 = IPPROTO_IPV4,           // IPv4.
#endif /* _WIN32 */
            PROTOCOL_TCP = IPPROTO_TCP,             //!< TCP.
            PROTOCOL_PUP = IPPROTO_PUP,             //!< PUP.
            PROTOCOL_UDP = IPPROTO_UDP,             //!< User Datagram Protocol.
            PROTOCOL_IDP = IPPROTO_IDP,             //!< XNS IDP.
            PROTOCOL_IPV6 = IPPROTO_IPV6,           //!< IPv6.
            PROTOCOL_ROUTING = IPPROTO_ROUTING,     //!< IPv6 routing header.
            PROTOCOL_FRAGMENT = IPPROTO_FRAGMENT,   //!< IPv6 fragmentation hdr.
            PROTOCOL_ESP = IPPROTO_ESP,             //!< IPsec ESP header.
            PROTOCOL_AH = IPPROTO_AH,               //!< IPsec AH.
            PROTOCOL_ICMPV6 = IPPROTO_ICMPV6,       //!< ICMPv6.
            PROTOCOL_NONE = IPPROTO_NONE,           //!< IPv6 no next header.
            PROTOCOL_DSTOPTS = IPPROTO_DSTOPTS,     //!< IPv6 destination options.
            //PROTOCOL_ND = IPPROTO_ND,             //!< UNOFFICIAL net disk prot.
            //PROTOCOL_ICLFXBM = IPPROTO_ICLFXBM,
            PROTOCOL_RAW = IPPROTO_RAW              //!< Raw IP packet.
        };


        /** The supported protocol families. */
        enum ProtocolFamily {
            FAMILY_UNSPEC = PF_UNSPEC,      //!< Unspecified.
            //FAMILY_UNIX = PF_UNIX,
            FAMILY_INET = PF_INET,          //!< UDP, TCP etc. version 4.
            //FAMILY_IMPLINK = PF_IMPLINK,
            //FAMILY_PUP = PF_PUP,
            //FAMILY_CHAOS = PF_CHAOS,
            //FAMILY_NS = PF_NS,
            //FAMILY_IPX = PF_IPX,
            //FAMILY_ISO = PF_ISO,
            //FAMILY_OSI = PF_OSI,
            //FAMILY_ECMA = PF_ECMA,
            //FAMILY_DATAKIT = PF_DATAKIT,
            //FAMILY_CCITT = PF_CCITT,
            //FAMILY_SNA = PF_SNA,
            //FAMILY_DECnet = PF_DECnet,
            //FAMILY_DLI = PF_DLI,
            //FAMILY_LAT = PF_LAT,
            //FAMILY_HYLINK = PF_HYLINK,
            //FAMILY_APPLETALK = PF_APPLETALK,
            //FAMILY_VOICEVIEW = PF_VOICEVIEW,
            //FAMILY_FIREFOX = PF_FIREFOX,
            //FAMILY_UNKNOWN1 = PF_UNKNOWN1,
            //FAMILY_BAN = PF_BAN,
            //FAMILY_ATM = PF_ATM,
            FAMILY_INET6 = PF_INET6     //!< Internet version 6.
        };


        /** 
         * Flag that describes what types of operation will no longer be 
         * allowed when calling Shutdown() on a socket. 
         */
        enum ShutdownManifest {
#ifdef _WIN32
            BOTH = SD_BOTH,                 // Shutdown send and receive.
            RECEIVE = SD_RECEIVE,           // Shutdown receive only.
            SEND = SD_SEND                  // Shutdown sent only
#else /* _WIN32 */
            BOTH = SHUT_RDWR,
            RECEIVE = SHUT_RD,
            SEND = SHUT_WR
#endif /* _WIN32 */
        };


        /** The supported socket types. */
        enum Type {
            TYPE_STREAM = SOCK_STREAM,      // Stream socket.
            TYPE_DGRAM = SOCK_DGRAM,        // Datagram socket.
            TYPE_RAW = SOCK_RAW,            // Raw socket.
            TYPE_RDM = SOCK_RDM,            // Reliably-delivered message.
            TYPE_SEQPACKET = SOCK_SEQPACKET // Sequenced packet stream.
        };


        /**
         * Cleanup after use of sockets. This method does nothing on Linux. On
         * Windows, it calls WSACleanup. Call this method after you finished 
         * using sockets for releasing the resources associated with the 
         * Winsock-DLL.
         *
         * It is safe to call Cleanup multiple times as long as Startup has been
         * called multiple times before.
         *
         * @throws SocketException In case of an error.
         */
        static void Cleanup(void);

        /**
         * Initialise the use of sockets. This method does nothing on Linux. On
         * Windows, this method calls WSAStartup and initialises the 
         * Winsock-DLL. You must call this method once before using any sockets.
         * Call Cleanup() after you finished the use of sockets.
         *
         * It is safe to call Startup multiple times as long as the same number
         * of Cleanup calls is made afterwards.
         *
         * @throws SocketException In case of an error.
         */
        static void Startup(void);

        /** Constant for specifying an infinite timeout. */
        static const UINT TIMEOUT_INFINITE;

        /**
         * Create an invalid socket. Call Create() on the new object to create 
         * a new socket.
         */
        inline Socket(void) : handle(INVALID_SOCKET) {}

        /**
         * Create socket wrapper from an existing handle.
         *
         * @param handle The socket handle.
         */
        explicit inline Socket(SOCKET handle) : handle(handle) {}

        /**
         * Create a copy of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Socket(const Socket& rhs) : handle(rhs.handle) {}

        /** Dtor. */
        virtual ~Socket(void);

        /**
         * Permit incoming connection attempt on the socket.
         *
         * Note for Linux: The implementation handles EINTR by itself and 
         * retries the operation until it succeeds or fails with another
         * error.
         *
         * @param outConnAddr Optional pointer to an IPEndPoint that receives 
         *                    the address of the connecting entity, as known to 
         *                    the communications layer. The exact format of the 
         *                    address is determined by the address family 
         *                    established when the socket was created. This 
         *                    parameter can be NULL.
         *
         * @return The accepted socket.
         *
         * @throws SocketException If the operation fails.
         */
        virtual Socket Accept(IPEndPoint *outConnAddr = NULL);

        /**
         * Permit incoming connection attempt on the socket.
         *
         * This operation is only allowed on IPv4 sockets. It is recommended 
         * to use an IPEndPoint instead of a SocketAddress in order to support
         * both, IPv4 and IPv6 addresses.
         *
         * Note for Linux: The implementation handles EINTR by itself and 
         * retries the operation until it succeeds or fails with another
         * error.
         *
         * @param outConnAddr Optional pointer to a SocketAddress that receives 
         *                    the address of the connecting entity, as known to 
         *                    the communications layer. This parameter can be 
         *                    NULL.
         *
         * @return The accepted socket.
         *
         * @throws SocketException If the operation fails.
         */
        virtual Socket Accept(SocketAddress *outConnAddr);

        /**
         * Bind the socket to the specified address.
         *
         * @param address The address to bind.
         *
         * @throws SocketException If the operation fails.
         */
        virtual void Bind(const IPEndPoint& address);

        /**
         * Bind the socket to the specified address.
         *
         * This method is only allowed on IPv4 sockets. Use the version with the
         * IP-agnostic IPEndPoint on IPv6 sockets.
         *
         * @param address The address to bind.
         *
         * @throws SocketException If the operation fails.
         */
        virtual void Bind(const SocketAddress& address);

        /**
         * This is a Linux-only method. It is not required nor supported on 
         * Windows. On Windows, the method has no effect.
         *
         * Bind this socket to a particular device like "eth0", as specified 
         * in 'name'.
         *
         * @param name The name of the device to bind to, e. g. "eth0".
         *
         * @throws SocketException If the operation fails.
         */
        virtual void BindToDevice(const StringA& name);

        /**
         * Close the socket. If the socket is not open, i. e. not valid, this 
         * method succeeds, too.
         */
        virtual void Close(void);

        /**
         * Connect to the specified socket address using this socket.
         *
         * Note for Linux: The implementation handles EINTR by itself and 
         * retries the operation until it succeeds or fails with another
         * error.
         *
         * @param address The address to connect to.
         *
         * @throws SocketException If the operation fails.
         */
        virtual void Connect(const IPEndPoint& address);

        /**
         * Connect to the specified socket address using this socket.
         *
         * This method is only allowed on IPv4 sockets. Use the version with the
         * IP-agnostic IPEndPoint on IPv6 sockets.
         *
         * Note for Linux: The implementation handles EINTR by itself and 
         * retries the operation until it succeeds or fails with another
         * error.
         *
         * @param address The address to connect to.
         *
         * @throws SocketException If the operation fails.
         */
        virtual void Connect(const SocketAddress& address);

        /**
         * Create a new socket.
         *
         * @param protocolFamily The protocol family.
         * @param type           The type of the socket.
         * @param protocol       The protocol to use.
         *
         * @throws SocketException If the operation fails.
         */
        virtual void Create(const ProtocolFamily protocolFamily, const Type type, 
            const Protocol protocol);

        /**
         * Create a new socket.
         *
         * @param familySpecAddr An IPEndPoint which specifies the protocol 
         *                       family. The socket will be created either for 
         *                       IPv4 or IPv6 depending on the type of endpoint
         *                       specified here
         * @param type           The type of the socket.
         * @param protocol       The protocol to use.
         *
         * @throws IllegalParamException If 'familiySpecAddr' does not specify a
         *                               supported address family, i.e. IPv4 or
         *                               IPv6.
         * @throws SocketException If the operation fails.
         */
        void Create(const IPEndPoint& familySpecAddr, const Type type, 
            const Protocol protocol);

        /**
         * Retrieve status of transmission and receipt of broadcast messages on 
         * the socket.
         *
         * @return The current value of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline bool GetBroadcast(void) const {
            return this->getOption(SOL_SOCKET, SO_BROADCAST);
        }

        /**
         * Answer whether sockets delay the acknowledgment of a connection until
         * after the WSAAccept condition function is called. This method returns
         * always false on Linux.
         *
        * @return The current value of the option, false on Linux.
         *
         * @throws SocketException If the operation fails.
         */
        inline bool GetConditionalAccept(void) const {
#ifdef _WIN32
            return this->getOption(SOL_SOCKET, SO_CONDITIONAL_ACCEPT);
#else /* _WIN32 */
            return false;
#endif /* _WIN32 */
        }

        /**
         * Answer whether debugging information is recorded.
         *
         * @return The current value of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline bool GetDebug(void) const {
            return this->getOption(SOL_SOCKET, SO_DEBUG);
        }

        /**
         * Answer whether routing is disabled.
         *
         * @return The current value of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline bool GetDontRoute(void) const {
            return this->getOption(SOL_SOCKET, SO_DONTROUTE);
        }

        /**
         * Answer whether keep-alives are sent.
         *
         * @return The current value of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline bool GetKeepAlive(void) const {
            return this->getOption(SOL_SOCKET, SO_DONTROUTE);
        }

        /**
         * Answer the linger state of the socket
         *
         * @param outLinger Receives the current state.
         *
         * @throws SocketException If the operation fails.
         */
        inline void GetLinger(struct linger& outLinger) const {
            SIZE_T size = sizeof(struct linger);
            this->GetOption(SOL_SOCKET, SO_LINGER, &outLinger, size);
        }

        /**
         * Answer the address the socket is locally bound to (this equals to the
         * socket API function getsockname()).
         *
         * @return The address of the local end point.
         *
         * @throw SocketException If the operation fails.
         */
        IPEndPoint GetLocalEndPoint(void) const;

        /**
         * Gets the adapter that multicast packets shall be received from.
         *
         * @param outAddr Receives the IPAddress identifying the adapter.
         *
         * @throws SocketException If the operation fails.
         */
        void GetMulticastInterface(IPAddress& outAddr) const;

        /**
         * Answer whether the socket receives multicast packets sent by this
         * node and directed to a multicast group the node is member of.
         *
         * @param pf FAMILY_INET for retrieving the value for IPv4, 
         *           FAMILIY_INET6 for IPv6.
         *
         * @return true if the socket receives local multicast packets, 
         *         false otherwise.
         *
         * @throws IllegalParamException If 'pf' has an unsupported value.
         * @throws SocketException If the operation fails.
         */
        bool GetMulticastLoop(const ProtocolFamily pf) const;

        /**
         * Gets the lifetime of multicast packets.
         *
         * @param pf FAMILY_INET for retrieving the value for IPv4.
         * @return The number of routers multicast packets may pass.
         *
         * @throws IllegalParamException If 'pf' has an unsupported value.
         * @throws SocketException If the operation fails.
         */
        BYTE GetMulticastTimeToLive(const ProtocolFamily pf) const;

        /**
         * Answer the deactivation state of the Nagle algorithm for send 
         * coalescing.
         *
         * This operation fails on other sockets that TCP/IP stream sockets.
         *
         * @return true, if send coalescing is disabled, false otherwise.
         *
         * @throws SocketException If the operation fails.
         */
        inline bool GetNoDelay(void) const {
            return this->getOption(IPPROTO_TCP, TCP_NODELAY);
        }

        /**
         * Answer whether OOB data are received in the normal data stream.
         *
         * @return The current value of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline bool GetOOBInline(void) const {
            return this->getOption(SOL_SOCKET, SO_OOBINLINE);
        }

        /**
         * Retrieve a socket option.
         *
         * @param level            Level at which the option is defined.
         * @param optName          Socket option for which the value is to be
         *                         retrieved. 
         * @param outValue         Pointer to the buffer in which the value for 
         *                         the requested option is to be returned. 
         * @param inOutValueLength The size of 'outValue'. When the method 
         *                         returns, this variable will contain the 
         *                         number of bytes actually retrieved.
         */
        void GetOption(const INT level, const INT optName, void *outValue,
            SIZE_T& inOutValueLength) const;

        /**
         * Answer the address of the peer to which a socket is connected.
         *
         * @return The address of the peer end point.
         *
         * @throw SocketException If the operation fails.
         */
        IPEndPoint GetPeerEndPoint(void) const;

        /**
         * Answer the total per-socket buffer space reserved for receives.
         *
         * @return The current value of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline INT GetRcvBuf(void) const {
            INT retval;
            SIZE_T size = sizeof(INT);
            this->GetOption(SOL_SOCKET, SO_RCVBUF, &retval, size);
            return retval;
        }

        /**
         * Answer the receive time-out in milliseconds.
         *
         * @return The timeout in milliseconds.
         *
         * @throws SocketException If the operation fails.
         */
        inline INT GetRcvTimeo(void) const {
            INT retval;
            SIZE_T size = sizeof(INT);
            this->GetOption(SOL_SOCKET, SO_RCVTIMEO, &retval, size);
            return retval;
        }

        /**
         * Answer whether it is allowed that the socket is bound to an address
         * that is already in use.
         *
         * @return The current value of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline bool GetReuseAddr(void) const {
            return this->getOption(SOL_SOCKET, SO_REUSEADDR);
        }

        /**
         * Answer whether a socket is bound for exclusive access. Returns false 
         * on Linux unconditionally.
         *
         * @return The current value of the option, false on Linux.
         *
         * @throws SocketException If the operation fails.
         */
        inline bool GetExclusiveAddrUse(void) const {
#ifdef _WIN32
            return this->getOption(SOL_SOCKET, SO_EXCLUSIVEADDRUSE);
#else /* _WIN32 */
            return false;
#endif /* _WIN32 */
        }

        /**
         * Answer the total per-socket buffer space reserved for sends. 
         *
         * @return The buffer size for send operations in bytes.
         *
         * @throws SocketException If the operation fails.
         */
        inline INT GetSndBuf(void) const {
            INT retval;
            SIZE_T size = sizeof(INT);
            this->GetOption(SOL_SOCKET, SO_SNDBUF, &retval, size);
            return retval;
        }

        /**
         * Answer the send time-out in milliseconds.
         *
         * @return The timeout in milliseconds.
         *
         * @throws SocketException If the operation fails.
         */
        inline INT GetSndTimeo(void) const {
            INT retval;
            SIZE_T size = sizeof(INT);
            this->GetOption(SOL_SOCKET, SO_SNDTIMEO, &retval, size);
            return retval;
        }

        /**
         * Performs a graceful TCP client disconnect sequence. 
         *
         * This method must only be called on TCP sockets! It is intended
         * for sockets that have the linger option not enabled.
         * It performs a shutdown of the send channel and waits for any pending 
         * data to receive. The caller should close the socket afterwards if
         * 'isClose' is not set. Note that this method is blocking.
         *
         * See http://msdn.microsoft.com/en-us/library/ms738547(VS.85).aspx for
         * more information.
         *
         * In case the operation fails and 'isClose' is set, it is guaranteed 
         * that Close() is called in any case. However, the Close() call may
         * fail itself.
         *
         * @param isClose If true, the socket will be closed once FD_CLOSE was
         *                signaled. Otherwise, the method will return without
         *                closing the socket after it was signaled.
         *
         * @throws SocketException In case the operation fails.
         */
        void GracefulDisconnect(const bool isClose);

        /**
         * Send an I/O Control message to the socket (Windows only).
         *
         * @param ioControlCode    The I/O control code. See WSAIoctl function
         *                         in winsock2 documentation for possible codes.
         * @param inBuffer         Pointer to the input buffer.
         * @param cntInBuffer      Size of 'inBuffer' in bytes.
         * @param outBuffer        Pointer to the output buffer.
         * @param cntOutBuffer     Size of 'outBuffer' in bytes.
         * @param outBytesReturned Bytes that have actually been returned into
         *                         'outBuffer'.
         *
         * @throws SocketException If the operation fails.
         */
        // TODO: Linux IOCTLs?
        void IOControl(const DWORD ioControlCode, void *inBuffer, 
            const DWORD cntInBuffer, void *outBuffer, const DWORD cntOutBuffer,
            DWORD& outBytesReturned);

        /**
         * Answer whether the socket is valid. Only use sockets that return true
         * in this method.
         *
         * @return true, if the socket is valid, false otherwise.
         */
        inline bool IsValid(void) const {
            return (this->handle != INVALID_SOCKET);
        }

        /**
         * Leave the IPv4 multicast group identified by the mutlicast address 
         * 'group'.
         *
         * @param group   The address of the multicast group.
         * @param adapter The local address of the interface on which the 
         *                multicast group should be joined or dropped. If ANY 
         *                is used, the default multicast interface is used.
         *
         * @throws SocketException If the operation fails.
         */
        void LeaveMulticastGroup(const IPAddress& group, 
            const IPAddress& adapter = IPAddress::ANY);

        /**
         * Leave the IPv6 multicast group identified by the mutlicast address 
         * 'group'.
         *
         * @param group   The address of the multicast group.
         * @param adapter The interface index of the local interface on which
         *                the multicast group should be joined or dropped. If 
         *                0 is used, the default multicast interface is used.
         *
         * @throws SocketException If the operation fails.
         */
        void LeaveMulticastGroup(const IPAddress6& group, 
            const unsigned int adapter = 0);

        /**
         * Place the socket in a state in which it is listening for an incoming 
         * connection.
         *
         * @param backlog Maximum length of the queue of pending connections.
         *
         * @throws SocketException If the operation fails.
         */
        virtual void Listen(const INT backlog = SOMAXCONN);

        /**
         * Join the IPv4 multicast group identified by the mutlicast address 
         * 'group'.
         *
         * @param group   The address of the multicast group.
         * @param adapter The local address of the interface on which the 
         *                multicast group should be joined or dropped. If ANY 
         *                is used, the default multicast interface is used.
         *
         * @throws SocketException If the operation fails.
         */
        void JoinMulticastGroup(const IPAddress& group, 
            const IPAddress& adapter = IPAddress::ANY);

        /**
         * Join the IPv6 multicast group identified by the mutlicast address 
         * 'group'.
         *
         * @param group   The address of the multicast group.
         * @param adapter The interface index of the local interface on which
         *                the multicast group should be joined or dropped. If 
         *                0 is used, the default multicast interface is used.
         *
         * @throws SocketException If the operation fails.
         */
        void JoinMulticastGroup(const IPAddress6& group, 
            const unsigned int adapter = 0);

        /**
         * Receives 'cntBytes' from the socket and saves them to the memory 
         * designated by 'outData'. 'outData' must be large enough to receive at
         * least 'cntBytes'.
         *
         * Note that the timeout is specified for each receive call. When
         * setting the 'forceReceive' flag, multiple calls might be needed to get
         * all requested data and thus the overall timeout will be a multiple
         * of the specified timeout.
         *
         * Note for Linux: The implementation handles EINTR by itself and 
         * retries the operation until it succeeds or fails with another
         * error.
         *
         * @param outData      The buffer to receive the data. The caller must
         *                     allocate this memory and remains owner.
         * @param cntBytes     The number of bytes to receive.
         * @param timeout      A timeout in milliseconds. A value less than 1 
         *                     specifies an infinite timeout. If the operation 
         *                     timeouts, an exception will be thrown.
         * @param flags        The flags that specify the way in which the call 
         *                     is made.
         * @param forceReceive If this flag is set, the method will not return
         *                     until 'cntBytes' have been read.
         *
         * @return The number of bytes actually received.
         *
         * @throws SocketException       If the operation fails or timeouts.
         */
        virtual SIZE_T Receive(void *outData, const SIZE_T cntBytes, 
            const INT timeout = TIMEOUT_INFINITE, const INT flags = 0, 
            const bool forceReceive = false);

        /**
         * Receives a datagram from 'fromAddr' and stores it to 'outData'. 
         * 'outData' must be large enough to receive at least 'cntBytes'. 
         *
         * Note that the timeout is specified for each receive call. When
         * setting the 'forceReceive' flag, multiple calls might be needed to get
         * all requested data and thus the overall timeout will be a multiple
         * of the specified timeout.
         *
         * This method can only be used on datagram sockets.
         *
         * Note for Linux: The implementation handles EINTR by itself and 
         * retries the operation until it succeeds or fails with another
         * error.
         *
         * @param outFromAddr  The socket address the datagram was received 
         *                     from. This variable is only valid upon successful
         *                     return from the method.
         * @param outData      The buffer to receive the data. The caller must
         *                     allocate this memory and remains owner.
         * @param cntBytes     The number of bytes to receive.
         * @param timeout      A timeout in milliseconds. A value less than 1 
         *                     specifies an infinite timeout. If the operation 
         *                     timeouts, an exception will be thrown.
         * @param flags        The flags that specify the way in which the call 
         *                     is made.
         * @param forceReceive If this flag is set, the method will not return
         *                     until 'cntBytes' have been read.
         *
         * @return The number of bytes actually received.
         *
         * @throws SocketException       If the operation fails or timeouts.
         */
        virtual SIZE_T Receive(IPEndPoint& outFromAddr, void *outData, 
            const SIZE_T cntBytes, const INT timeout = TIMEOUT_INFINITE, 
            const INT flags = 0, const bool forceReceive = false);

        /**
         * Receives a datagram from 'fromAddr' and stores it to 'outData'. 
         * 'outData' must be large enough to receive at least 'cntBytes'. 
         *
         * Note that the timeout is specified for each receive call. When
         * setting the 'forceReceive' flag, multiple calls might be needed to get
         * all requested data and thus the overall timeout will be a multiple
         * of the specified timeout.
         *
         * This method can only be used on datagram sockets.
         *
         * This method is for backward compatibility and is only supported on 
         * IPv4 sockets. Use IPEndPoint instead of SocketAddress for IPv6 
         * support and better performance.
         *
         * Note for Linux: The implementation handles EINTR by itself and 
         * retries the operation until it succeeds or fails with another
         * error.
         *
         * @param outFromAddr  The socket address the datagram was received 
         *                     from. This variable is only valid upon successful
         *                     return from the method.
         * @param outData      The buffer to receive the data. The caller must
         *                     allocate this memory and remains owner.
         * @param cntBytes     The number of bytes to receive.
         * @param timeout      A timeout in milliseconds. A value less than 1 
         *                     specifies an infinite timeout. If the operation 
         *                     timeouts, an exception will be thrown.
         * @param flags        The flags that specify the way in which the call 
         *                     is made.
         * @param forceReceive If this flag is set, the method will not return
         *                     until 'cntBytes' have been read.
         *
         * @return The number of bytes actually received.
         *
         * @throws SocketException       If the operation fails or timeouts.
         */
        virtual SIZE_T Receive(SocketAddress& outFromAddr, void *outData, 
            const SIZE_T cntBytes, const INT timeout = TIMEOUT_INFINITE, 
            const INT flags = 0, const bool forceReceive = false);

        ///**
        // * Receives one object of type T to 'outData'. The method does not 
        // * return until a full object of type T has been read.
        // *
        // * Note: Be careful when communicating objects or structures to
        // * systems that have a different aligmnent!
        // *
        // * @param outData The variable that receives the data from the socket.
        // * @param flags   The flags that specify the way in which the call is 
        // *                made.
        // *
        // * @throws SocketException If the operation fails.
        // */
        //template<class T> inline void Receive(T& outData, const INT flags = 0) {
        //    return this->Receive(&outData, sizeof(T), flags, true);
        //}

        /**
         * Send 'cntBytes' from the location designated by 'data' using this 
         * socket.
         *
         * Note that the timeout is specified for each send call. When
         * setting the 'forceSend' flag, multiple calls might be needed to get
         * all requested data and thus the overall timeout will be a multiple
         * of the specified timeout.
         *
         * Note for Linux: The implementation handles EINTR by itself and 
         * retries the operation until it succeeds or fails with another
         * error.
         *
         * @param data      The data to be sent. The caller remains owner of the
         *                  memory.
         * @param cntBytes  The number of bytes to be sent. 'data' must contain
         *                  at least this number of bytes.
         * @param timeout   A timeout in milliseconds. A value less than 1 
         *                  specifies an infinite timeout. If the operation 
         *                  timeouts, an exception will be thrown.
         * @param flags     The flags that specify the way in which the call is 
         *                  made.
         *
         * @return The number of bytes acutally sent.
         *
         * @throws SocketException       If the operation fails.
         * @throws IllegalParamException If 'timeout' is not TIMEOUT_INFINITE 
         *                               and 'forceSend' is true.
         */
        virtual SIZE_T Send(const void *data, const SIZE_T cntBytes, 
            const INT timeout = TIMEOUT_INFINITE, const INT flags = 0, 
            const bool forceSend = false);

        /**
         * Send a datagram of 'cntBytes' bytes from the location designated by 
         * 'data' using this socket to the socket 'toAddr'.
         *
         * Note that the timeout is specified for each send call. When
         * setting the 'forceSend' flag, multiple calls might be needed to get
         * all requested data and thus the overall timeout will be a multiple
         * of the specified timeout.
         *
         * This method can only be used on datagram sockets.
         *
         * Note for Linux: The implementation handles EINTR by itself and 
         * retries the operation until it succeeds or fails with another
         * error.
         *
         * @param toAddr    Socket address of the destination host.
         * @param data      The data to be sent. The caller remains owner of the
         *                  memory.
         * @param cntBytes  The number of bytes to be sent. 'data' must contain
         *                  at least this number of bytes.
         * @param timeout   A timeout in milliseconds. A value less than 1 
         *                  specifies an infinite timeout. If the operation 
         *                  timeouts, an exception will be thrown.
         * @param flags     The flags that specify the way in which the call is 
         *                  made.
         *
         * @return The number of bytes acutally sent.
         *
         * @throws SocketException       If the operation fails.
         * @throws IllegalParamException If 'timeout' is not TIMEOUT_INFINITE 
         *                               and 'forceSend' is true.
         */
        virtual SIZE_T Send(const IPEndPoint& toAddr, const void *data, 
            const SIZE_T cntBytes, const INT timeout = TIMEOUT_INFINITE, 
            const INT flags = 0, const bool forceSend = false);

        /**
         * Send a datagram of 'cntBytes' bytes from the location designated by 
         * 'data' using this socket to the socket 'toAddr'.
         *
         * Note that the timeout is specified for each send call. When
         * setting the 'forceSend' flag, multiple calls might be needed to get
         * all requested data and thus the overall timeout will be a multiple
         * of the specified timeout.
         *
         * This method can only be used on datagram sockets.
         *
         * This method is for backward compatibilty and is only supported on 
         * IPv4 sockets. Use IPEndPoint instead of SocketAddress for IPv6 
         * support and better performance.
         *
         * Note for Linux: The implementation handles EINTR by itself and 
         * retries the operation until it succeeds or fails with another
         * error.
         *
         * @param toAddr    Socket address of the destination host.
         * @param data      The data to be sent. The caller remains owner of the
         *                  memory.
         * @param cntBytes  The number of bytes to be sent. 'data' must contain
         *                  at least this number of bytes.
         * @param timeout   A timeout in milliseconds. A value less than 1 
         *                  specifies an infinite timeout. If the operation 
         *                  timeouts, an exception will be thrown.
         * @param flags     The flags that specify the way in which the call is 
         *                  made.
         *
         * @return The number of bytes acutally sent.
         *
         * @throws SocketException       If the operation fails.
         * @throws IllegalParamException If 'timeout' is not TIMEOUT_INFINITE 
         *                               and 'forceSend' is true.
         */
        inline SIZE_T Send(const SocketAddress& toAddr, const void *data,
                const SIZE_T cntBytes, const INT timeout = TIMEOUT_INFINITE,
                const INT flags = 0, const bool forceSend = false) {
            return this->Send(IPEndPoint(toAddr), data, cntBytes, timeout, 
                flags, forceSend);
        }

        ///**
        // * Sends an object of type T.
        // *
        // * Note: Be careful when communicating objects or structures to
        // * systems that have a different aligmnent!
        // *
        // * @param data  The object to be sent.
        // * @param flags The flags that specify the way in which the call is 
        // *              made.
        // *
        // * @throws SocketException If the operation fails.
        // */
        //template<class T> inline void Send(const T& data, const INT flags = 0) {
        //    return this->Send(&data, sizeof(T), flags, true);
        //}

        /**
         * Enable or disable transmission and receipt of broadcast messages on 
         * the socket.
         *
         * @param enable The new activation state of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline void SetBroadcast(const bool enable) {
            this->setOption(SOL_SOCKET, SO_BROADCAST, enable);
        }

        /**
         * Enables or disables sockets to delay the acknowledgment of a 
         * connection until after the WSAAccept condition function is called.
         * This method has no effect on Linux.
         *
         * @param enable The new activation state of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline void SetConditionalAccept(const bool enable) {
#ifdef _WIN32
            this->setOption(SOL_SOCKET, SO_CONDITIONAL_ACCEPT, enable);
#endif /* _WIN32 */
        }

        /**
         * Enable or disable recording of debugging information.
         *
         * @param enable The new activation state of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline void SetDebug(const bool enable) {
            this->setOption(SOL_SOCKET, SO_DEBUG, enable);
        }

        /**
         * Enables or disables routing.
         *
         * @param enable The new activation state of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline void SetDontRoute(const bool enable) {
            this->setOption(SOL_SOCKET, SO_DONTROUTE, enable);
        }

        /**
         * Enables or disables sending keep-alives.
         *
         * @param enable The new activation state of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline void SetKeepAlive(const bool enable) {
            this->setOption(SOL_SOCKET, SO_DONTROUTE, enable);
        }

        /**
         * Enables or disables lingering on close if unsent data is present.
         *
         * @param enable     The new activation state of the option.
         * @parma lingerTime The linger time in seconds
         *
         * @throws SocketException If the operation fails.
         */
        inline void SetLinger(const bool enable, const SHORT lingerTime) {
            struct linger l = { enable, lingerTime };
            this->SetOption(SOL_SOCKET, SO_LINGER, &l, sizeof(struct linger));
        }

        /**
         * Disables the Nagle algorithm for send coalescing.
         *
         * This operation fails on other sockets that TCP/IP stream sockets.
         *
         * @param enable true for enabling immediate sending of packets (disable
         *               send coalescing), false for enabling send coalescing.
         *
         * @throws SocketException If the operation fails.
         */
        inline void SetNoDelay(const bool enable) {
            this->setOption(IPPROTO_TCP, TCP_NODELAY, enable);
        }

        /**
         * Enable or disable receive of OOB data in the normal data stream.
         *
         * @param enable The new activation state of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline void SetOOBInline(const bool enable) {
            this->setOption(SOL_SOCKET, SO_OOBINLINE, enable);
        }

        /**
         * Set a socket option.
         *
         * @param level       Level at which the option is defined.
         * @param optName     Socket option for which the value is to be set.
         * @param value       Pointer to the buffer in which the value for the 
         *                    requested option is specified. 
         * @param valueLength Size of 'value'.
         *
         * @throws SocketException If the operation fails.
         */
        virtual void SetOption(const INT level, const INT optName, 
            const void *value, const SIZE_T valueLength);

        /**
         * Enables a socket to receive all IP packets on the network through a
         * WSAIoctl call. The method has only an effect on Windows systems.
         * This IOCTL is only available on Windows 2000 or above.
         * 
         * The socket must be of FAMILY_INET address family, TYPE_RAW socket 
         * type, and PROTOCOL_IP protocol. The socket also must be bound to an 
         * explicit local interface, which means that you cannot bind to 
         * INADDR_ANY.
         *
         * Once the socket is bound and the ioctl set, calls to the Receive 
         * methods return IP datagrams passing through the given interface. 
         * Note that you must supply a sufficiently large buffer. 
         *
         * Setting this ioctl requires Administrator privilege on the local 
         * computer.
         *
         * @param enable
         * 
         * @throws SocketException If the operation fails.
         */
        void SetRcvAll(const bool enable);

        /**
         * Specifies the total per-socket buffer space reserved for receives.
         * This is unrelated to SO_MAX_MSG_SIZE or the size of a TCP window. 
         *
         * @param size The buffer size for receive operations in bytes.
         *
         * @throws SocketException If the operation fails.
         */
        inline void SetRcvBuf(const INT size) {
            this->SetOption(SOL_SOCKET, SO_RCVBUF, &size, sizeof(INT));
        }

        /**
         * Set the receive time-out in milliseconds.
         *
         * Note that this timeout does not affect datagram sockets. Use the
         * special timeouted Receive for receiving datagrams with a timeout.
         *
         * @param timeout The timeout in milliseconds.
         *
         * @throws SocketException If the operation fails.
         */
        inline void SetRcvTimeo(const INT timeout) {
            this->SetOption(SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(INT));
        }

        /**
         * Allows the socket to be bound to an address that is already in use 
         * and in a TIME_WAIT state. This does not allow multiple servers to
         * use the same address.
         *
         * @param enable The new activation state of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline void SetReuseAddr(const bool enable) {
            this->setOption(SOL_SOCKET, SO_REUSEADDR, enable);
        }

        /**
         * Enable or disable a socket to be bound for exclusive access. Does 
         * not require administrative privilege. 
         *
         * This option has no effect but on Windows.
         *
         * @param enable The new activation state of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline void SetExclusiveAddrUse(const bool enable) {
#ifdef _WIN32
            this->setOption(SOL_SOCKET, SO_EXCLUSIVEADDRUSE, enable);
#endif /* _WIN32 */
        }

        /**
         * Sets the adapter that multicast packets shall be received from.
         * Setting IPAddress::ANY will revert to the system default 
         * configuration.
         *
         * @param addr The IPAddress identifying the adapter to receive 
         *             multicast packets from.
         *
         * @throws SocketException If the operation fails.
         */
        inline void SetMulticastInterface(const IPAddress& addr) {
            this->SetOption(IPPROTO_IP, 
                IP_MULTICAST_IF,
                static_cast<const struct in_addr *>(addr), 
                sizeof(struct in_addr));
        };

        /**
         * Enable or disable the receipt of multicast packets sent by this
         * socket to a multicast group the socket is member of.
         *
         * In Winsock, the IP_MULTICAST_LOOP option applies only to the 
         * receive path. 
         * In the UNIX version, the IP_MULTICAST_LOOP option applies to the 
         * send path.
         *
         * @param pf     The protocol (FAMILY_INET, FAMILY_INET6) to 
         *               address.
         * @param enable The new activation state of the option.
         *
         * @throws IllegalParamException If 'pf' is unsupported.
         * @throws SocketException If the operation fails.
         */
        void SetMulticastLoop(const ProtocolFamily pf, const bool enable);

        /**
         * Sets the lifetime of multicast packets.
         *
         * Setting this value to one will restrict multicast packets to the
         * local subnet. Using larger values will enable routing of such 
         * packets.
         *
         * @param pf     The protocol (FAMILY_INET) to address.
         * @param ttl The number of routers multicast packets may pass.
         *
         * @throws IllegalParamException If 'pf' is unsupported.
         * @throws SocketException If the operation fails.
         */
        void SetMulticastTimeToLive(const ProtocolFamily pf, const BYTE ttl);

//IP_ADD_MEMBERSHIP           yes                      no
//IP_DROP_MEMBERSHIP          yes                      no

        /**
         * Specifies the total per-socket buffer space reserved for sends. 
         * This is unrelated to SO_MAX_MSG_SIZE or the size of a TCP window.
         *
         * @param size The buffer size for send operations in bytes.
         *
         * @throws SocketException If the operation fails.
         */
        inline void SetSndBuf(const INT size) {
            this->SetOption(SOL_SOCKET, SO_SNDBUF, &size, sizeof(INT));
        }

        /**
         * Set the send time-out in milliseconds.
         *
         * Note that this timeout does not affect datagram sockets. Use the
         * special timeouted Send method on datagram sockets.
         *
         * @param timeout The timeout in milliseconds.
         *
         * @throws SocketException If the operation fails.
         */
        inline void SetSndTimeo(const INT timeout) {
            this->SetOption(SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(INT));
        }

        /**
         * Disable send or receive operations on the socket.
         *
         * @param how What to shutdown.
         *
         * @throws SocketException If the operation fails.
         */
        virtual void Shutdown(const ShutdownManifest how = BOTH);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        Socket& operator =(const Socket& rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        bool operator ==(const Socket& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const Socket& rhs) const {
            return !(*this == rhs);
        }

    protected:

        /**
         * Answer a boolean socket option.
         *
         * @param level   Level at which the option is defined.
         * @param optName Socket option for which the value is to be retrieved.
         *
         * @return The value of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline bool getOption(const INT level, const INT optName) const {
            INT value = 0;
            SIZE_T valueSize = sizeof(INT);
            this->GetOption(level, optName, &value, valueSize);
            return (value != 0);
        }

        /**
         * Receives 'cntBytes' from the socket and saves them to the memory 
         * designated by 'outData'. 'outData' must be large enough to receive at
         * least 'cntBytes'. 
         *
         * This is a blocking call!
         *
         * Note for Linux: The implementation handles EINTR by itself and 
         * retries the operation until it succeeds or fails with another
         * error.
         *
         * @param outData      The buffer to receive the data. The caller must
         *                     allocate this memory and remains owner.
         * @param cntBytes     The number of bytes to receive.
         * @param flags        The flags that specify the way in which the call 
         *                     is made.
         * @param forceReceive If this flag is set, the method will not return
         *                     until 'cntBytes' have been read.
         *
         * @return The number of bytes actually received.
         *
         * @throws SocketException If the operation fails.
         */
        SIZE_T receive(void *outData, const SIZE_T cntBytes, const INT flags,
            const bool forceReceive);

        /**
         * Receives a datagram from 'fromAddr' and stores it to 'outData'. 
         * 'outData' must be large enough to receive at least 'cntBytes'. 
         *
         * This is a blocking call!
         *
         * Note for Linux: The implementation handles EINTR by itself and 
         * retries the operation until it succeeds or fails with another
         * error.
         *
         * @param outFromAddr  The socket address the datagram was received 
         *                     from. This variable is only valid upon successful
         *                     return from the method.
         * @param outData      The buffer to receive the data. The caller must
         *                     allocate this memory and remains owner.
         * @param cntBytes     The number of bytes to receive.
         * @param flags        The flags that specify the way in which the call 
         *                     is made.
         * @param forceReceive If this flag is set, the method will not return
         *                     until 'cntBytes' have been read.
         *
         * @return The number of bytes actually received.
         *
         * @throws SocketException If the operation fails.
         */
        SIZE_T receiveFrom(IPEndPoint& outFromAddr, void *outData, 
            const SIZE_T cntBytes, const INT flags, const bool forceReceive);

        /**
         * Send 'cntBytes' from the location designated by 'data' using this 
         * socket.
         *
         * This is a blocking call!
         *
         * Note for Linux: The implementation handles EINTR by itself and 
         * retries the operation until it succeeds or fails with another
         * error.
         *
         * @param data      The data to be sent. The caller remains owner of the
         *                  memory.
         * @param cntBytes  The number of bytes to be sent. 'data' must contain
         *                  at least this number of bytes.
         * @param flags     The flags that specify the way in which the call is 
         *                  made.
         * @param forceSend If this flag is set, the method will not return 
         *                  until 'cntBytes' have been sent.
         *
         * @return The number of bytes acutally sent.
         *
         * @throws SocketException If the operation fails.
         */
        SIZE_T send(const void *data, const SIZE_T cntBytes, 
            const INT flags = 0, const bool forceSend = false);

        /**
         * Send a datagram of 'cntBytes' bytes from the location designated by 
         * 'data' using this socket to the socket 'toAddr'.
         *
         * This is a blocking call!
         *
         * Note for Linux: The implementation handles EINTR by itself and 
         * retries the operation until it succeeds or fails with another
         * error.
         *
         * @param toAddr    Socket address of the destination host.
         * @param data      The data to be sent. The caller remains owner of the
         *                  memory.
         * @param cntBytes  The number of bytes to be sent. 'data' must contain
         *                  at least this number of bytes.
         * @param flags     The flags that specify the way in which the call is 
         *                  made.
         * @param forceSend If this flag is set, the method will not return 
         *                  until 'cntBytes' have been sent.
         *
         * @return The number of bytes acutally sent.
         *
         * @throws SocketException If the operation fails.
         */
        SIZE_T sendTo(const IPEndPoint& toAddr, const void *data, 
            const SIZE_T cntBytes, const INT flags, const bool forceSend);

        /**
         * Set a boolean socket option.
         *
         * @param level   Level at which the option is defined.
         * @param optName Socket option for which the value is to be set.
         * @param value   The new value of the option.
         *
         * @throws SocketException If the operation fails.
         */
        inline void setOption(const INT level, const INT optName, 
                const bool value) {
            INT tmp = value ? 1 : 0;
            return this->SetOption(level, optName, &tmp, sizeof(INT));
        }

        /** The socket handle. */
        SOCKET handle;

        /** The asynchronous sender must be able to access the handle. */
        friend class AsyncSocketSender;
    };

} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SOCKET_H_INCLUDED */
