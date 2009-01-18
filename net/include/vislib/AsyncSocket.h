/*
 * AsyncSocket.h
 *
 * Copyright (C) 2006 - 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ASYNCSOCKET_H_INCLUDED
#define VISLIB_ASYNCSOCKET_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Socket.h"  // Must be first.
#include "vislib/CriticalSection.h"
#include "vislib/Runnable.h"
#include "vislib/StackTrace.h"
#include "vislib/ThreadPool.h"


namespace vislib {
namespace net {

    /* Forward declarations. */
    class AsyncSocketContext;


    /**
     * The AsyncSocket specialisation allows for executing send and receive 
     * operations in an asynchronous manner.
     *
     * It is safe to call the synchronous methods on this socket, too.
     *
     * Rationale: The asynchronous operations are not included in the standard
     * socket class as these require additional resources that are not required
     * for a purely synchronous socket.
     *
     * Rationale: On Linux, the implementation is done via separate threads 
     * rather than using AIO, because the documentation on the behaviour of AIO
     * on socket handles is more than sketchy. We assume that it does not work
     * after reading a lot of posts in newsgroups and mailing lists.
     */
    class AsyncSocket : public Socket {

    public:

        /**
         * Create an invalid socket. Call Create() on the new object to create
         * a new socket.
         */
        inline AsyncSocket(void) : Super() {
            VISLIB_STACKTRACE(AsyncSocket, __FILE__, __LINE__);
        }

        /**
         * Create socket wrapper from an existing handle.
         *
         * @param handle The socket handle.
         */
        explicit inline AsyncSocket(SOCKET handle) : Super(handle) {
            VISLIB_STACKTRACE(AsyncSocket, __FILE__, __LINE__);
        }

        /**
         * Create an AsyncSocket wrapper for an existing socket.
         *
         * @param socket The socket to be cloned
         */
        explicit inline AsyncSocket(Socket socket) : Super(socket) {
            VISLIB_STACKTRACE(AsyncSocket, __FILE__, __LINE__);
        }

        /**
         * Create a copy of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline AsyncSocket(const AsyncSocket& rhs) : Super(rhs) {
            VISLIB_STACKTRACE(AsyncSocket, __FILE__, __LINE__);
        }

        /** Dtor. */
        virtual ~AsyncSocket(void);

        /**
         * Receives 'cntBytes' from the socket and saves them to the memory 
         * designated by 'outData'. 'outData' must be large enough to receive at
         * least 'cntBytes'. 
         *
         * @param outData      The buffer to receive the data. The caller must
         *                     allocate this memory and remains owner.
         * @param cntBytes     The number of bytes to receive.
         * @param context      The AsyncSocketContext that is used to complete
         *                     the operation. This context allows for 
         *                     notification via a callback function or for 
         *                     waiting synchronously. The caller remains owner
         *                     of the memory designated by 'context' and must
         *                     ensure that it exists until a call to 
         *                     EndReceive() was made.
         * @param timeout      A timeout in milliseconds. A value less than 1 
         *                     specifies an infinite timeout. If the operation 
         *                     timeouts, an exception will be thrown.
         * @param flags        The flags that specify the way in which the call 
         *                     is made.
         *
         * @throws SocketException       If the operation fails.
         * @throws IllegalParamException If 'context' is a NULL pointer.
         */
        void BeginReceive(void *outData, const SIZE_T cntBytes,
            AsyncSocketContext *context, const INT timeout = TIMEOUT_INFINITE, 
            const INT flags = 0);

        /**
         * Receives a datagram from 'fromAddr' and stores it to 'outData'. 
         * 'outData' must be large enough to receive at least 'cntBytes'. 
         *
         * This method can only be used on datagram sockets.
         *
         * @param outFromAddr  The socket address the datagram was received 
         *                     from. This must not be NULL.
         * @param outData      The buffer to receive the data. The caller must
         *                     allocate this memory and remains owner.
         * @param cntBytes     The number of bytes to receive.
         * @param context      The AsyncSocketContext that is used to complete
         *                     the operation. This context allows for 
         *                     notification via a callback function or for 
         *                     waiting synchronously. The caller remains owner
         *                     of the memory designated by 'context' and must
         *                     ensure that it exists until a call to 
         *                     EndReceive() was made.
         * @param timeout      A timeout in milliseconds. A value less than 1 
         *                     specifies an infinite timeout. If the operation 
         *                     timeouts, an exception will be thrown.
         * @param flags        The flags that specify the way in which the call 
         *                     is made.
         *
         * @throws SocketException       If the operation fails.
         * @throws IllegalParamException If 'outFromAddr' or 'context' is a 
         *                               NULL pointer.
         */
        void BeginReceive(IPEndPoint *outFromAddr, void *outData, 
            const SIZE_T cntBytes, AsyncSocketContext *context, 
            const INT timeout = TIMEOUT_INFINITE, const INT flags = 0);

        /**
         * Send 'cntBytes' from the location designated by 'data' using this 
         * socket.
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
        void BeginSend(const void *data, const SIZE_T cntBytes,
            AsyncSocketContext *context, const INT timeout = TIMEOUT_INFINITE, 
            const INT flags = 0);

        /**
         * Send a datagram of 'cntBytes' bytes from the location designated by 
         * 'data' using this socket to the socket 'toAddr'.
         *
         * This method can only be used on datagram sockets.
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
        void BeginSend(const IPEndPoint& toAddr, const void *data, 
            const SIZE_T cntBytes, AsyncSocketContext *context, 
            const INT timeout = TIMEOUT_INFINITE, const INT flags = 0);

        /**
         * Completes an asynchronous receive operation. This method must be 
         * called to complete the operation once 'context' was signaled or 
         * within the callback function.
         *
         * @param context The context object of an asynchronous receive 
         *                operation.
         *
         * @return The number of bytes actually received.
         *
         * @throws SocketException       If the previously initiated operation 
         *                               failed.
         * @throws IllegalParamException If 'context' is a NULL pointer.
         */
        SIZE_T EndReceive(AsyncSocketContext *context);

        /**
         * Completes an asynchronous receive operation. This method must be 
         * called to complete the operation once 'context' was signaled or 
         * within the callback function.
         *
         * @param context The context object of an asynchronous receive 
         *                operation.
         *
         * @return The number of bytes actually received.
         *
         * @throws SocketException       If the previously initiated operation 
         *                               failed.
         * @throws IllegalParamException If 'context' is a NULL pointer.
         */
        SIZE_T EndSend(AsyncSocketContext *context);


#ifndef _WIN32
        /**
         * Receives 'cntBytes' from the socket and saves them to the memory 
         * designated by 'outData'. 'outData' must be large enough to receive at
         * least 'cntBytes'. 
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
         * @throws IllegalParamException If 'timeout' is not TIMEOUT_INFINITE 
         *                               and 'forceReceive' is true.
         */
        virtual SIZE_T Receive(void *outData, const SIZE_T cntBytes, 
            const INT timeout = TIMEOUT_INFINITE, const INT flags = 0, 
            const bool forceReceive = false);

        /**
         * Receives a datagram from 'fromAddr' and stores it to 'outData'. 
         * 'outData' must be large enough to receive at least 'cntBytes'. 
         *
         * This method can only be used on datagram sockets.
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
         * @throws IllegalParamException If 'timeout' is not TIMEOUT_INFINITE 
         *                               and 'forceReceive' is true.
         */
        virtual SIZE_T Receive(IPEndPoint& outFromAddr, void *outData, 
            const SIZE_T cntBytes, const INT timeout = TIMEOUT_INFINITE, 
            const INT flags = 0, const bool forceReceive = false);

        /**
         * Receives a datagram from 'fromAddr' and stores it to 'outData'. 
         * 'outData' must be large enough to receive at least 'cntBytes'. 
         *
         * This method can only be used on datagram sockets.
         *
         * This method is for backward compatibilty and is only supported on 
         * IPv4 sockets. Use IPEndPoint instead of SocketAddress for IPv6 
         * support and better performance.
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
         * @throws IllegalParamException If 'timeout' is not TIMEOUT_INFINITE 
         *                               and 'forceReceive' is true.
         */
        virtual SIZE_T Receive(SocketAddress& outFromAddr, void *outData, 
            const SIZE_T cntBytes, const INT timeout = TIMEOUT_INFINITE, 
            const INT flags = 0, const bool forceReceive = false);

        /**
         * Send 'cntBytes' from the location designated by 'data' using this 
         * socket.
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
         * This method can only be used on datagram sockets.
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
#endif /* !_WIN32 */

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AsyncSocket& operator =(const AsyncSocket& rhs) {
            VISLIB_STACKTRACE(operator=, __FILE__, __LINE__);
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        inline bool operator ==(const AsyncSocket& rhs) const {
            VISLIB_STACKTRACE(operator==, __FILE__, __LINE__);
            return Super::operator ==(rhs);
        }

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const AsyncSocket& rhs) const {
            VISLIB_STACKTRACE(operator!=, __FILE__, __LINE__);
            return !(*this == rhs);
        }

    protected:
        
        /** Superclass typedef. */
        typedef Socket Super;

#ifdef _WIN32
        static void CALLBACK completedFunc(DWORD dwError, DWORD cbTransferred, 
            LPWSAOVERLAPPED lpOverlapped, DWORD dwFlags);
#endif /* _WIN32 */

//#ifndef _WIN32
        static DWORD receiveFunc(void *asyncSocketContext);

        static DWORD sendFunc(void *asyncSocketContext);

        /** This lock ensures exclusive receive access on Linux. */
        vislib::sys::CriticalSection lockRecv;

        /** This lock ensures exclusive send access on Linux. */
        vislib::sys::CriticalSection lockSend;

        /** The thread pool that runs the asynchronous operations on Linux. */
        vislib::sys::ThreadPool threadPool;
//#endif /* !_WIN32 */

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ASYNCSOCKET_H_INCLUDED */
