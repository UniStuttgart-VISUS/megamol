/*
 * AsyncSocketContext.h
 *
 * Copyright (C) 2009 by Visualisierungsinstitut der Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ASYNCSOCKETCONTEXT_H_INCLUDED
#define VISLIB_ASYNCSOCKETCONTEXT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <winsock2.h>
#endif /* _WIN32 */

#include "vislib/AbstractAsyncContext.h"
#include "vislib/assert.h"
#include "vislib/Event.h"
#include "vislib/IPEndPoint.h"
#include "vislib/Socket.h"
#include "vislib/StackTrace.h"
#include "vislib/UnsupportedOperationException.h"


namespace vislib {
namespace net {

    /* Forward declarations. */
    class AsyncSocket;


    /**
     * This class is used for handling asynchronous socket operations.
     *
     * AsyncSocketContext can be used to wait synchronously for an asynchronous
     * socket operation to complete like in the following example:
     * <code>
     * vislib::net::AsyncSocket socket;
     * vislib::net::AsyncSocketContext context;
     * BYTE data[16];
     *
     * // Setup the socket and connect.
     *
     * context.Reset();     // Optional if you do not reuse 'context'.
     * socket.BeginReceive(&data, sizeof(data), &context);
     *
     * // Do some other stuff here. (Otherwise, asnychronous I/O would not make
     * // any sense.)
     *
     * context.Wait();
     * socket.EndReceive(&context);
     * </code>
     *
     * Alternatively, the asynchronous callback function of the context
     * can be used to complete the operation. In this case, the caller is
     * required to ensure that the context exists until the asynchronous
     * operation completes. As the socket is required in the callback, it
     * must be passed with the AbstractAsyncContext, either directly as the
     * 'userContext' or as a member of a structure or class that is the
     * 'userContext'. The following example illustrates how the Socket
     * is passed as the 'userContext' directly.
     * <code>
     * void OnReceiveCompleted(AbstractAsyncContext *context) {
     *     vislib::net::AsyncSocket *socket 
     *          = static_cast<vislib::net::AsyncSocket>(
     *          context->GetUserContext());
     *     socket->EndReceive(context);
     * }
     *
     * // Somewhere else:
     * vislib::net::AsyncSocket socket;
     * BYTE data[16];
     *
     * // Setup the socket and connect.
     *
     * vislib::net::AsyncSocketContext context(OnReceiveCompleted, &socket);
     * socket.BeginReceive(&data, sizeof(data), &context);
     * </code>
     */
    class AsyncSocketContext : public vislib::sys::AbstractAsyncContext {

    public:

        /**
         * This function pointer defines the callback that is called once
         * the operation was completed.
         */
        typedef vislib::sys::AbstractAsyncContext::AsyncCallback AsyncCallback;

        /** Ctor. */
        AsyncSocketContext(AsyncCallback callback = NULL, 
            void *userContext = NULL);

        /** Dtor. */
        virtual ~AsyncSocketContext(void);

        /**
         * Reset the state of the context. If a context object is reused for
         * more than one asynchronous operation, Reset() must be called before
         * every operation.
         *
         * Reset() must not be called between the begin and end of an 
         * asynchronous operation. The results of the call are undefined in this
         * case.
         */
        virtual void Reset(void);

        /**
         * Wait for the operation associated with this context to complete.
         *
         * @throws SystemException If the operation failed.
         */
        virtual void Wait(void);

    protected:

        /** Superclass typedef. */
        typedef vislib::sys::AbstractAsyncContext Super;

#ifdef _WIN32
        /**
         * Provide access to the WSAOVERLAPPED structure of the operating system
         * that is associated with this request. 
         *
         * This WSAOVERLAPPED structure is used to initiate the asynchronous
         * Winsock operation.
         *
         * @return A pointer to the WSAOVERLAPPED structure.
         */
        operator WSAOVERLAPPED *(void) {
            VLSTACKTRACE("AsyncSocketContext::operator WSAOVERLAPPED *", 
                __FILE__, __LINE__);
            ASSERT(sizeof(WSAOVERLAPPED) == sizeof(OVERLAPPED));
            return (Super::operator OVERLAPPED *());
        }
#endif /* _WIN32 */

        /**
         * Completes the asynchronous operation by setting the result parameters
         * 'cntData' and 'errorCode', calling any registered callback function and
         * signaling the event object.
         *
         * Implementation note: The parameters 'cntData' and 'errorCode' are 
         * ignored on Windows, as this information can be retrieved using 
         * WSAGetOverlappedResult in AsyncSocket::endAsync().
         *
         * @param cntData   The amount of data acutally sent/received.
         * @param errorCode 0 in case the operation completed successfully, an 
         *                  appropriate system error code otherwise.
         */
        virtual void notifyCompleted(const DWORD cntData, 
            const DWORD errorCode);

#if (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN))
        /**
         * Pass the input parameters for an asynchronous datagram operation.
         *
         * @param socket    The socket to be used.
         * @param dgramAddr The peer address. This must not be NULL.
         * @param data      Pointer to the data buffer.
         * @param cntData   Size of the data buffer.
         * @param flags     Socket flags.
         * @param timeout   Timeout for the operation.
         */
        inline void setDgramParams(AsyncSocket *socket, 
                const IPEndPoint *dgramAddr, const void *data, 
                const SIZE_T cntData, const INT flags, const INT timeout) {
            VLSTACKTRACE("AsyncSocketContext::setDgramParams", __FILE__, 
                __LINE__);
            ASSERT(dgramAddr != NULL);
            this->socket = socket;
            this->dgramAddrOrg = const_cast<IPEndPoint *>(dgramAddr);
            this->dgramAddrCpy = *this->dgramAddrOrg;
            this->data = const_cast<void *>(data);
            this->cntData = cntData;
            this->flags = flags;
            this->timeout = timeout;
        }

        /**
         * Pass the input parameters for an asynchronous stream operation.
         *
         * @param socket    The socket to be used.
         * @param data      Pointer to the data buffer.
         * @param cntData   Size of the data buffer.
         * @param flags     Socket flags.
         * @param timeout   Timeout for the operation.         */
        inline void setStreamParams(AsyncSocket *socket, const void *data, 
                const SIZE_T cntData, const INT flags, const INT timeout) {
            VLSTACKTRACE("AsyncSocketContext::setStreamParams", __FILE__, 
                __LINE__);
            this->socket = socket;
            this->dgramAddrOrg = NULL;
            this->data = const_cast<void *>(data);
            this->cntData = cntData;
            this->flags = flags;
            this->timeout = timeout;
        }

#else /* (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */
        /**
         * Pass the input parameters for an asynchronous operation.
         *
         * @param data      Pointer to the data buffer.
         * @param cntData   Size of the data buffer.
         */
        inline void setWsaParams(const void *data, const SIZE_T cntData) {
            VLSTACKTRACE("AsyncSocketContext::setWsaParams", __FILE__, 
                __LINE__);
            wsaBuf.buf = static_cast<char *>(const_cast<void *>(data));
            wsaBuf.len = static_cast<u_long>(cntData);
        }
#endif /* (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */

    private:

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        inline AsyncSocketContext(const AsyncSocketContext& rhs)
                : Super(NULL, NULL) {
            VLSTACKTRACE("AsyncSocketContext::AsyncSocketContext", __FILE__, 
                __LINE__);
            throw UnsupportedOperationException("AsyncSocketContext", __FILE__, 
                __LINE__);
        }

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        AsyncSocketContext& operator =(const AsyncSocketContext& rhs);

        /** The event that is signaled on completion. */
        vislib::sys::Event evt;
        
#if (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN))
        /** 
         * The size of the data packet sent or received.
         * This variable is used to describe the size of the input 
         * parameter as well as the return value of the operation.
         */
        int cntData;

        /**
         * An error code that may have been raised by the asynchronous send or
         * receive operation.
         */
        int errorCode;

        /** 
         * Passes the data pointer from AsyncSocket::BeginSend() or 
         * AsyncSocket::BeginReceive() to the worker thread.
         */
        void *data; 

        /**
         * The address of the peer address in case of a datagram socket 
         * operation. Otherwise, this parameter must be NULL (for stream
         * sockets).
         */
        IPEndPoint *dgramAddrOrg;

        /** The socket that is used to send/receive data. */
        AsyncSocket *socket;

        /**
         * A deep copy of 'dgramAddrOrg' that is used only for send 
         * operations.
         */
        IPEndPoint dgramAddrCpy;

        /** 
         * Passes the socket flags from AsyncSocket::BeginSend() or 
         * AsyncSocket::BeginReceive() to the worker thread.
         */
        INT flags;

        /** 
         * Passes the timeout from AsyncSocket::BeginSend() or 
         * AsyncSocket::BeginReceive() to the worker thread.
         */
        INT timeout;

#else /* (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */
        /** 
         * The input/ouput buffer that is used for the asynchronous operation.
         * This variable is placed in the context structure as it must remain
         * valid until the operation completes.
         */
        WSABUF wsaBuf;
#endif /* (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */

        /** Allow access to protected cast operation to the socket. */
        friend class AsyncSocket;
    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ASYNCSOCKETCONTEXT_H_INCLUDED */
