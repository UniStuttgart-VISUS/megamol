/*
 * AsyncSocket.cpp
 *
 * Copyright (C) 2009 by Visualisierungsinstitut der Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/AsyncSocket.h"

#include "vislib/AsyncSocketContext.h"
#include "vislib/error.h"
#include "the/argument_exception.h"
#include "vislib/SocketException.h"
#include "the/system/system_exception.h"
#include "the/trace.h"
#include "vislib/unreferenced.h"


/*
 * vislib::net::AsyncSocket::~AsyncSocket
 */
vislib::net::AsyncSocket::~AsyncSocket(void) {
    THE_STACK_TRACE;

#if (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN))
    try {
        this->threadPool.Terminate(true);
    } catch (...) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Exception in AsyncSocket dtor.");
    }
#endif /* (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */
}


/*
 * vislib::net::AsyncSocket::BeginReceive
 */
void vislib::net::AsyncSocket::BeginReceive(void *outData, 
        const size_t cntBytes, AsyncSocketContext *context, const int timeout, 
        const int flags) {
    THE_STACK_TRACE;

    /* Sanity checks. */
    if (context == NULL) {
        throw the::argument_exception("context", __FILE__, __LINE__);
    }

#if (defined(_WIN32) && !defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN))
    DWORD errorCode = 0;        // WSA error during last operation.
    DWORD cntReceived = 0;      // Receives # of received bytes.
    DWORD inOutFlags = flags;   // Flags for WSA.

    context->setWsaParams(outData, cntBytes);

    if (::WSARecv(this->handle, &context->wsaBuf, 1, 
            reinterpret_cast<DWORD *>(&cntReceived), &inOutFlags, 
            static_cast<WSAOVERLAPPED *>(*context), AsyncSocket::completedFunc)
            != 0) {
        if ((errorCode = ::WSAGetLastError()) != WSA_IO_PENDING) {
            throw SocketException(errorCode, __FILE__, __LINE__);
        }
    } else {
        context->notifyCompleted(cntReceived, ::WSAGetLastError());
    }

#else /* (defined(_WIN32) && !defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */
    context->setStreamParams(this, outData, cntBytes, timeout, flags);
    this->threadPool.QueueUserWorkItem(AsyncSocket::receiveFunc, context);
#endif /* (defined(_WIN32) && !defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */
}


/*
 * vislib::net::AsyncSocket::BeginReceive
 */
void vislib::net::AsyncSocket::BeginReceive(IPEndPoint *outFromAddr, 
        void *outData, const size_t cntBytes, AsyncSocketContext *context, 
        const int timeout, const int flags) {
    THE_STACK_TRACE;

    /* Sanity checks. */
    if (outFromAddr == NULL) {
        throw the::argument_exception("outFromAddr", __FILE__, __LINE__);
    }
    if (context == NULL) {
        throw the::argument_exception("context", __FILE__, __LINE__);
    }

#if (defined(_WIN32) && !defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN))
    DWORD errorCode = 0;        // WSA error during last operation.
    DWORD cntReceived = 0;      // Receives # of received bytes.
    DWORD inOutFlags = flags;   // Flags for WSA.
    int fromLen = sizeof(struct sockaddr_storage);

    context->setWsaParams(outData, cntBytes);

    if (::WSARecvFrom(this->handle, &context->wsaBuf, 1, 
            reinterpret_cast<DWORD *>(&cntReceived), &inOutFlags, 
            static_cast<sockaddr *>(*outFromAddr), &fromLen,
            static_cast<WSAOVERLAPPED *>(*context), AsyncSocket::completedFunc)
            != 0) {
        if ((errorCode = ::WSAGetLastError()) != WSA_IO_PENDING) {
            throw SocketException(errorCode, __FILE__, __LINE__);
        }
    } else {
        context->notifyCompleted(cntReceived, ::WSAGetLastError());
    }
#else /* (defined(_WIN32) && !defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */
    context->setDgramParams(this, outFromAddr, outData, cntBytes, timeout, 
        flags);
    this->threadPool.QueueUserWorkItem(AsyncSocket::receiveFunc, context);
#endif /* (defined(_WIN32) && !defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */
}


/*
 * vislib::net::AsyncSocket::BeginSend
 */
void vislib::net::AsyncSocket::BeginSend(const void *data, 
        const size_t cntBytes, AsyncSocketContext *context, const int timeout,
        const int flags) {
    THE_STACK_TRACE;

    /* Sanity checks. */
    if (context == NULL) {
        throw the::argument_exception("context", __FILE__, __LINE__);
    }

#if (defined(_WIN32) && !defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN))
    DWORD errorCode = 0;        // WSA error during last operation.
    DWORD cntSent = 0;          // Receives # of sent bytes.

    context->setWsaParams(data, cntBytes);

    if (::WSASend(this->handle, &context->wsaBuf, 1, 
            reinterpret_cast<DWORD *>(&cntSent), flags, 
            static_cast<WSAOVERLAPPED *>(*context), AsyncSocket::completedFunc)
            != 0) {
        if ((errorCode = ::WSAGetLastError()) != WSA_IO_PENDING) {
            throw SocketException(errorCode, __FILE__, __LINE__);
        }
    } else {
        context->notifyCompleted(cntSent, ::WSAGetLastError());
    }
#else /* (defined(_WIN32) && !defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */
    context->setStreamParams(this, data, cntBytes, timeout, flags);
    this->threadPool.QueueUserWorkItem(AsyncSocket::sendFunc, context);
#endif /* (defined(_WIN32) && !defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */
}


/*
 * vislib::net::AsyncSocket::BeginSend
 */
void vislib::net::AsyncSocket::BeginSend(const IPEndPoint& toAddr, 
        const void *data, const size_t cntBytes, AsyncSocketContext *context,
        const int timeout, const int flags) {
    THE_STACK_TRACE;

    /* Sanity checks. */
    if (context == NULL) {
        throw the::argument_exception("context", __FILE__, __LINE__);
    }

#if (defined(_WIN32) && !defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN))
    DWORD errorCode = 0;        // WSA error during last operation.
    DWORD cntSent = 0;          // Receives # of sent bytes.
    int toLen = sizeof(struct sockaddr_storage);

    context->setWsaParams(data, cntBytes);

    if (::WSASendTo(this->handle, &context->wsaBuf, 1, 
            reinterpret_cast<DWORD *>(&cntSent), flags,
            static_cast<const sockaddr *>(toAddr), toLen,
            static_cast<WSAOVERLAPPED *>(*context), AsyncSocket::completedFunc)
            != 0) {
        if ((errorCode = ::WSAGetLastError()) != WSA_IO_PENDING) {
            throw SocketException(errorCode, __FILE__, __LINE__);
        }
    } else {
        context->notifyCompleted(cntSent, ::WSAGetLastError());
    }
#else /* (defined(_WIN32) && !defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */
    throw 1;
    context->setDgramParams(this, &toAddr, data, cntBytes, timeout, flags);
    this->threadPool.QueueUserWorkItem(AsyncSocket::receiveFunc, context);
#endif /* (defined(_WIN32) && !defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */
}


#if (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN))
/*
 * vislib::net::AsyncSocket::Receive
 */
size_t vislib::net::AsyncSocket::Receive(void *outData, const size_t cntBytes,
        const int timeout, const int flags, const bool forceReceive) {
    THE_STACK_TRACE;
    size_t retval = 0;

    this->lockRecv.Lock();
    try {
        retval = Super::Receive(outData, cntBytes, timeout, flags, forceReceive);
    } catch (...) {
        this->lockRecv.Unlock();
        throw;
    }

    this->lockRecv.Unlock();
    return retval;
}


/*
 * vislib::net::AsyncSocket::Receive
 */
size_t vislib::net::AsyncSocket::Receive(IPEndPoint& outFromAddr, void *outData,
        const size_t cntBytes, const int timeout, const int flags, 
        const bool forceReceive) {
    THE_STACK_TRACE;
    size_t retval = 0;

    this->lockRecv.Lock();
    try {
        retval = Super::Receive(outFromAddr, outData, cntBytes, timeout, flags,
            forceReceive);
    } catch (...) {
        this->lockRecv.Unlock();
        throw;
    }

    this->lockRecv.Unlock();
    return retval;
}


/*
 * vislib::net::AsyncSocket::Receive
 */
size_t vislib::net::AsyncSocket::Receive(SocketAddress& outFromAddr, 
        void *outData, const size_t cntBytes, const int timeout, 
        const int flags, const bool forceReceive) {
    THE_STACK_TRACE;
    size_t retval = 0;

    this->lockRecv.Lock();
    try {
        retval = Super::Receive(outFromAddr, outData, cntBytes, timeout, flags, 
            forceReceive);
    } catch (...) {
        this->lockRecv.Unlock();
        throw;
    }

    this->lockRecv.Unlock();
    return retval;
}


/*
 * vislib::net::AsyncSocket::Send
 */
size_t vislib::net::AsyncSocket::Send(const void *data, const size_t cntBytes, 
        const int timeout, const int flags, const bool forceSend) {
    THE_STACK_TRACE;
    size_t retval = 0;

    this->lockSend.Lock();
    try {
        retval = Super::Send(data, cntBytes, timeout, flags, forceSend);
    } catch (...) {
        this->lockSend.Unlock();
        throw;
    }

    this->lockSend.Unlock();
    return retval;
}


/*
 * vislib::net::AsyncSocket::Send
 */
size_t vislib::net::AsyncSocket::Send(const IPEndPoint& toAddr, const void *data, 
        const size_t cntBytes, const int timeout, const int flags, 
        const bool forceSend) {
    THE_STACK_TRACE;
    size_t retval = 0;

    this->lockSend.Lock();
    try {
        retval = Super::Send(toAddr, data, cntBytes, timeout, flags, forceSend);
    } catch (...) {
        this->lockSend.Unlock();
        throw;
    }

    this->lockSend.Unlock();
    return retval;
}
#endif /* (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */


#ifdef _WIN32
/*
 * vislib::net::AsyncSocket::completedFunc
 */
void CALLBACK vislib::net::AsyncSocket::completedFunc(DWORD dwError, 
        DWORD cbTransferred, LPWSAOVERLAPPED lpOverlapped, DWORD dwFlags) {
    THE_STACK_TRACE;
    AsyncSocketContext *ctx = reinterpret_cast<AsyncSocketContext *>(
        lpOverlapped->hEvent);
    ctx->notifyCompleted(cbTransferred, dwError);
}
#endif /* _WIN32 */


#if (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN))
/*
 * vislib::net::AsyncSocket::receiveFunc
 */
unsigned int vislib::net::AsyncSocket::receiveFunc(void *asyncSocketContext) {
    THE_STACK_TRACE;
    AsyncSocketContext *ctx = static_cast<AsyncSocketContext *>(
        asyncSocketContext);
    unsigned int retval = 0;   // The return value (error code of socket call).
    unsigned int cnt = 0;      // The number of bytes received.

    // Assert this as we require the overridden methods for locking!
    // The call to the overloaded ctx->socket->Receive will ensure locking, 
    // we do not have to do this ourselves in the thread function.
    THE_ASSERT(dynamic_cast<AsyncSocket *>(ctx->socket) != NULL);

    try {
        Socket::Startup();
    } catch (SocketException e) {
        return e.get_error().native_error();
    }

    try {
        if (ctx->dgramAddrOrg != NULL) {
            /* Receive on datagram socket. */
            cnt = ctx->socket->Receive(*ctx->dgramAddrOrg, ctx->data, 
                ctx->cntData, ctx->timeout, ctx->flags, false);
        } else {
            /* Receive on stream socket. */
            cnt = ctx->socket->Receive(ctx->data, ctx->cntData, 
                ctx->timeout, ctx->flags, false);
        }

        ctx->notifyCompleted(cnt, 0);
    } catch (the::system::system_exception& e) {
        retval = e.get_error().native_error();
        ctx->notifyCompleted(cnt, retval);
    } catch (...) {
        ctx->notifyCompleted(0, ::GetLastError());
        THE_ASSERT(false);
    }

    try {
        Socket::Cleanup();
    } catch (SocketException e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Socket::Cleanup failed in AsyncSocket "
            "AsyncSocket::receiveFunc.");
    }

    return retval;
}


/*
 * vislib::net::AsyncSocket::sendFunc
 */
unsigned int vislib::net::AsyncSocket::sendFunc(void *asyncSocketContext) {
    THE_STACK_TRACE;
    AsyncSocketContext *ctx = static_cast<AsyncSocketContext *>(
        asyncSocketContext);
    unsigned int retval = 0;   // The return value (error code of socket call).
    unsigned int cnt = 0;      // The number of bytes sent.

    // Assert this as we require the overridden methods for locking!
    // The call to the overloaded ctx->socket->Send will ensure locking,
    // we do not have to do this ourselves in the thread function.
    THE_ASSERT(dynamic_cast<AsyncSocket *>(ctx->socket) != NULL);

    try {
        Socket::Startup();
    } catch (SocketException e) {
        return e.get_error().native_error();
    }

    try {
        if (ctx->dgramAddrOrg != NULL) {
            /* Send on datagram socket. */
            cnt = ctx->socket->Send(ctx->dgramAddrCpy, ctx->data, 
                ctx->cntData, ctx->timeout, ctx->flags, false);
        } else {
            /* Send on stream socket. */
            cnt = ctx->socket->Send(ctx->data, ctx->cntData, 
                ctx->timeout, ctx->flags, false);
        }

        ctx->notifyCompleted(cnt, 0);
    } catch (the::system::system_exception& e) {
        retval = e.get_error().native_error();
        ctx->notifyCompleted(cnt, retval);
    } catch (...) {
        ctx->notifyCompleted(0, ::GetLastError());
        THE_ASSERT(false);
    }

    try {
        Socket::Cleanup();
    } catch (SocketException e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Socket::Cleanup failed in AsyncSocket "
            "AsyncSocket::sendFunc.");
    }

    return retval;
}
#endif /* (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */


/*
 * vislib::net::AsyncSocket::endAsync
 */
size_t vislib::net::AsyncSocket::endAsync(AsyncSocketContext *context) {
    THE_STACK_TRACE;

#ifdef _WIN32
    DWORD retval = 0;
    DWORD flags = 0;
#else /* _WIN32 */
    unsigned int retval = 0;
    unsigned int flags = 0;
    VL_UNREFERENCED_LOCAL_VARIABLE(flags);
#endif /* _WIN32 */

    /* Sanity checks. */
    if (context == NULL) {
        throw the::argument_exception("context", __FILE__, __LINE__);
    }

    /* Check for success of the operation. */
#if (defined(_WIN32) && !defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN))
    if (!::WSAGetOverlappedResult(this->handle, 
            static_cast<WSAOVERLAPPED *>(*context), &retval, TRUE, &flags)) {
        throw SocketException(__FILE__, __LINE__);
    }
#else /* (defined(_WIN32) && !defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */
    if (context->errorCode != 0) {
        throw SocketException(context->errorCode, __FILE__, __LINE__);
    }
    retval = static_cast<unsigned int>(context->cntData);
#endif /* (defined(_WIN32) && !defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */

    return static_cast<size_t>(retval);
}
