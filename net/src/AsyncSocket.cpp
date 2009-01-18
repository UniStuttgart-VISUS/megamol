/*
 * AsyncSocket.cpp
 *
 * Copyright (C) 2006 - 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/AsyncSocket.h"

#include "vislib/AsyncSocketContext.h"
#include "vislib/IllegalParamException.h"
#include "vislib/SocketException.h"
#include "vislib/Trace.h"


/*
 * vislib::net::AsyncSocket::~AsyncSocket
 */
vislib::net::AsyncSocket::~AsyncSocket(void) {
    VISLIB_STACKTRACE(~AsyncSocket, __FILE__, __LINE__);

    try {
        this->threadPool.Terminate(true);
    } catch (...) {
        VLTRACE(Trace::LEVEL_VL_WARN, "Exception in AsyncSocket dtor.");
    }
}


/*
 * vislib::net::AsyncSocket::BeginReceive
 */
void vislib::net::AsyncSocket::BeginReceive(void *outData, 
        const SIZE_T cntBytes, AsyncSocketContext *context, const INT timeout, 
        const INT flags) {
    VISLIB_STACKTRACE(BeginReceive, __FILE__, __LINE__);

    /* Sanity checks. */
    if (context == NULL) {
        throw IllegalParamException("context", __FILE__, __LINE__);
    }

#ifdef _WIN32
    WSABUF wsaBuf;              // Buffer for WSA output.
    DWORD errorCode = 0;        // WSA error during last operation.
    DWORD cntReceived = 0;      // Receives # of received bytes.
    DWORD inOutFlags = flags;   // Flags for WSA.

    wsaBuf.buf = static_cast<char *>(outData);
    wsaBuf.len = static_cast<u_long>(cntBytes);

    if (::WSARecv(this->handle, &wsaBuf, 1, 
            reinterpret_cast<DWORD *>(&cntReceived), &inOutFlags, 
            static_cast<WSAOVERLAPPED *>(*context), AsyncSocket::completedFunc)
            != 0) {
        if ((errorCode = ::WSAGetLastError()) != WSA_IO_PENDING) {
            throw SocketException(errorCode, __FILE__, __LINE__);
        }
    } else {
        context->notifyCompleted(cntReceived, ::WSAGetLastError());
    }
#else /* _WIN32 */
    context->setStreamParams(this, outData, cntBytes, timeout, flags);
    this->threadPool.QueueUserWorkItem(AsyncSocket::receiveFunc, context);
#endif /* _WIN32 */
}


/*
 * vislib::net::AsyncSocket::BeginReceive
 */
void vislib::net::AsyncSocket::BeginReceive(IPEndPoint *outFromAddr, 
        void *outData, const SIZE_T cntBytes, AsyncSocketContext *context, 
        const INT timeout, const INT flags) {
    VISLIB_STACKTRACE(BeginReceive, __FILE__, __LINE__);

    /* Sanity checks. */
    if (outFromAddr == NULL) {
        throw IllegalParamException("outFromAddr", __FILE__, __LINE__);
    }
    if (context == NULL) {
        throw IllegalParamException("context", __FILE__, __LINE__);
    }

    context->setDgramParams(this, outFromAddr, outData, cntBytes, timeout, 
        flags);
    this->threadPool.QueueUserWorkItem(AsyncSocket::receiveFunc, context);
}


/*
 * vislib::net::AsyncSocket::BeginSend
 */
void vislib::net::AsyncSocket::BeginSend(const void *data, 
        const SIZE_T cntBytes, AsyncSocketContext *context, const INT timeout,
        const INT flags) {
    VISLIB_STACKTRACE(BeginSend, __FILE__, __LINE__);

    /* Sanity checks. */
    if (context == NULL) {
        throw IllegalParamException("context", __FILE__, __LINE__);
    }

    context->setStreamParams(this, data, cntBytes, timeout, flags);
    this->threadPool.QueueUserWorkItem(AsyncSocket::sendFunc, context);
}


/*
 * vislib::net::AsyncSocket::BeginSend
 */
void vislib::net::AsyncSocket::BeginSend(const IPEndPoint& toAddr, 
        const void *data, const SIZE_T cntBytes, AsyncSocketContext *context,
        const INT timeout, const INT flags) {
    // TODO
    throw 1;
}


/*
 * vislib::net::AsyncSocket::EndReceive
 */
SIZE_T vislib::net::AsyncSocket::EndReceive(AsyncSocketContext *context) {
    VISLIB_STACKTRACE(EndReceive, __FILE__, __LINE__);

    /* Sanity checks. */
    if (context == NULL) {
        throw IllegalParamException("context", __FILE__, __LINE__);
    }

    // TODO: Check errors

    return context->cntData;
}


/*
 * vislib::net::AsyncSocket::EndSend
 */
SIZE_T vislib::net::AsyncSocket::EndSend(AsyncSocketContext *context) {
    VISLIB_STACKTRACE(EndSend, __FILE__, __LINE__);

    /* Sanity checks. */
    if (context == NULL) {
        throw IllegalParamException("context", __FILE__, __LINE__);
    }

    // TODO: Check errors

    return context->cntData;
}


#ifndef _WIN32
/*
 * vislib::net::AsyncSocket::Receive
 */
SIZE_T vislib::net::AsyncSocket::Receive(void *outData, const SIZE_T cntBytes,
        const INT timeout, const INT flags, const bool forceReceive) {
    VISLIB_STACKTRACE(Receive, __FILE__, __LINE__);
    SIZE_T retval = 0;

    this->lockRecv.Lock();
    try {
        retval = this->Receive(outData, cntBytes, timeout, flags, forceReceive);
    } catch (SocketException e) {
        this->lockRecv.Unlock();
        throw e;
    }

    this->lockRecv.Unlock();
    return retval;
}


/*
 * vislib::net::AsyncSocket::Receive
 */
SIZE_T vislib::net::AsyncSocket::Receive(IPEndPoint& outFromAddr, void *outData,
        const SIZE_T cntBytes, const INT timeout, const INT flags, 
        const bool forceReceive) {
    VISLIB_STACKTRACE(Receive, __FILE__, __LINE__);
    SIZE_T retval = 0;

    this->lockRecv.Lock();
    try {
        retval = this->Receive(outFromAddr, outData, cntBytes, timeout, flags,
            forceReceive);
    } catch (SocketException e) {
        this->lockRecv.Unlock();
        throw e;
    }

    this->lockRecv.Unlock();
    return retval;
}


/*
 * vislib::net::AsyncSocket::Receive
 */
SIZE_T vislib::net::AsyncSocket::Receive(SocketAddress& outFromAddr, 
        void *outData, const SIZE_T cntBytes, const INT timeout, 
        const INT flags, const bool forceReceive) {
    VISLIB_STACKTRACE(Receive, __FILE__, __LINE__);
    SIZE_T retval = 0;

    this->lockRecv.Lock();
    try {
        retval = this->Receive(outFromAddr, outData, cntBytes, timeout, flags, 
            forceReceive);
    } catch (SocketException e) {
        this->lockRecv.Unlock();
        throw e;
    }

    this->lockRecv.Unlock();
    return retval;
}


/*
 * vislib::net::AsyncSocket::Send
 */
SIZE_T vislib::net::AsyncSocket::Send(const void *data, const SIZE_T cntBytes, 
        const INT timeout, const INT flags, const bool forceSend) {
    VISLIB_STACKTRACE(Send, __FILE__, __LINE__);
    SIZE_T retval = 0;

    this->lockSend.Lock();
    try {
        retval = this->Send(data, cntBytes, timeout, flags, forceSend);
    } catch (SocketException e) {
        this->lockSend.Unlock();
        throw e;
    }

    this->lockSend.Unlock();
    return retval;
}


/*
 * vislib::net::AsyncSocket::Send
 */
SIZE_T vislib::net::AsyncSocket::Send(const IPEndPoint& toAddr, const void *data, 
        const SIZE_T cntBytes, const INT timeout, const INT flags, 
        const bool forceSend) {
    VISLIB_STACKTRACE(Send, __FILE__, __LINE__);
    SIZE_T retval = 0;

    this->lockSend.Lock();
    try {
        retval = this->Send(toAddr, data, cntBytes, timeout, flags, forceSend);
    } catch (SocketException e) {
        this->lockSend.Unlock();
        throw e;
    }

    this->lockSend.Unlock();
    return retval;
}
#endif /* !_WIN32 */


#ifdef _WIN32
/*
 * vislib::net::AsyncSocket::completedFunc
 */
void CALLBACK vislib::net::AsyncSocket::completedFunc(DWORD dwError, 
        DWORD cbTransferred, LPWSAOVERLAPPED lpOverlapped, DWORD dwFlags) {
    AsyncSocketContext *ctx = reinterpret_cast<AsyncSocketContext *>(
        lpOverlapped->hEvent);
    ctx->notifyCompleted(cbTransferred, dwError);
}
#endif /* _WIN32 */


//#ifndef _WIN32
/*
 * vislib::net::AsyncSocket::receiveFunc
 */
DWORD vislib::net::AsyncSocket::receiveFunc(void *asyncSocketContext) {
    AsyncSocketContext *ctx = static_cast<AsyncSocketContext *>(
        asyncSocketContext);
    DWORD retval = 0;

    try {
        Socket::Startup();
    } catch (SocketException e) {
        return e.GetErrorCode();
    }

    if (ctx->dgramAddr != NULL) {
        /* Receive on datagram socket. */
        ctx->socket->lockRecv.Lock();
        try {
            ctx->cntData = ctx->socket->Receive(*ctx->dgramAddr, ctx->data, 
                ctx->cntData, ctx->timeout, ctx->flags, false);
        } catch (SocketException e) {
            retval = e.GetErrorCode();
        }
        ctx->socket->lockRecv.Unlock();

    } else {
        /* Receive on stream socket. */
        ctx->socket->lockRecv.Lock();
        try {
            ctx->cntData = ctx->socket->Receive(ctx->data, ctx->cntData, 
                ctx->timeout, ctx->flags, false);
        } catch (SocketException e) {
            retval = e.GetErrorCode();
        }
        ctx->socket->lockRecv.Unlock();
    }

    try {
        Socket::Cleanup();
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_WARN, "SocketCleanup failed in AsyncSocket "
            "thread function.");
    }

    return retval;
}


/*
 * vislib::net::AsyncSocket::sendFunc
 */
DWORD vislib::net::AsyncSocket::sendFunc(void *asyncSocketContext) {
    AsyncSocketContext *ctx = static_cast<AsyncSocketContext *>(
        asyncSocketContext);
    DWORD retval = 0;

    try {
        Socket::Startup();
    } catch (SocketException e) {
        return e.GetErrorCode();
    }

    if (ctx->dgramAddr != NULL) {
        /* Send on datagram socket. */
        ctx->socket->lockSend.Lock();
        try {
            ctx->cntData = ctx->socket->Send(*ctx->dgramAddr, ctx->data, 
                ctx->cntData, ctx->timeout, ctx->flags, false);
        } catch (SocketException e) {
            retval = e.GetErrorCode();
        }
        ctx->socket->lockSend.Unlock();

    } else {
        /* Send on stream socket. */
        ctx->socket->lockSend.Lock();
        try {
            ctx->cntData = ctx->socket->Send(ctx->data, ctx->cntData, 
                ctx->timeout, ctx->flags, false);
        } catch (SocketException e) {
            retval = e.GetErrorCode();
        }
        ctx->socket->lockRecv.Unlock();
    }

    try {
        Socket::Cleanup();
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_WARN, "SocketCleanup failed in AsyncSocket "
            "thread function.");
    }

    return retval;
}
//#endif /* !_WIN32 */
