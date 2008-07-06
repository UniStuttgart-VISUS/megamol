/*
 * AsyncSocketSender.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AsyncSocketSender.h"

#include <climits>

#include "vislib/Exception.h"
#include "vislib/SocketException.h"
#include "vislib/Thread.h"
#include "vislib/Trace.h"


/*
 * vislib::net::AsyncSocketSender::AsyncSocketSender
 */
vislib::net::AsyncSocketSender::AsyncSocketSender(void)
        : semBlockSender(0l, LONG_MAX), socket(NULL), 
        storagePool(new RawStoragePool) {
    // Nothing else to do.
}


/*
 * vislib::net::AsyncSocketSender::~AsyncSocketSender
 */
vislib::net::AsyncSocketSender::~AsyncSocketSender(void) {
    try {
        this->Terminate();
        this->lockStoragePool.Lock();
        SAFE_DELETE(this->storagePool);
    } catch (Exception& e) {
        TRACE(Trace::LEVEL_VL_WARN, "Exception while destroying "
            "AsyncSocketSender: %s\n", e.GetMsgA());
    }

    try {
        this->lockStoragePool.Unlock();
    } catch (...) {
    }
}


/*
 * vislib::net::AsyncSocketSender::Run
 */
DWORD vislib::net::AsyncSocketSender::Run(void *socket) {
    return this->Run(static_cast<Socket *>(socket));
}


/*
 * vislib::net::AsyncSocketSender::Run
 */
DWORD vislib::net::AsyncSocketSender::Run(Socket *socket) {
    const void *data = NULL;
    SIZE_T cntSent = 0;
    DWORD result = 0;

    try {
        Socket::Startup();
    } catch (SocketException e) {
        TRACE(Trace::LEVEL_VL_ERROR, "Socket::Startup failed in "
            "AsyncSocketSender::Run().\n");
        return e.GetErrorCode();;
    }

    this->lockSocket.Lock();
    this->socket = socket;
    this->lockSocket.Unlock();

    while (true) {

        /* Acquire locks. */
        this->semBlockSender.Lock();
        this->lockQueue.Lock();

        /*
         * We use an empty queue as trigger for a thread to leave: If we wake a
         * thread and it does not find any work to do, it should exit.
         */
        if (this->queue.IsEmpty()) {
            this->lockQueue.Unlock();
            TRACE(Trace::LEVEL_VL_INFO, "AsyncSocketSender [%u] is "
                "exiting because of empty queue ...\n",
                sys::Thread::CurrentID());
            break;
        }

        /* Set send task and send it using our socket. */
        ASSERT(!this->queue.IsEmpty());
        SendTask task = this->queue.First();
        this->queue.RemoveFirst();
        this->lockQueue.Unlock();

        this->lockSocket.Lock();

        /* The socket is also a termination trigger. */
        if (this->socket == NULL) {
            this->lockSocket.Unlock();
            break;
        }

        /* Send the data. */
        data = (task.data0 != NULL) ? task.data0
                                    : static_cast<const void *>(*task.data1);
        result = 0;
        try {
            cntSent = this->socket->Send(data, task.cntBytes, task.timeout,
                task.flags, task.forceSend);
        } catch (SocketException e) {
            result = e.GetErrorCode();
        }
        this->lockSocket.Unlock();

        /* Call completion function. */
        if (task.onCompleted != NULL) {
            task.onCompleted(result, task.data0, cntSent, task.userContext);
        }

        /* Reuse RawStorage. */
        if (task.data1 != NULL) {
            this->lockStoragePool.Lock();
            this->storagePool->Return(task.data1);
            this->lockStoragePool.Unlock();
        }
    }

    /* Do cleanup. */
    try {
        Socket::Cleanup();
    } catch (...) {
    }

    return 0;
}


/*
 * vislib::net::AsyncSocketSender::Terminate
 */
bool vislib::net::AsyncSocketSender::Terminate(void) {
    bool retval = true;

    try {
        this->lockSocket.Lock();
        this->lockQueue.Lock();

        this->socket = NULL;
        this->queue.Clear();
        this->semBlockSender.Unlock();  // Ensure that the thread can run.

    } catch (vislib::Exception& e) {
        TRACE(Trace::LEVEL_VL_ERROR, "Terminating AsyncSocketSender "
            "failed. %s\n", e.GetMsgA());
        retval = false;
    }

    try {
        this->lockSocket.Unlock();
        this->lockQueue.Unlock();
    } catch (...) {
    }

    return retval;
}


/*
 * vislib::net::AsyncSocketSender::Send
 */
void vislib::net::AsyncSocketSender::Send(const void *data,
        const SIZE_T cntBytes, const CompletedFunction onCompleted,
        void *userContext, const bool doNotCopy, const INT timeout,
        const INT flags, const bool forceSend) {
    SendTask task;

    /* Sanity checks. */
    if (data == NULL) {
        throw IllegalParamException("data", __FILE__, __LINE__);
    }
    if ((onCompleted == NULL) && doNotCopy) {
        throw IllegalParamException("onCompleted", __FILE__, __LINE__);
    }

    /* Prepare the task. */
    if (doNotCopy) {
        /* Use user pointer for sending. */
        task.data0 = data;
        task.data1 = NULL;

    } else {
        /* Create own copy for sending. */
        task.data0 = NULL;
        this->lockStoragePool.Lock();
        task.data1 = this->storagePool->RaiseAtLeast(cntBytes);
        this->lockStoragePool.Unlock();
        ::memcpy(static_cast<void *>(*task.data1), data, cntBytes);
    }

    task.cntBytes = cntBytes;
    task.onCompleted = onCompleted;
    task.userContext = userContext;
    task.timeout = timeout;
    task.flags = flags;
    task.forceSend = forceSend;

    /* Queue the task. */
    this->lockQueue.Lock();
    this->queue.Append(task);
    this->semBlockSender.Unlock();
    this->lockQueue.Unlock();
}


/*
 * vislib::net::AsyncSocketSender::onSendCompleted
 */
void vislib::net::AsyncSocketSender::onSendCompleted(const DWORD result,
        const void *data, const SIZE_T cntBytesSent, void *userContext) {
    if (userContext != NULL) {
        static_cast<sys::Event *>(userContext)->Set();
    }
}
