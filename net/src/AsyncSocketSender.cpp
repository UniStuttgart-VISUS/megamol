/*
 * AsyncSocketSender.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AsyncSocketSender.h"

#include <climits>

#include "vislib/error.h"
#include "the/exception.h"
#include "vislib/SocketException.h"
#include "vislib/Thread.h"
#include "the/trace.h"
#include "vislib/unreferenced.h"


/*
 * vislib::net::AsyncSocketSender::AsyncSocketSender
 */
vislib::net::AsyncSocketSender::AsyncSocketSender(void)
        : flags(FLAG_IS_CLOSE_SOCKET | FLAG_IS_LINGER), isQueueOpen(true), 
        semBlockSender(0l, LONG_MAX), storagePool(new RawStoragePool) {
    // Nothing else to do.
}


/*
 * vislib::net::AsyncSocketSender::~AsyncSocketSender
 */
vislib::net::AsyncSocketSender::~AsyncSocketSender(void) {
    try {
        this->Terminate();
        this->lockStoragePool.Lock();
        the::safe_delete(this->storagePool);
    } catch (the::exception& e) {
        VL_DBGONLY_REFERENCED_LOCAL_VARIABLE(e);
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Exception while destroying "
            "AsyncSocketSender: %s\n", e.what());
    }

    try {
        this->lockStoragePool.Unlock();
    } catch (...) {
    }
}


/*
 * vislib::net::AsyncSocketSender::Run
 */
unsigned int vislib::net::AsyncSocketSender::Run(void *socket) {
    return this->Run(static_cast<Socket *>(socket));
}


/*
 * vislib::net::AsyncSocketSender::Run
 */
unsigned int vislib::net::AsyncSocketSender::Run(Socket *socket) {
    THE_ASSERT(socket != NULL);
    const void *data = NULL;
    size_t cntSent = 0;
    unsigned int result = 0;

    try {
        Socket::Startup();
    } catch (SocketException e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Socket::Startup failed in "
            "AsyncSocketSender::Run().\n");
        return e.get_error().native_error();
    }

    while (true) {

        /* Acquire locks. */
        this->semBlockSender.Lock();
        this->lockQueue.Lock();

        /*
         * We use an empty queue as trigger for a thread to leave: If we wake a
         * thread and it does not find any work to do, it should exit.
         */
        if (!this->isQueueOpen && this->queue.empty()) {
            this->lockQueue.Unlock();
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "AsyncSocketSender [%u] is "
                "exiting because of empty queue ...\n",
                sys::Thread::CurrentID());
            break;
        }

        /* Set send task and send it using our socket. */
        THE_ASSERT(!this->queue.empty());
        SendTask task = this->queue.First();
        this->queue.RemoveFirst();
        this->lockQueue.Unlock();


        /* Send the data. */
        data = (task.data0 != NULL) ? task.data0
                                    : static_cast<const void *>(*task.data1);
        result = 0;
        try {
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Sending %u bytes "
                "starting at 0x%x ...\n", task.cntBytes, data);
            cntSent = socket->Send(data, task.cntBytes, task.timeout,
                task.flags, task.forceSend);
        } catch (SocketException e) {
            result = e.get_error().native_error();
            // TODO: Could leave the loop, if exception does not designate timeout.
        }
        this->finaliseSendTask(task, result, cntSent);
    } /* end while (true) */

    /* Do cleanup. */
    try {
        if ((this->flags & FLAG_IS_CLOSE_SOCKET) != 0) {
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "AsyncSocketSender [%u] is closing the "
                "socket ...\n", sys::Thread::CurrentID());
            socket->Close();
        }
    } catch (...) {
    }

    try {
        Socket::Cleanup();
    } catch (...) {
    }

    return 0;
}


/*
 * vislib::net::AsyncSocketSender::Send
 */
void vislib::net::AsyncSocketSender::Send(const void *data,
        const size_t cntBytes, const CompletedFunction onCompleted,
        void *userContext, const bool doNotCopy, const int timeout,
        const int flags, const bool forceSend) {
    SendTask task;

    /* Sanity checks. */
    if (data == NULL) {
        throw the::argument_exception("data", __FILE__, __LINE__);
    }
    if ((onCompleted == NULL) && doNotCopy) {
        throw the::argument_exception("onCompleted", __FILE__, __LINE__);
    }

    this->lockQueue.Lock();
    
    /* Check the queue state. */
    if (!this->isQueueOpen) {
        this->lockQueue.Unlock();
        throw the::invalid_operation_exception("The send queue has been closed. No more "
            "tasks may be added.", __FILE__, __LINE__);
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
    this->queue.Append(task);
    this->semBlockSender.Unlock();
    this->lockQueue.Unlock();
}


/*
 * vislib::net::AsyncSocketSender::SetIsLinger
 */
void vislib::net::AsyncSocketSender::SetIsLinger(const bool isLinger) {
    if (isLinger) {
        this->flags |= FLAG_IS_LINGER;
    } else {
        this->flags &= ~FLAG_IS_LINGER;
    }
}


/*
 * vislib::net::AsyncSocketSender::Terminate
 */
bool vislib::net::AsyncSocketSender::Terminate(void) {
    bool retval = true;

    try {
        this->lockQueue.Lock();
        this->isQueueOpen = false;

        if ((this->flags & FLAG_IS_LINGER) == 0) {
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Discarding all pending asynchronous "
                "socket sends ...\n");

            /* Notify the user about his/her task being aborted. */
            SingleLinkedList<SendTask>::Iterator it = this->queue.GetIterator();
            while (it.HasNext()) {
                SendTask& task = it.Next();
#ifdef _WIN32
                this->finaliseSendTask(task, ERROR_OPERATION_ABORTED, 0);
#else /* _WIN32 */
                this->finaliseSendTask(task, ECANCELED, 0);
#endif /* _WIN32 */
            }

            this->queue.Clear();
        }
        this->semBlockSender.Unlock();  // Ensure that the thread can run.

    } catch (the::exception& e) {
        VL_DBGONLY_REFERENCED_LOCAL_VARIABLE(e);
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Terminating AsyncSocketSender "
            "failed. %s\n", e.what());
        retval = false;
    }

    try {
        this->lockQueue.Unlock();
    } catch (...) {
    }

    return retval;
}


/*
 * vislib::net::AsyncSocketSender::onSendCompletedEvt
 */
void vislib::net::AsyncSocketSender::onSendCompletedEvt(const unsigned int result,
        const void *data, const size_t cntBytesSent, void *userContext) {
    if (userContext != NULL) {
        static_cast<sys::Event *>(userContext)->Set();
    }
}


/*
 * vislib::net::AsyncSocketSender::onSendCompletedSem
 */
void vislib::net::AsyncSocketSender::onSendCompletedSem(const unsigned int result,
        const void *data, const size_t cntBytesSent, void *userContext) {
    if (userContext != NULL) {
        static_cast<sys::Semaphore *>(userContext)->Unlock();
    }
}


/*
 * vislib::net::AsyncSocketSender::FLAG_IS_CLOSE_SOCKET
 */
const uint32_t vislib::net::AsyncSocketSender::FLAG_IS_CLOSE_SOCKET = 0x00000001;


/*
 * vislib::net::AsyncSocketSender::FLAG_IS_LINGER
 */
const uint32_t vislib::net::AsyncSocketSender::FLAG_IS_LINGER= 0x00000002;


/*
 * vislib::net::AsyncSocketSender::finaliseSendTask
 */
void vislib::net::AsyncSocketSender::finaliseSendTask(SendTask& task, 
        const unsigned int result, const size_t cntBytesSent) {
    /* Call completion function. */
    if (task.onCompleted != NULL) {
        task.onCompleted(result, task.data0, cntBytesSent, task.userContext);
    }

    /* Return RawStorage. */
    if (task.data1 != NULL) {
        this->lockStoragePool.Lock();
        this->storagePool->Return(task.data1);
        this->lockStoragePool.Unlock();
    }
}
