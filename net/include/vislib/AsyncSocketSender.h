/*
 * AsyncSocketSender.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ASYNCSOCKETSENDER_H_INCLUDED
#define VISLIB_ASYNCSOCKETSENDER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Socket.h"      // Must be first.
#include "vislib/CriticalSection.h"
#include "vislib/Event.h"
#include "vislib/RawStoragePool.h"
#include "vislib/Runnable.h"
#include "vislib/Semaphore.h"
#include "vislib/SingleLinkedList.h"


namespace vislib {
namespace net {


    /**
     * This is a Runnable that manages a queue for sending data asynchronously
     * using a Socket.
     */
    class AsyncSocketSender : public vislib::sys::Runnable {

    public:

        /**
         * The callback function type that can be passed to the Send() methods
         * in order to be notified about the completion of the asynchronous
         * operation.
         *
         * @param result       Indicate the success of the operation. If
         *                     successful, this is 0, otherwise a 
         *                     system-dependent error code that can be converted
         *                     to a vislib::sys::SystemMessage.
         * @param data         The data pointer that was passed to Send(). If
         *                     Send() created a local copy for the queue, this
         *                     parameter will be NULL.
         * @param cntBytesSent The number of bytes actually sent.
         * @param userContext  A user-defined pointer that was passed to Send().
         */
        typedef void (* CompletedFunction)(const DWORD result, const void *data,
            const SIZE_T cntBytesSent, void *userContext);

        /** Ctor. */
        AsyncSocketSender(void);

        /** Dtor. */
        virtual ~AsyncSocketSender(void);

        /**
         * Start sending data on the specified socket.
         *
         * @param socket The socket used for sending data. The caller is 
         *               responsible for ensuring that the socket exists as 
         *               long as the thread is running.
         *
         * @return 0, or a socket error code if the thread could not start.
         */
        virtual DWORD Run(void *socket);

        /**
         * Start sending data on the specified socket.
         *
         * @param socket The socket used for sending data. The caller is 
         *               responsible for ensuring that the socket exists as 
         *               long as the thread is running.
         *
         * @return 0, or a socket error code if the thread could not start.
         */
        DWORD Run(Socket *socket);

        /**
         * TODO
         */
        void Send(const void *data, const SIZE_T cntBytes,
            const CompletedFunction onCompleted, void *userContext,
            const bool doNotCopy = false,
            const INT timeout = Socket::TIMEOUT_INFINITE,
            const INT flags = 0, const bool forceSend = true);

        /**
         * TODO
         */
        inline void Send(const void *data, const SIZE_T cntBytes,
                sys::Event *evt, const bool doNotCopy,
                const INT timeout = Socket::TIMEOUT_INFINITE,
                const INT flags = 0, const bool forceSend = true) {
            this->Send(data, cntBytes, AsyncSocketSender::onSendCompleted,
                evt, doNotCopy, timeout, flags, forceSend);
        }

        /**
         * Send 'cntBytes' from the location designated by 'data' using this 
         * socket.
         *
         * @param data      The data to be sent. The caller remains owner of the
         *                  memory.
         * @param cntBytes  The number of bytes to be sent. 'data' must contain
         *                  at least this number of bytes.
         * @param flags     The flags that specify the way in which the call is 
         *                  made.
         *
         * @throws SocketException       If the operation fails.
         * @throws IllegalParamException If 'timeout' is not TIMEOUT_INFINITE 
         *                               and 'forceSend' is true.
         */
        inline void Send(const void *data, const SIZE_T cntBytes,
                const INT flags = 0) {
            this->Send(data, cntBytes, NULL, NULL, false, 
                Socket::TIMEOUT_INFINITE, flags, true);
        }

        /**
         * Terminate the thread by emptying the queue.
         *
         * @return true to acknowledge that the Runnable will finish as soon
         *         as possible, false if termination is not possible.
         */
        virtual bool Terminate(void);

    private:

        /** An element in the send queue. */
        typedef struct SendTask_t {
            const void *data0;
            RawStorage *data1;
            SIZE_T cntBytes;
            CompletedFunction onCompleted;
            void *userContext;
            INT timeout;
            INT flags;
            bool forceSend;

            inline bool operator ==(const struct SendTask_t& rhs) const {
                return ((this->data0 == rhs.data0)
                    && (this->data1 == rhs.data1)
                    && (this->cntBytes == rhs.cntBytes)
                    && (this->onCompleted == rhs.onCompleted)
                    && (this->userContext == rhs.userContext)
                    && (this->timeout == rhs.timeout)
                    && (this->flags == rhs.flags)
                    && (this->forceSend == rhs.forceSend));
            }
        } SendTask;

#ifdef _WIN32
        /**
         * Completion routine for asynchrous WSA operations.
         *
         * @param dwError
         * @param cbTransferred
         * @param lpOverlapped
         * @param dwFlags
         */
        void CALLBACK completionRoutine(DWORD dwError, DWORD cbTransferred,
            LPWSAOVERLAPPED lpOverlapped, DWORD dwFlags);
#endif /* _WIN32 */

        /**
         * Internal completed callback that is used to set an Event passed
         * as 'userContext'.
         *
         * @param result
         * @param data
         * @param cntBytesSent
         * @param userContext
         */
        static void onSendCompleted(const DWORD result, const void *data,
            const SIZE_T cntBytesSent, void *userContext);

        /** Critical section for protecting 'queue'. */
        sys::CriticalSection lockQueue;

        /** Critical section for protecting 'socket'. */
        sys::CriticalSection lockSocket;

        /** Critical section for protecting 'storagePool'. */
        sys::CriticalSection lockStoragePool;

        /** The queue of uncompleted send tasks. */
        SingleLinkedList<SendTask> queue;

        /** Blocks the sender if the queue is empty. */
        sys::Semaphore semBlockSender;

        /** The socket used for sending the data. */
        Socket *socket;

        /** Pool of raw storage objects for buffering data to send. */
        RawStoragePool *storagePool;

    };

} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ASYNCSOCKETSENDER_H_INCLUDED */
