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
     *
     * There are several flags configuring the behaviour of the sender:
     *
     * - If the isLinger property is set true (which is the default), the thread
     *   will perform a graceful shutdown on Terminate(), i.e. it will try to 
     *   process all pending requests. Otherwise, all pending requests are 
     *   discarded.
     * - If the isCloseSocket property is set true (which is the default), the
     *   thread will close the socket that it used for communication when it 
     *   exists. Otherwise, the socket will remain untouched and can possibly
     *   be reused for other tasks as long as it is still open.
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
         * Answer whether the thread should close its socket when exiting.
         *
         * @param true If the thread should close the socket, false otherwise.
         */
        inline bool IsCloseSocket(void) const {
            return ((this->flags & FLAG_IS_CLOSE_SOCKET) != 0);
        }

        /**
         * Answer whether terminating the thread causes all pending messages to
         * be sent or to be discarded.
         *
         * @return true If all pending messages are sent, false if they are
         *         discarded.
         */
        inline bool IsLinger(void) const {
            return ((this->flags & FLAG_IS_LINGER) != 0);
        }

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
            this->Send(data, cntBytes, AsyncSocketSender::onSendCompletedEvt,
                evt, doNotCopy, timeout, flags, forceSend);
        }

        /**
         * TODO
         */
        inline void Send(const void *data, const SIZE_T cntBytes,
                sys::Semaphore *semaphore, const bool doNotCopy,
                const INT timeout = Socket::TIMEOUT_INFINITE,
                const INT flags = 0, const bool forceSend = true) {
            this->Send(data, cntBytes, AsyncSocketSender::onSendCompletedSem,
                semaphore, doNotCopy, timeout, flags, forceSend);    
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
         * Change the behaviour of the thread when exiting: When setting 
         * 'isCloseSocket' true, the thread will close the socket that is used
         * for communciation once it exits. Otherwise, the socket will remain 
         * untouched and you can possibly resue it.
         *
         * @param isCloseSocket The new value of the flag.
         */
        void SetIsCloseSocket(const bool isCloseSocket);

        /**
         * Change the termination behaviour of the thread: If 'isLinger' is true,
         * the queue is emptied by sending all the data when terminating the 
         * thread. Otherwise, the queue is immediately emptied and all queued 
         * send operations are discarded.
         * 
         * @param isLinger The new value of the flag.
         */
        void SetIsLinger(const bool isLinger);

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
        static void onSendCompletedEvt(const DWORD result, const void *data,
            const SIZE_T cntBytesSent, void *userContext);

        /**
         * Internal completed callback that is used to unlock a Semaphore 
         * passed as 'userContext'.
         *
         * @param result
         * @param data
         * @param cntBytesSent
         * @param userContext
         */
        static void onSendCompletedSem(const DWORD result, const void *data,
            const SIZE_T cntBytesSent, void *userContext);

        /**
         * Call the completion callback of 'task', if there is one, and return
         * any RawStorage to the pool.
         *
         * @param result       The return code passed to the completion callback.
         * @param cntBytesSent The number of bytes sent passed to the completion
         *                     callback.
         */
        void finaliseSendTask(SendTask& task, const DWORD result, 
            const SIZE_T cntBytesSent);

        /**
         * This flag determines whether the thread should close the socket when
         * it exits.
         */
        static const UINT32 FLAG_IS_CLOSE_SOCKET;

        /**
         * This flag determines the termination behaviour of the thread: If set
         * true, the queue is closed and the thread runs as long as there are
         * still tasks in the queue. If false, the queue is emptied immediately
         * on calling terminate.
         */
        static const UINT32 FLAG_IS_LINGER;

        /** A bitmask of behaviour flags. */
        UINT32 flags;

        /** 
         * This flag determines whether it is possible to add new tasks in
         * the 'queue'. When terminating the thread, 'isQueueOpen' is set
         * false for making the queue run empty.
         * When accessing this attribute, 'lockQueue' must be held.
         */
        bool isQueueOpen;

        /** Critical section for protecting 'queue'. */
        sys::CriticalSection lockQueue;

        /** Critical section for protecting 'storagePool'. */
        sys::CriticalSection lockStoragePool;

        /** The queue of uncompleted send tasks. */
        SingleLinkedList<SendTask> queue;

        /** Blocks the sender if the queue is empty. */
        sys::Semaphore semBlockSender;

        /** Pool of raw storage objects for buffering data to send. */
        RawStoragePool *storagePool;

    };

} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ASYNCSOCKETSENDER_H_INCLUDED */
