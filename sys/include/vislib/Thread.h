/*
 * Thread.h  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */
#ifndef VISLIB_THREAD_H_INCLUDED
#define VISLIB_THREAD_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifndef _WIN32
#include <pthread.h>
#endif /* !_WIN32 */


#include "vislib/Runnable.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {

    /**
     * Implements a platform independent interface for system threads.
     *
     * On Linux, Threads are created joinable. Join() may only be called once. 
     * See remarks on this method.
     *
     * On Linux, the thread is detached in the destructor. Any native API calls,
     * in particular joining the thread, using the thread ID are illegal after 
     * the thread object has been destroyed.
     *
     * @author Christoph Mueller
     */
    class Thread {
        // TODO: Exit code handling/IsRunning impl is at least hugly.

    public:

        /**
         * Answer the ID of the calling thread.
         *
         * @eturn The ID of the calling thread.
         */
        static inline DWORD CurrentID(void) {
#ifdef _WIN32
            return ::GetCurrentThreadId();
#else /* _WIN32 */
            return static_cast<DWORD>(::pthread_self());
#endif /* _WIN32 */
        }

        // TODO: How to get exit code under Linux when using pthread_exit??
//        /**
//         * Exits the calling thread with the specified exit code.
//         *
//         * @param exitCode Exit code for the calling thread.
//         */
//        static inline void Exit(const DWORD exitCode) {
//#ifdef _WIN32
//            ::ExitThread(exitCode);
//#else /* _WIN32 */
//            ::pthread_exit(reinterpret_cast<void *>(exitCode));
//#endif /* _WIN32 */
//        }

        /**
         * Makes the calling thread sleep for 'millis' milliseconds.
         *
         * @param millis The milliseconds to block the calling thread.
         */
        static void Sleep(const DWORD millis);

        /**
         * Causes the calling thread to yield execution to another thread that 
         * is ready to run on the current processor.
         *
         * @throws SystemException If the operation could not be completed 
         *                         successfully (Linux only).
         */
        static void Reschedule(void);
        // Implementation note: Cannot be named Yield() because of macro with 
        // the same name in Windows API.

        /** 
         * Create a thread that executes the given Runnable.
         *
         * @param runnable The Runnable to run in a thread. The caller must
         *                 guarantee that the object designated by 'runnable'
         *                 exists at least as long as the thread is running.
         */
        explicit Thread(Runnable *runnable);

        /**
         * Create a thread that executes the function designated by 
         * 'runnableFunc'.
         *
         * @param runnableFunc The function to run in a thread.
         */
        explicit Thread(Runnable::Function runnableFunc);

        /** Dtor. */
        virtual ~Thread(void);

        /**
         * Answer the exit code of the thread. 
         *
         * @return The thread exit code.
         * 
         * @throws SystemException If the exit code could not be determined.
         */
        DWORD GetExitCode(void) const;

        /**
         * Answer a pointer to the Runnable executed by this thread. If the
         * thread executes a ThreadFunc instead of a runnable, this method
         * will return NULL.
         *
         * @return A pointer to the Runnable or NULL, if a function is used.
         */
        inline const Runnable *GetRunnable(void) const {
            return this->runnable;
        }

        /**
         * Answer whether the thread is currently running.
         *
         * @return true, if the thread is currently running, false otherwise.
         */
        bool IsRunning(void) const;

        /**
         * Waits for the thread to finish. If the thread was not started, the
         * method returns immediately.
         *
         * On Linux, Join() can only be called by one other thread.
         *
         * On Linux, Join() detaches the thread. Any further call to join is
         * illegal.
         *
         * @throws SystemException If waiting for the thread failed.
         */
        void Join(void);

        /**
         * Start the thread.
         *
         * A thread can only be started, if it is in the state NEW or
         * FINISHED.
         *
         * @param userData The user data that are passed to the new thread's
         *                 thread function or Run() method.
         *
         * @return true, if the thread was started, false, if it could not be
         *         started, because it is already running.
         *
         * @throws SystemException If the creation of the new thread failed.
         */
        bool Start(void *userData = NULL);

        /**
         * Terminate the thread. 
         *
         * If the thread has been constructed using a RunnableFunc, the 
         * behaviour is as follows: If 'forceTerminate' is true, the thread is
         * forcefully terminated and the method returns true. 'forceTerminate'
         * cannot be false when using a RunnableFunc. The method will throw
         * an IllegalParamException.
         *
         * If the thread has been constructed using a Runnable object, the
         * behaviour is as follows: If 'forceTerminate' is true, the thread
         * is forcefully terminated and the method returns true. Otherwise,
         * the method behaves as TryTerminate(true). Note, that this can cause
         * a deadlock, if your Runnable acknowledges a termination request but
         * does not finish.
         *
         * @param forceTerminate If true, the thread is terminated immediately,
         *                       if false, the thread has the possibility to do
         *                       some cleanup and finish in a controllend 
         *                       manner. 'forceTerminate' must be true, if the
         *                       thread has been constructed using a 
         *                       RunnableFunc.
         * @param exitCode       If 'forceTerminate' is true, this value will be
         *                       used as exit code of the thread. If 
         *                       'forceTerminate' is false, this value will be
         *                       ignored.
         * 
         * @returns true, if the thread has been terminated, false, otherwise.
         *
         * @throws IllegalStateException If 'forceTerminate' is false and the
         *                               thread has been constructed using a 
         *                               RunnableFunc.
         * @throws SystemException       If terminating the thread forcefully
         *                               failed.
         */
        bool Terminate(const bool forceTerminate, const int exitCode = 0);

        /**
         * This method tries to terminate a thread in a controlled manner. It
         * can only be called, if the thread has been constructed using a
         * Runnable object.
         *
         * The behaviour is as follows: The method will ask the Runnable to 
         * finish as soon as possible. If the Runnable acknowledges the request,
         * the method can wait for the thread to finish. It returns true in
         * both cases. If the Runnable does not acknowledge the request, the 
         * method returns false immediately.
         *
         * Note, that TryTerminate(true) can possibly cause a deadlock, if the 
         * Runnable acknowledges the request and does not return.
         *
         * @param doWait If set true, the method will wait for the thread to
         *               finish, if the Runnable acknowlegdes the termination
         *               request. Otherwise, the method will return immediately
         *               after requesting the termination.
         *
         * @return true, if the Runnable acknowledged the termination request,
         *         false otherwise.
         *
         * @throws IllegalStateException If the thread has been constructed 
         *                               using a RunnableFunc.
         */
        bool TryTerminate(const bool doWait = false);

    private:

        /**
         * A pointer to this structure is passed to ThreadFunc. This consists of
         * a pointer to the Thread object and of the user data for the thread.
         */
        typedef struct ThreadFuncParam_t {
            Thread *thread;             // The thread to execute.
            void *userData;             // The user parameters.
        } ThreadFuncParam;


#ifndef _WIN32
        /**
         * A cleanup handler for posix threads. We use this to determine whether a
         * thread is still running and for storing the exit code.
         *
         * @param param A pointer to this Thread.
         */
        static void CleanupFunc(void *param);
#endif /* !_WIN32 */

        /**
         * The thread function that is passed to the system API when starting a
         * new thread.
         *
         * @param param The parameters for the thread which must be a pointer 
         *              to a ThreadFuncParam structure.
         * 
         * @return The thread's exit code.
         */
#ifdef _WIN32
        static DWORD WINAPI ThreadFunc(void *param);
#else /* _WIN32 */
        static void *ThreadFunc(void *param);
#endif /* _WIN32 */

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        Thread(const Thread& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        Thread& operator =(const Thread& rhs);

#ifdef _WIN32 
        /** Handle of the thread. */
        HANDLE handle;

        /** The thread ID. */
        DWORD id;

#else /* _WIN32 */
        /** The thread attributes. */
        pthread_attr_t attribs;

        /** The exit code of the thread. */
        DWORD exitCode;

        /** The thread ID. */
        pthread_t id;

#endif /* _WIN32 */

        /** The Runnable to execute, or NULL, if 'threadFunc' should be used. */
        Runnable *runnable;

        /** The function to execute, or NULL, if 'runnable' should be used. */
        Runnable::Function runnableFunc;

        /** This is the parameter for the actual system thread function. */
        ThreadFuncParam threadFuncParam;
    };

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_THREAD_H_INCLUDED */
