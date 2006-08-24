/*
 * Thread.h  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */
#ifndef VISLIB_THREAD_H_INCLUDED
#define VISLIB_THREAD_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

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
     * @author Christoph Mueller
     */
    class Thread {

    public:

        /**
		 * The possible states of a thread:
		 *
		 * NEW:       The thread object has been created, but the thread has
		 *            not been started.
		 * RUNNING:   The thread is currently running.
		 * SUSPENDED: The thread was started, but is currently suspended.
		 * FINISHED:  The thread was started and has finished its work.
		 */
        enum State { NEW = 1, RUNNING, SUSPENDED, FINISHED };

        /** Functions with this signature can be run as threads. */
        typedef DWORD (* RunnableFunc)(const void *userData); 

		/**
		 * Makes the calling thread sleep for 'millis' milliseconds.
		 *
		 * @param millis The milliseconds to block the calling thread.
		 */
		static void Sleep(const DWORD millis);

		/** 
         * Create a thread that executes the given Runnable.
         *
         * @param runnable The Runnable to run in a thread.
         */
		Thread(Runnable& runnable);

        /**
         * Create a thread that executes the function designated by 
         * 'threadFunc'.
         *
         * @param threadFunc The function to run in a thread.
         */
        Thread(RunnableFunc threadFunc);

		/** Dtor. */
		~Thread(void);

		/**
		 * Answer the exit code of the thread. Note, that this value is only
		 * meaningful, if the thread is in the state FINISHED.
		 *
		 * @return The thread exit code.
		 */
        inline DWORD GetExitCode(void) const {
            return this->exitCode;
        }

		/**
		 * Answer the current state of the thread.
		 *
		 * @return The current state of the thread.
		 */
        State GetState(void) const {
            return this->state;
        }

		/**
		 * Answer whether the thread is currently running.
		 *
		 * @return true, if the thread is currently running, false otherwise.
		 */
		inline bool IsRunning(void) const {
			return (this->GetState() == RUNNING);
		}

        /**
         * Start the thread.
         *
         * A thread can only be started, if it is in the state NEW or
         * FINISHED.
         *
         * @param userData The user data that are passed to the new thread's
         *                 thread function or Run() method.
         *
         * @return true, if the thread was successfully started, false 
         *         otherwise.
         */
		bool Start(const void *userData = NULL);

        /**
         * Terminate the thread.
         *
         * If the thread is currently not running, this method has no effect and
         * returns true.
		 *
		 * Note, that using this method is inherently unsafe as a thread might have
		 * allocated resources that it cannot free savely, if it is being terminated.
		 * You might wish to override this method to allow for a save shutdown, if
		 * your thread allocates resources that must be freed.
         *
         * @param exitCode The exit code of the thread.
         *
         * @return true, if the thread was terminated successfully, false 
         *         otherwise.
         */
        virtual bool Terminate(const int exitCode);

        /**
         * Waits for the thread to finish. If the thread was not started, the
         * method just falls through returning true.
         *
         * The method returns false, iff the thread was started and the wait
         * operation failed.
         *
         * @return true, if the thread is not running any more, false, if the
         *         wait operation failed.
         */
        bool Wait(void);

        /**
         * Execute the thread's run() method synchronously in the calling 
         * thread.
         *
         * The thread can only be executed synchronously, if it is currently not
         * running asynchronously, i. e. if it is in the state NEW or FINISHED.
         *
         * @return true, if the thread was run, false, if it was in an illegal
         *         state.
         */
		bool operator ()(void);

	protected:

		/**
		 * This is the worker method of the thread. The default implementation
		 * does nothing. Subclasses should overwrite this method with their own
		 * job.
		 *
		 * @return The return code of the thread.
		 */
		virtual int run(void);

	private:

		/**
		 * The thread function that is passed to the system API when starting a
		 * new thread.
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

#ifndef _WIN32
        /** The thread attributes. */
        pthread_attr_t attribs;
#endif /* !_WIN32 */

		/** The exit code of the thread. */
		DWORD exitCode;

#ifdef _WIN32
        /** Handle of the thread. */
        HANDLE handle;

        /** The thread ID. */
        DWORD id;

#else /* _WIN32 */
        /** The thread ID. */
        pthread_t id;

#endif /* _WIN32 */

		/** The state of the thread. */
		State state;

	};

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_THREAD_H_INCLUDED */
