/*
 * ThreadPoolListener.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_THREADPOOLLISTENER_H_INCLUDED
#define VISLIB_THREADPOOLLISTENER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Runnable.h"


namespace vislib {
namespace sys {

    /* Forward declarations. */
    class ThreadPool;

    /**
     * Classes implementing this interface can listen for work items being 
     * completed or aborted by a ThreadPool. 
     *
     * This callback mechanism can also be used to transfer ownership of the 
     * Runnable object and its user data from the thread pool to the user, e.g.
     * if dynamically allocated objects are used. See the documentation of 
     * vislib::sys::ThreadPool for further information.
     */
    class ThreadPoolListener {

    public:

        /** Dtor. */
        virtual ~ThreadPoolListener(void);

        /**
         * The thread pool calls this method once a work item (Runnable) is 
         * removed from its queue without having been processed.
         *
         * Implementing methods should return as soon as possible.
         *
         * @param src      The thread pool that originated the event.
         * @param runnable The Runnable that has been aborted.
         * @param userData The user data pointer that would have been passed 
         *                 into the Runnable.
         */
        virtual void OnUserWorkItemAborted(ThreadPool& src, Runnable *runnable,
            void *userData) throw() = 0;

        /**
         * The thread pool calls this method once a work item 
         * (Runnable::Function) is removed from its queue without having been 
         * processed.
         *
         * Implementing methods should return as soon as possible.
         *
         * @param src      The thread pool that originated the event.
         * @param runnable The Runnable::Function that has been aborted.
         * @param userData The user data pointer that would have been passed 
         *                 into the Runnable::Function.
         */
        virtual void OnUserWorkItemAborted(ThreadPool& src, 
            Runnable::Function runnable, void *userData) throw() = 0;

        /**
         * The thread pool calls this method once a work item (Runnable) has been 
         * completed.
         *
         * This method runs in the thread context of the thread pool. The 
         * implementation should return a soon as possible.
         *
         * @param src      The thread pool that originated the event.
         * @param runnable The Runnable that has been completed.
         * @param userData The user data pointer that has been passed into the
         *                 Runnable.
         * @param exitCode The exit code that the Runnable returned.
         */
        virtual void OnUserWorkItemCompleted(ThreadPool& src, 
            Runnable *runnable, void *userData, 
            const DWORD exitCode) throw() = 0;


        /**
         * The thread pool calls this method once a work item 
         * (Runnable::Function) has been completed.
         *
         * This method runs in the thread context of the thread pool. The 
         * implementation should return a soon as possible.
         *
         * @param src      The thread pool that originated the event.
         * @param runnable The Runnable::Function that has been completed.
         * @param userData The user data pointer that has been passed into the
         *                 Runnable::Function.
         * @param exitCode The exit code that the Runnable::Function returned.
         */
        virtual void OnUserWorkItemCompleted(ThreadPool& src, 
            Runnable::Function runnable, void *userData, 
            const DWORD exitCode) throw() = 0;

    protected:

        /** Ctor. */
        ThreadPoolListener(void);

        /**
         * Clone rhs
         *
         * @param rhs The object to be cloned.
         */
        ThreadPoolListener(ThreadPoolListener& rhs);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        ThreadPoolListener& operator =(const ThreadPoolListener& rhs);
    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_THREADPOOLLISTENER_H_INCLUDED */
