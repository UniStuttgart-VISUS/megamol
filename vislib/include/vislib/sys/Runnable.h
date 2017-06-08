/*
 * Runnable.h  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_RUNNABLE_H_INCLUDED
#define VISLIB_RUNNABLE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/types.h"


namespace vislib {
namespace sys {

    /**
     * Defines the interface for objects that can be run in threads.
     */
    class Runnable {

    public:

        /** 
         * Functions with this signature can also be run as threads instead of
         * Runnable derive objects.
         *
         * @param userData A pointer to data that are passed to the Start() 
         *                 method of the Thread.
         *
         * @return The application dependent return code of the thread. This 
         *         must not be STILL_ACTIVE (259).
         */
        typedef DWORD (* Function)(void *userData); 

        /** Dtor. */
        virtual ~Runnable(void);

        /**
         * The thread will call this method immediately after it was started
         * in the context of the new thread and before Run() is called. The
         * 'userData' parameter is the same as will be passed to Run().
         *
         * @param userData The user data to be passed to Run().
         */
        virtual void OnThreadStarted(void *userData);

        /**
         * The thread will call this method immediately before it will be 
         * started. This is done in the context of the old thread. The
         * 'userData' parameter is the same as will be passed to Run().
         *
         * @param userData The user data to be passed to Run().
         */
        virtual void OnThreadStarting(void *userData);

        /**
         * Perform the work of a thread.
         *
         * @param userData A pointer to user data that are passed to the thread,
         *                 if it started.
         *
         * @return The application dependent return code of the thread. This 
         *         must not be STILL_ACTIVE (259).
         */
        virtual DWORD Run(void *userData) = 0;

        /**
         * The Runnable should abort its work as soon as possible. This method
         * should never block! If the Runnable will really stop its work at the
         * next possible possition, return true. Return false (the default 
         * implementation), if the Runnable is not able to interrupt its work.
         * Note, that the return value of this method is only a hint to prevent
         * deadlocks and that a thread might be forcefully terminated anyway.
         *
         * @return true to acknowledge that the Runnable will finish as soon
         *         as possible, false if termination is not possible.
         */
        virtual bool Terminate(void);

    };

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_RUNNABLE_H_INCLUDED */
