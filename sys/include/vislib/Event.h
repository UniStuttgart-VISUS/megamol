/*
 * Event.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_EVENT_H_INCLUDED
#define VISLIB_EVENT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#include "vislib/Semaphore.h"
#endif /* _WIN32 */


namespace vislib {
namespace sys {


    /**
     * This class implements a synchronisation object whose state can be 
     * explicitly set. There are two states, signaled and non-signaled. If
     * the event is in signaled state, threads calling the Wait function
     * can proceed, in non-signaled state the threads are blocked.
     *
     * The event can be used as auto-reset and manual-reset event. An
     * auto-reset event resets its state to non-signaled after one thread
     * has been released. A manual-reset event stays signaled until it is
     * explicitly reset.
     */
    class Event {

    public:

        /** 
         * Use this value to wait infinitely for an event to become 
         * signaled. 
         */
        static const DWORD TIMEOUT_INFINITE;

        /** 
         * Create a new event.
         *
         * The default ctor creates an auto-reset event, which is initially
         * non-signaled.
         *
         * @param isManualReset       Make the event a manual-reset event. If 
         *                            the flag is not set, an auto-reset event 
         *                            is created.
         * @param isInitiallySignaled Specifies whether the event is initially
         *                            signaled.
         */
        Event(const bool isManualReset = false, 
            const bool isInitiallySignaled = false);

        /**
         * Opens an existing event with the specified name 'name' or creates a 
         * new named event if no such event exists.
         *
         * Windows: If an existing event is opened, the 'isManualReset' and 
         * 'isInitiallySignaled' parameters are ignored as these are defined by
         * the original creator of the event.
         *
         * Linux: The user must ensure that the 'isManualReset' flag is set
         * correctly when opening an existing event to match the value of the
         * original creator. The behaviour of the event is undefined if this
         * condition is not met.
         *
         * The default ctor creates an auto-reset event, which is initially
         * non-signaled.
         *
         * @param name                The name of the event.
         * @param isManualReset       Make the event a manual-reset event. If 
         *                            the flag is not set, an auto-reset event 
         *                            is created.
         * @param isInitiallySignaled Specifies whether the event is initially
         *                            signaled.
         * @param outIsNew            If not NULL, the ctor returns whether the
         *                            semaphore was created (true) or opened 
         *                            (false).
         */
        Event(const char *name, const bool isManualReset = false,
            const bool isInitiallySignaled = false, bool *outIsNew = NULL);

        /**
         * Opens an existing event with the specified name 'name' or creates a 
         * new named event if no such event exists.
         *
         * Windows: If an existing event is opened, the 'isManualReset' and 
         * 'isInitiallySignaled' parameters are ignored as these are defined by
         * the original creator of the event.
         *
         * Linux: The user must ensure that the 'isManualReset' flag is set
         * correctly when opening an existing event to match the value of the
         * original creator. The behaviour of the event is undefined if this
         * condition is not met.
         *
         * The default ctor creates an auto-reset event, which is initially
         * non-signaled.
         *
         * @param name                The name of the event.
         * @param isManualReset       Make the event a manual-reset event. If 
         *                            the flag is not set, an auto-reset event 
         *                            is created.
         * @param isInitiallySignaled Specifies whether the event is initially
         *                            signaled.
         * @param outIsNew            If not NULL, the ctor returns whether the
         *                            semaphore was created (true) or opened 
         *                            (false).
         */
        Event(const wchar_t *name, const bool isManualReset = false,
            const bool isInitiallySignaled = false, bool *outIsNew = NULL);

        /** Dtor. */
        ~Event(void);

        /**
         * Resets the event to non-signaled state.
         * 
         * It is safe to reset an already non-signaled event. The operation has 
         * no effect in this case.
         *
         * @throws SystemException If the operation failed.
         */
        void Reset(void);

        /**
         * Sets the event to signaled state.
         *
         * It is safe to set an already signaled event. The operation has no
         * effect in this case.
         *
         * @throws SystemException If the operation failed.
         */
        void Set(void);

        /**
         * Wait for the event to become signaled.
         *
         * @param timeout The timeout for the wait operation in milliseconds.
         *
         * @return true, if the event was signaled, false, if the operation
         *         timed out.
         * 
         * @throws SystemException If the operation failed.
         */
        bool Wait(const DWORD timeout = TIMEOUT_INFINITE);

    protected:

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        Event(const Event& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        Event& operator =(const Event& rhs);

    private:

#ifdef _WIN32
        /** Handle to the event object. */
        HANDLE handle;

#else /* _WIN32 */
        /** Remember whether the event is a manual reset event. */
        bool isManualReset;

        /** The semaphore used to emulate the event. */
        Semaphore semaphore;

#endif /* _WIN32 */

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_EVENT_H_INCLUDED */

