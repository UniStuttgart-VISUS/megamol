/*
 * ThreadSafeStackTrace.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_THREADSAFESTACKTRACE_H_INCLUDED
#define VISLIB_THREADSAFESTACKTRACE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/StackTrace.h"
#include "vislib/Thread.h"
#include "vislib/CriticalSection.h"


namespace vislib {
namespace sys {


    /**
     * The thread-safe manager class for stack trace.
     *
     * Each modul of your application must initialise the stack trace manager
     * object by calling 'vislib::StackTrace::Initialise', however, you must
     * ensure that only a single manager object is created and is used by all
     * of your modules to get correct results.
     *
     * If your application is multi-threaded you must use
     * 'vislib::sys::ThreadSafeStackTrace::Initialise'. Otherwise the stack
     * returned may be wrong or corrupted.
     *
     * You should use the 'VISLIB_STACKTRACE' macro to push functions onto
     * the tracing stack.
     */
    class ThreadSafeStackTrace : public StackTrace {
    public:

        /**
         * Gets the whole current stack.
         *
         * If 'outStr' is NULL 'strSize' receives the required size in
         * characters to store the whole stack (including the terminating
         * zero).
         * If 'outStr' is not NULL 'strSize' specifies the size of the
         * buffer 'outStr' is pointing to. The method will only write that
         * amount of characters to the buffer. The content of 'strSize' will
         * be changed to the number of characters actually written (including
         * the terminating zero).
         *
         * @param outStr The buffer to recieve the stack.
         * @param strSize The size of the buffer.
         */
        static void GetStackString(char *outStr, unsigned int &strSize);

        /**
         * Gets the whole current stack.
         *
         * If 'outStr' is NULL 'strSize' receives the required size in
         * characters to store the whole stack (including the terminating
         * zero).
         * If 'outStr' is not NULL 'strSize' specifies the size of the
         * buffer 'outStr' is pointing to. The method will only write that
         * amount of characters to the buffer. The content of 'strSize' will
         * be changed to the number of characters actually written (including
         * the terminating zero).
         *
         * @param outStr The buffer to recieve the stack.
         * @param strSize The size of the buffer.
         */
        static void GetStackString(wchar_t *outStr, unsigned int &strSize);

        /**
         * Initialises the stack trace. You can specify the manager object for
         * the stack trace. If you do not specify the manager object, a new
         * instance will be created.
         *
         * It is usful to specify the manager object if your application
         * consists of multiple modules each linking against an own instance
         * of the vislib. In that case each modul must initialise the stack
         * trace by itself, but you should only use a single manager object.
         *
         * @param manager The manager object to be used for stack trace.
         * @param force If 'Initialise' has already been called for this
         *              module this flag controls whether the previous
         *              manager object is keept ('false') or if the new one
         *              (parameter 'manager') is used ('true') and the old
         *              one is removed.
         *
         * @return 'true' if the initialisation was successful and the object
         *         specified by 'manager' is now used to manage the stack
         *         trace; 'false' otherwise.
         */
        static bool Initialise(SmartPtr<StackTrace> manager = NULL,
            bool force = false);

        /**
         * Answer the manager object used by this modul.
         *
         * @return The manager object used by this modul.
         */
        static SmartPtr<StackTrace> Manager(void);

        /** Dtor. */
        virtual ~ThreadSafeStackTrace(void);

    protected:

        /** Ctor. */
        ThreadSafeStackTrace(void);

        /**
         * Pops a position from the stack.
         *
         * @param id The id of the position to be popped (used for sanity checks).
         */
        virtual void pop(int id);

        /**
         * Pushes a position onto the stack.
         *
         * @param func The name of the function.
         * @param file The name of the file.
         * @param line The line in the file.
         *
         * @return The id of the pushed position.
         */
        virtual int push(const char* func, const char* file, const int line);

        /**
         * Gets the anchor element to the stack and starts using this stack.
         *
         * @return The anchor element to the stack.
         */
        virtual StackElement* startUseStack(void);

        /**
         * Stops using the stack.
         *
         * @param stack The stack
         */
        virtual void stopUseStack(StackElement* stack);

    private:

        /** The critical section synchronizing the access to the stacks */
        vislib::sys::CriticalSection critSect;

        /**
         * Nested class of the thread stack roots.
         */
        typedef struct _threadstackroot_t {

            /** The thread id of this stack */
            DWORD id;

            /** The stack for this thread */
            StackElement *stack;

            /** The next stack */
            struct _threadstackroot_t *next;

        } ThreadStackRoot;

        /** The thread stacks */
        ThreadStackRoot* stacks;

        /** guard flag */
        bool valid;

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_THREADSAFESTACKTRACE_H_INCLUDED */

