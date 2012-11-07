/*
 * StackTrace.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_STACKTRACE_H_INCLUDED
#define VISLIB_STACKTRACE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/SmartPtr.h"


namespace vislib {


    /**
     * The manager class for stack trace.
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
     * You should use the 'VL_STACKTRACE' macro to push functions onto
     * the tracing stack.
     */
    class StackTrace {
    public:

        /**
         * Stack trace marker used to automatically push trace information
         * for functions. You should not create instances of this type on the
         * heap!
         */
        class Marker {
        public:

            /**
             * Ctor.
             * Create this object to push 'functionName' on the trace stack.
             * It is automatically popped when this object is destroied.
             *
             * @param functionName The name to be pushed onto the trace stack.
             * @param file The file where is marker is placed (use '__FILE__')
             * @param line The line number where is marker is placed
             *             (use '__LINE__')
             */
            Marker(const char* functionName, const char* file, const int line);

            /**
             * Dtor.
             */
            ~Marker(void);

        private:

            /**
             * Forbidden copy ctor.
             *
             * @param src The object to clone from.
             */
            Marker(const Marker& src);

            /**
             * Forbidden assignment operator.
             *
             * @param rhs The right hand side operand.
             *
             * @return Reference to 'this'.
             */
            Marker& operator=(const Marker& rhs);

            /** The pushed position id */
            int id;

        };

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
        static void GetStackString(char *str, unsigned int &strSize);

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
        static void GetStackString(wchar_t *str, unsigned int &strSize);

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

        /**
         * Dtor.
         *
         * Note: Do not call this dtor directly or directly destroy any
         * instance of this class. The 'SmartPtr' objects will handle the
         * deletion of the 'StackTrace' objects automatically.
         */
        virtual ~StackTrace(void);

        /** The marker object can access the stack */
        friend class Marker;

    protected:

        /**
         * Private nested helper class building up the stack
         */
        typedef struct _stackelement_t {

            /** The function name */
            const char* func;

            /** The file name */
            const char* file;

            /** The line */
            int line;

            /** The next element on the stack */
            struct _stackelement_t *next;

        } StackElement;

        /** Ctor. */
        StackTrace(void);

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
        virtual void getStackString(char *str, unsigned int &strSize);

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
        virtual void getStackString(wchar_t *str, unsigned int &strSize);

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

        /** The anchor pointer of the stack */
        StackElement *stack;

    private:

        /** The stack trace manager of this module */
        static SmartPtr<StackTrace> &manager;

    };

} /* end namespace vislib */


#if defined(DEBUG) || defined(_DEBUG)
#define VLSTACKTRACE(FN, FILE, LINE) vislib::StackTrace::Marker \
    __localStackTraceMarker(FN, FILE, LINE);
#else /* defined(DEBUG) || defined(_DEBUG) */
#define VLSTACKTRACE(FN, FILE, LINE)
#endif /* defined(DEBUG) || defined(_DEBUG) */


#if __GNUC__
#define VLAUTOSTACKTRACE VLSTACKTRACE(__PRETTY_FUNCTION__, __FILE__, __LINE__)
#else /* __GNUC__ */
// Note: VC __FUNCTION__ is pretty by design...
#define VLAUTOSTACKTRACE VLSTACKTRACE(__FUNCTION__, __FILE__, __LINE__)
#endif /* __GNUC__ */


/* Deprecated */
#define VISLIB_STACKTRACE(FN, FILE, LINE) VLSTACKTRACE(#FN, FILE, LINE)


#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_STACKTRACE_H_INCLUDED */

