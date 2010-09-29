/*
 * Trace.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_TRACE_H_INCLUDED
#define VISLIB_TRACE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <cstdarg>
#include <cstdio>

#include "vislib/types.h"


namespace vislib {

    /**
     * This is a utility for tracing some debug output.
     *
     * @author Christoph Mueller
     */
    class Trace {

    public:

        /**
         * Answer the only instance of this class.
         *
         * @return A reference to the only instance of this class.
         */
        static Trace& GetInstance(void);

        /**
         * Overrides the currently used 'main-and-most-of-the-time-only'
         * instance with 'inst'. The memory ownership of the object 'inst'
         * points to is not changed and the caller is responsible to keep it
         * valid as long as it is used, as well as to delete it when it is
         * no longer used.
         *
         * @param inst The new instance object. Must not be NULL
         */
        static void OverrideInstance(Trace *inst);

        /** 
         * Set this level to display all tracing information. If you use this
         * constant for tracing itself, the messages will only be output, if
         * LEVEL_ALL is also set as current tracing level.
         */
        static const UINT LEVEL_ALL;

        /**
         * Use this for logging errors. The value of this constant is 1, i. e.
         * messages with LEVEL_ERROR will always be printed, if any logging is
         * enabled.
         */
        static const UINT LEVEL_ERROR;

        /**
         * Use this for informative messages. The value of this constant 
         * is 200. 
         */
        static const UINT LEVEL_INFO;

        /** 
         * Use this for disabling tracing. The value is 0. It cannot be used
         * for tracing itself, but only for the current tracing level.
         */
        static const UINT LEVEL_NONE;

        /** Messages above this level are used for VISlib internal tracing. */
        static const UINT LEVEL_VL;

        /**
         * Messages with this level are used for VISlib internal output of
         * frequently recurring tasks.
         */
        static const UINT LEVEL_VL_ANNOYINGLY_VERBOSE;

        /** Messages with this level represent a VISlib internal error. */
        static const UINT LEVEL_VL_ERROR;

        /** Messages with this level represent a VISlib internal message. */
        static const UINT LEVEL_VL_INFO;

        /** 
        * Messages with this level represent a VISlib less important internal
        * message.
        */
        static const UINT LEVEL_VL_VERBOSE;

        /** Messages with this level represent a VISlib internal warning. */
        static const UINT LEVEL_VL_WARN;

        /**
         * Use this for warning messages. The value of this constant 
         * is 100. 
         */
        static const UINT LEVEL_WARN;

        /** Dtor. */
        ~Trace(void);

        /**
         * Enables the output of the tracer messages to the application or 
         * system debugger, i. e. to the Output window of Visual Studio.
         *
         * This setting has currently no effect on non-Windows systems.
         *
         * @param useDebugger true for enabling the debugger output, false for
         *                    disabling it.
         *
         * @return true, if the debugger output was successfully enabled or
         *         disabled, false otherwise.
         */
        bool EnableDebuggerOutput(const bool useDebugger);

        /**
         * Enables the output of the tracer messages to the file with the 
         * specified name. 
         *
         * @param filename The name of the file. If NULL, file output is 
         *                 disabled.
         *
         * @return true, if the log file was successfully opened, false 
         *         otherwise. If ('filename' == NULL), return true always.
         */
        bool EnableFileOutput(const char *filename);

        /**
         * Answer the current tracing level. Messages above this level will be
         * ignored.
         *
         * @return The current tracing level.
         */
        inline UINT GetLevel(void) const {
            return this->level;
        }

        /**
         * Set a new tracing level. Messages above this level will be ignored.
         *
         * @param level The new tracing level.
         */
        inline void SetLevel(const UINT level) {
            this->level = level;
        }

        /**
         * Set the trace prefix for console output. If NULL, no prefix will be 
         * added.
         *
         * @param prefix The new prefix or NULL for disabling prefixing.
         */
        void SetPrefix(const char *prefix);

        /**
         * Trace the message 'fmt', if an appropriate tracing level was set.
         *
         * @param level The trace level for the message.
         * @param fmt   The format string for the trace message.
         */
        void operator ()(const UINT level, const char *fmt, ...) throw();

        ///**
        // * Trace the message 'fmt'.
        // *
        // * @param fmt The format string for the trace message.
        // */
        //void operator ()(const char *fmt, ...);

    private:

        /** The default prefix of the console output. */
        static const char *DEFAULT_PREFIX;

        /** The 'main' and most of the time 'only' instance of this class. */
        static Trace *instance;

    public: // TODO: Not good! Think of better solution!

        /** 
         * Ctor.
         */
        Trace(void);

    private: // TODO: Not good! Think of better solution!

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        Trace(const Trace& rhs);

        /**
         * Does the actual tracing work.
         *
         * @param level The trace level for the message.
         * @param fmt   The format string for the trace message.
         * @param list  The variable argument list.
         */
        void trace(const UINT level, const char *fmt, va_list list) throw();

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        Trace& operator =(const Trace& rhs);

        /** Name of the output file. */
        char *filename;

        /** Handle for file output of tracer. */
        FILE *fp;

        /** 
         * The prefix of the console output. Prefixing is disabled, if this pointer
         * is NULL.
         */
        char *prefix;

        /** The current trace level. */
        UINT level;

        /** The the debugger API for writing strings. */
        bool useDebugger;

    };

} /* end namespace vislib */


#if defined(DEBUG) || defined(_DEBUG)
#define VLTRACE vislib::Trace::GetInstance()
#else /* defined(DEBUG) || defined(_DEBUG) */
#define VLTRACE(level, ...)
#endif /* defined(DEBUG) || defined(_DEBUG) */


#if defined(VISLIB_LEGACY_TRACE)
#define TRACE VLTRACE
#endif /* VISLIB_LEGACY_TRACE */


/* Short names for the predefined trace levels. */
#define VISLIB_TRCELVL_ERROR (vislib::Trace::LEVEL_ERROR)
#define VISLIB_TRCELVL_INFO (vislib::Trace::LEVEL_INFO)
#define VISLIB_TRCELVL_WARN (vislib::Trace::LEVEL_WARN)

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_TRACE_H_INCLUDED */
