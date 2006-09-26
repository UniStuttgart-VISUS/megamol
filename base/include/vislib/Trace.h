/*
 * Trace.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_TRACE_H_INCLUDED
#define VISLIB_TRACE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


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

        /**
         * Use this for warning messages. The value of this constant 
         * is 100. 
         */
        static const UINT LEVEL_WARN;

        /** Dtor. */
        ~Trace(void);

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
		 * Trace the message 'fmt', if an appropriate tracing level was set.
		 *
		 * @param level The trace level for the message.
		 * @param fmt   The format string for the trace message.
		 */
		void operator ()(const UINT level, const char *fmt, ...);

		///**
		// * Trace the message 'fmt'.
		// *
		// * @param fmt The format string for the trace message.
		// */
		//void operator ()(const char *fmt, ...);

    private:

		/** The only instance of this class. */
		static Trace instance;

		/** 
         * Ctor.
         */
		Trace(void);

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
		void trace(const UINT level, const char *fmt, va_list list);

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

		/** The current trace level. */
		UINT level;

	};

} /* end namespace vislib */


#if defined(DEBUG) || defined(_DEBUG)
#define TRACE vislib::Trace::GetInstance()
#else /* defined(DEBUG) || defined(_DEBUG) */
#define TRACE if (0) vislib::Trace::GetInstance()
#endif /* defined(DEBUG) || defined(_DEBUG) */

#endif /* VISLIB_TRACE_H_INCLUDED */
