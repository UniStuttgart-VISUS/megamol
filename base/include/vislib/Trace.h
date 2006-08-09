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

#include "vislib/tchar.h"
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

        /** Dtor. */
        ~Trace(void);

		inline UINT GetLevel(void) const {
			return this->level;
		}

		inline void SetLevel(const UINT level) {
			this->level = level;
		}

		/**
		 * Trace the message 'fmt', if an appropriate tracing level was set.
		 *
		 * @param level The trace level for the message.
		 * @param fmt   The format string for the trace message.
		 */
		void operator ()(const UINT level, const TCHAR *fmt, ...);

		/**
		 * Trace the message 'fmt'.
		 *
		 * @param fmt The format string for the trace message.
		 */
		void operator ()(const TCHAR *fmt, ...);

    private:

		/** The only instance of this class. */
		static Trace *instance;

		/** Ctor. */
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
		void trace(const UINT level, const TCHAR *fmt, va_list list);

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
