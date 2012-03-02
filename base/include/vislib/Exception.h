/*
 * Exception.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_EXCEPTION_H_INCLUDED
#define VISLIB_EXCEPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/tchar.h"
#include "vislib/types.h"


namespace vislib {

    /**
     * Superclass for all exceptions. The class provides basic functionality for
     * exception messages.
     *
     * @author Christoph Mueller
     */
    class Exception {

    public:

        /**
         * Create a new exception. The ownership of the memory designated by
         * 'msg' and 'file' remains at the caller, the class creates a deep
         * copy.
         *
         * @param msg  A description of the exception.
         * @param file The file the exception was thrown in.
         * @param line The line the exception was thrown in.
         */
        Exception(const char *msg, const char *file, const int line);

        /**
         * Create a new exception. The ownership of the memory designated by
         * 'msg' and 'file' remains at the caller, the class creates a deep
         * copy.
         *
         * @param msg  A description of the exception.
         * @param file The file the exception was thrown in.
         * @param line The line the exception was thrown in.
         */
        Exception(const wchar_t *msg, const char *file, const int line);

        /**
         * Create a new exception. The ownership of the memory designated by
         * 'file' remains at the caller, the class creates a deep copy.
         *
         * @param file The file the exception was thrown in.
         * @param line The line the exception was thrown in.
         */
        Exception(const char *file, const int line);

        /**
         * Create a clone of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        Exception(const Exception& rhs);

        /** Dtor. */
        virtual ~Exception(void);

        /**
         * Answer the file the exception was thrown in. The onwnership of the
         * memory remains at the object.
         *
         * @return The file the exception was thrown in.
         */
        inline const char *GetFile(void) const {
            return this->file;
        }

        /**
         * Answer the line the exception was thrown in.
         *
         * @return The line the exception was thrown in.
         */
        inline int GetLine(void) const {
            return this->line;
        }

        /**
         * Answer the file the exception description text. The pointer returned
         * is valid until the next call to a GetMsg* method. The ownership of the
         * memory remains at the object.
         *
         * @return The exception message.
         */
        inline const TCHAR *GetMsg(void) const {
#if defined(UNICODE) || defined(_UNICODE)
            return reinterpret_cast<const TCHAR *>(this->GetMsgW());
#else /* defined(UNICODE) || defined(_UNICODE) */
            return reinterpret_cast<const TCHAR *>(this->GetMsgA());
#endif /* defined(UNICODE) || defined(_UNICODE) */
        }

        /**
         * Answer the file the exception description text. The pointer returned
         * is valid until the next call to a GetMsg* method. The ownership of the
         * memory remains at the object.
         *
         * @return The exception message.
         */
        virtual const char *GetMsgA(void) const;

        /**
         * Answer the file the exception description text. The pointer returned
         * is valid until the next call to a GetMsg* method. The ownership of the
         * memory remains at the object.
         *
         * @return The exception message.
         */
        virtual const wchar_t *GetMsgW(void) const;

        /**
         * Answer the call stack trace from where the exception was thrown. If
         * no call stack is available an empty string is returned.
         *
         * @return The call stack trace from where the exception was thrown.
         */
        inline const char *GetStack(void) const {
            return (this->stack == NULL) ? "" : this->stack;
        }

        /**
         * Answer whether the exception knows the call stack trace from where
         * it was thrown.
         *
         * @return 'true' if a call stack trace is available,
         *         'false' otherwise.
         */
        inline bool HasStack(void) const {
            return this->stack != NULL;
        }

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        virtual Exception& operator =(const Exception& rhs);

    protected:

        /**
         * Tries to fetch the current call stack trace.
         */
        void fetchStack(void);

        /**
         * Set a new detail message.
         *
         * @param fmt The format string like in printf.
         * @param ... Additional parameters.
         */
        void formatMsg(const char *fmt, ...);

        /**
         * Set a new detail message.
         *
         * @param fmt The format string like in printf.
         * @param ... Additional parameters.
         */
        void formatMsg(const wchar_t *fmt, ...);

        /**
         * Set a new file.
         *
         * @param file The file name.
         */
        void setFile(const char *file);

        /**
         * Set a new detail message.
         *
         * @param msg The new exception detail message.
         */
        void setMsg(const char *msg) const;

        /**
         * Set a new detail message.
         *
         * @param msg The new exception detail message.
         */
        void setMsg(const wchar_t *msg) const;

    private:

        /** The file the exception was thrown in. */
        char *file;

        /** The line number the exception was thrown in. */
        int line;

        /** Remember whether 'msg' points to a Unicode or ANSI string. */
        mutable bool isMsgUnicode;

        /** The exception message. */
        mutable void *msg;

        /** The stack trace, if available */
        char *stack;

    };
}

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_EXCEPTION_H_INCLUDED */
