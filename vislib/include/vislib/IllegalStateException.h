/*
 * IllegalStateException.h  22.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ILLEGALSTATEEXCEPTION_H_INCLUDED
#define VISLIB_ILLEGALSTATEEXCEPTION_H_INCLUDED
#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "Exception.h"


namespace vislib {

/**
 * This exception indicates that an object would enter an illegal state when
 * calling a method or is already in an illegal state.
 *
 * @author Christoph Mueller
 */
class IllegalStateException : public Exception {

public:
    /**
     * Ctor.
     *
     * @param msg  The exception detail message.
     * @param file The file the exception was thrown in.
     * @param line The line the exception was thrown in.
     */
    IllegalStateException(const char* msg, const char* file, const int line);

    /**
     * Ctor.
     *
     * @param msg  The exception detail message.
     * @param file The file the exception was thrown in.
     * @param line The line the exception was thrown in.
     */
    IllegalStateException(const wchar_t* msg, const char* file, const int line);

    /**
     * Create a clone of 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    IllegalStateException(const IllegalStateException& rhs);

    /** Dtor. */
    ~IllegalStateException() override;

    /**
     * Assignment operator.
     *
     * @param rhs The right hand side operand.
     *
     * @return *this.
     */
    virtual IllegalStateException& operator=(const IllegalStateException& rhs);
};
} // namespace vislib

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ILLEGALSTATEEXCEPTION_H_INCLUDED */
