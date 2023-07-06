/*
 * NoSuchElementException.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Exception.h"


namespace vislib {


/**
 * This exception should be used to indicated that a non-existing
 * element was requested from some kind of collection.
 */
class NoSuchElementException : public Exception {

public:
    /**
     * Ctor.
     *
     * @param msg  A human readable message.
     * @param file The file the exception was thrown in.
     * @param line The line the exception was thrown in.
     */
    NoSuchElementException(const char* msg, const char* file, const int line);

    /**
     * Create a clone of 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    NoSuchElementException(const NoSuchElementException& rhs);

    /** Dtor. */
    ~NoSuchElementException() override;

    /**
     * Assignment operator.
     *
     * @param rhs The right hand side operand.
     *
     * @return *this.
     */
    NoSuchElementException& operator=(const NoSuchElementException& rhs);
};

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
