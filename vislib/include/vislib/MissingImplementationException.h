/*
 * MissingImplementationException.h
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
 * This exception should be used to indicated that function or methode has
 * been called, which is not yet implement.
 */
class MissingImplementationException : public Exception {

public:
    /**
     * Ctor.
     *
     * @param method The name of the method called.
     * @param file   The file the exception was thrown in.
     * @param line   The line the exception was thrown in.
     */
    MissingImplementationException(const char* method, const char* file, const int line);

    /**
     * Ctor.
     *
     * @param method The name of the method called.
     * @param file   The file the exception was thrown in.
     * @param line   The line the exception was thrown in.
     */
    MissingImplementationException(const wchar_t* method, const char* file, const int line);

    /**
     * Create a clone of 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    MissingImplementationException(const MissingImplementationException& rhs);

    /** Dtor. */
    ~MissingImplementationException() override;

    /**
     * Assignment operator.
     *
     * @param rhs The right hand side operand.
     *
     * @return *this.
     */
    MissingImplementationException& operator=(const MissingImplementationException& rhs);
};

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
