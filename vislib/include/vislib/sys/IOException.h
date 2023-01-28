/*
 * IOException.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/sys/SystemException.h"


namespace vislib::sys {

/**
 * This exception indicates an I/O error.
 *
 * @author Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de)
 */
class IOException : public SystemException {

public:
    /**
     * Ctor.
     *
     * @param errorCode A system dependent error code.
     * @param file      The file the exception was thrown in.
     * @param line      The line the exception was thrown in.
     */
    IOException(const DWORD errorCode, const char* file, const int line);

    /**
     * Ctor.
     *
     * @param errorCode A system dependent error code.
     * @param msg       A detailed error message.
     * @param file      The file the exception was thrown in.
     * @param line      The line the exception was thrown in.
     */
    IOException(const DWORD errorCode, const char* msg, const char* file, const int line);

    /**
     * Ctor.
     *
     * @param errorCode A system dependent error code.
     * @param msg       A detailed error message.
     * @param file      The file the exception was thrown in.
     * @param line      The line the exception was thrown in.
     */
    IOException(const DWORD errorCode, const wchar_t* msg, const char* file, const int line);

    /**
     * Create a clone of 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    IOException(const IOException& rhs);

    /** Dtor. */
    ~IOException() override;

    /**
     * Assignment operator.
     *
     * @param rhs The right hand side operand.
     *
     * @return *this.
     */
    virtual IOException& operator=(const IOException& rhs);
};

} // namespace vislib::sys

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
