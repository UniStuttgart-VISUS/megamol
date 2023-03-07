/*
 * IllegalParamException.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "Exception.h"


namespace vislib {

/**
 * This exception indicates an illegal parameter value.
 *
 * @author Christoph Mueller
 */
class IllegalParamException : public Exception {

public:
    /**
     * Ctor.
     *
     * @param paramName Name of the illegal parameter.
     * @param file      The file the exception was thrown in.
     * @param line      The line the exception was thrown in.
     */
    IllegalParamException(const char* paramName, const char* file, const int line);

    /**
     * Ctor.
     *
     * @param paramName Name of the illegal parameter.
     * @param file      The file the exception was thrown in.
     * @param line      The line the exception was thrown in.
     */
    IllegalParamException(const wchar_t* paramName, const char* file, const int line);

    /**
     * Create a clone of 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    IllegalParamException(const IllegalParamException& rhs);

    /** Dtor. */
    ~IllegalParamException() override;

    /**
     * Assignment operator.
     *
     * @param rhs The right hand side operand.
     *
     * @return *this.
     */
    virtual IllegalParamException& operator=(const IllegalParamException& rhs);
};
} // namespace vislib

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
