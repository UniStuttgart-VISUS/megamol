/*
 * OutOfRangeException.h
 *
 * Copyright (C) 2006 - 2010 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. All rights reserved.
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
class OutOfRangeException : public Exception {

public:
    /**
     * Ctor.
     *
     * @param val    The actual value.
     * @param minVal The allowed minimum value.
     * @param maxVal The allowed maximum value.
     * @param file      The file the exception was thrown in.
     * @param line      The line the exception was thrown in.
     */
    OutOfRangeException(const int val, const int minVal, const int maxVal, const char* file, const int line);

    /**
     * Ctor.
     *
     * @param val    The actual value.
     * @param minVal The allowed minimum value.
     * @param maxVal The allowed maximum value.
     * @param file      The file the exception was thrown in.
     * @param line      The line the exception was thrown in.
     */
    template<class T1, class T2, class T3>
    OutOfRangeException(const T1& val, const T2& minVal, const T3& maxVal, const char* file, const int line);

    /**
     * Create a clone of 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    OutOfRangeException(const OutOfRangeException& rhs);

    /** Dtor. */
    ~OutOfRangeException() override;

    /**
     * Assignment operator.
     *
     * @param rhs The right hand side operand.
     *
     * @return *this.
     */
    OutOfRangeException& operator=(const OutOfRangeException& rhs);

private:
    /**
     * Stores the formated message in the base class' storage
     *
     * @param val    The actual value.
     * @param minVal The allowed minimum value.
     * @param maxVal The allowed maximum value.
     */
    void storeMsg(int val, int minVal, int maxVal);
};


/*
 * OutOfRangeException::OutOfRangeException
 */
template<class T1, class T2, class T3>
OutOfRangeException::OutOfRangeException(
    const T1& val, const T2& minVal, const T3& maxVal, const char* file, const int line)
        : Exception(file, line) {
    this->storeMsg(static_cast<int>(val), static_cast<int>(minVal), static_cast<int>(maxVal));
}


} // namespace vislib

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
