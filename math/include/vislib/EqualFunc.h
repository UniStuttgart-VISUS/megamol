/*
 * EqualFunc.h  14.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_EQUALFUNC_H_INCLUDED
#define VISLIB_EQUALFUNC_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/IllegalParamException.h"
#include "vislib/mathfunctions.h"
#include "vislib/UnsupportedOperationException.h"


namespace vislib {
namespace math {

    /**
     * This class implements a functor for comparing integral and floating point
     * numbers in a meaningful way.
     */
    template<class T> class EqualFunc {

    public:

        /** Ctor. */
        inline EqualFunc(void) {}

        /**
         * Answer whether 'lhs' and 'rhs' are equal.
         *
         * @param lhs The left hand side operand, a number.
         * @param rhs The right hand side operand, a number.
         *
         * @return true, if 'lhs' and 'rhs' are equal, false otherwise.
         */
        inline bool operator ()(const T lhs, const T rhs) const {
            return (lhs == rhs);
        }

    private:

        /**
         * Forbidden copy-ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        inline EqualFunc(const EqualFunc& rhs) {
            throw UnsupportedOperationException("vislib::math::EqualFunc",
                __FILE__, __LINE__);
        }

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If &'rhs' != this.
         */
        EqualFunc& operator =(const EqualFunc& rhs);
    };


    /*
     * vislib::math::EqualFunc::operator =
     */
    template<class T> 
    EqualFunc<T>& EqualFunc<T>::operator =(const EqualFunc& rhs) {
        if (this != &rhs) {
            throw IllegalParamException("rhs", __FILE__, __LINE__);
        }

        return *this;
    }


    /**
     * Specialisation for float. 
     */
    class FltEqualFunc : public EqualFunc<float> {

    public:

        /**
         * Create a new functor using 'epsilon' as epsilon value when
         * comparing.
         */
        inline FltEqualFunc(const float epsilon = FLOAT_EPSILON) 
                : epsilon(epsilon) {}

        inline bool operator ()(const float lhs, const float rhs) const {
            return (::fabsf(lhs - rhs) < this->epsilon);
        }

    private:

        /** The epsilon value for comparing. */
        float epsilon;
    };


    /**
     * Specialisation for double.
     */
    class DblEqualFunc : public EqualFunc<double> {

    public:

        /**
         * Create a new functor using 'epsilon' as epsilon value when
         * comparing.
         */
        inline DblEqualFunc(const double epsilon = DOUBLE_EPSILON) 
                : epsilon(epsilon) {}

        inline bool operator ()(const double lhs, const double rhs) const {
            return (::fabs(lhs - rhs) < this->epsilon);
        }

    private:

        /** The epsilon value for comparing. */
        double epsilon;
    };

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_EQUALFUNC_H_INCLUDED */
