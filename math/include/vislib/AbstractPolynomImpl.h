/*
 * AbstractPolynomImpl.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTPOLYNOMIMPL_H_INCLUDED
#define VISLIB_ABSTRACTPOLYNOMIMPL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/mathfunctions.h"
#include "vislib/UnsupportedOperationException.h"


namespace vislib {
namespace math {


    /**
     * Implementation of polynom behaviour. Do not use this class directly. It
     * is used to implement the same inheritance pattern as for vectors,
     * points etc. See the documentation of AbstractVectorImpl for further
     * details.
     *
     * The one-dimensional polynom is defined by its coefficients a_0 ... a_d
     * as:
     *  f(x) := a_d * x^d + a_{d-1} * x^{d-1} + ... + a_1 * x + a_0
     *
     * T scalar type
     * D Degree of the polynom
     * S Coefficient storage
     * C Deriving subclass.
     */
    template<class T, unsigned int D, class S,
            template<class T, unsigned int D, class S> class C>
    class AbstractPolynomImpl {
    public:

        /** Dtor. */
        ~AbstractPolynomImpl(void);

        /**
         * Tests if the polynom 'src' can be assigned to this object. This is
         * possible if the effective degree of 'src' is equal or smaller than
         * D.
         *
         * @param src The polynom to test
         *
         * @return True if 'src' can be assigned to this object.
         */
        template<class Tp, unsigned int Dp, class Sp,
            template<class Tp, unsigned int Dp, class Sp> class Cp>
        inline bool CanAssign(
                const AbstractPolynomImpl<Tp, Dp, Sp, Cp>& src) const {
            return (src.EffectiveDegree() <= D);
        }

        /**
         * Sets all coefficients to zero.
         */
        void Clear(void);

        /**
         * Calculates the derivative of this polynom
         *
         * @return The derivative of this polynom
         */
        C<T, D - 1, T[D]> Derivative(void) const;

        /**
         * Answers the effective degree of the polynom. The effective degree
         * is less than D if a_d is zero.
         *
         * @return The effective degree of the polynom.
         */
        unsigned int EffectiveDegree(void) const;

        /**
         * Answer whether all coefficients are zero.
         *
         * @return True if all coefficients are zero.
         */
        bool IsZero(void) const;

        /**
         * Peeks at the internal representation of the coefficients of the
         * polynom (a_0, a_1, ... a_d).
         *
         * @return The internal representation of the coefficients of the
         *         polynom.
         */
        inline T* PeekCoefficients(void) {
            return this->coefficients;
        }

        /**
         * Peeks at the internal representation of the coefficients of the
         * polynom (a_0, a_1, ... a_d).
         *
         * @return The internal representation of the coefficients of the
         *         polynom.
         */
        inline const T* PeekCoefficients(void) const {
            return this->coefficients;
        }

        /**
         * Evaluates the polynom for 'x'
         *
         * @param x The value of the variable of the polynom
         *
         * @return The value of the polynom for 'x'
         */
        T operator()(T x);

        /**
         * Access to the i-th coefficient of the polynom (a_0, a_1, ... a_d).
         *
         * @param i The zero-based index of the coefficient of the polynom
         *          to be returned.
         *
         * @return The i-th coefficient of the polynom
         */
        inline T& operator[](int i) {
            return this->coefficients[i];
        }

        /**
         * Answer the i-th coefficient of the polynom (a_0, a_1, ... a_d).
         *
         * @param i The zero-based index of the coefficient of the polynom
         *          to be returned.
         *
         * @return The i-th coefficient of the polynom
         */
        inline T operator[](int i) const {
            return this->coefficients[i];
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         *
         * @throw IllegalParamException if 'rhs' has an effective degree larger
         *        than D.
         */
        template<class Tp, unsigned int Dp, class Sp,
            template<class Tp, unsigned int Dp, class Sp> class Cp>
        AbstractPolynomImpl<T, D, S, C>& operator=(
            const AbstractPolynomImpl<Tp, Dp, Sp, Cp>& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return true if this and rhs are equal
         */
        template<class Tp, unsigned int Dp, class Sp,
            template<class Tp, unsigned int Dp, class Sp> class Cp>
        bool operator==(const AbstractPolynomImpl<Tp, Dp, Sp, Cp>& rhs) const;

        /**
         * Test for inequality
         *
         * @param rhs The right hand side operand
         *
         * @return false if this and rhs are equal
         */
        template<class Tp, unsigned int Dp, class Sp,
            template<class Tp, unsigned int Dp, class Sp> class Cp>
        inline bool operator!=(const AbstractPolynomImpl<Tp, Dp, Sp, Cp>& rhs)
                const {
            return !(*this == rhs);
        }

    protected:

        /** Ctor. */
        inline AbstractPolynomImpl(void) { };

        /**
         * Finds the roots of the polynom for the linear case
         *
         * @param outRoots Array to receive the found roots
         * @param size The size of 'outRoots' in number of elements
         *
         * @return The found roots
         */
        unsigned int findRootsDeg1(T *outRoots, unsigned int size) const;

        /**
         * Finds the roots of the polynom for the quadratic case
         *
         * @param outRoots Array to receive the found roots
         * @param size The size of 'outRoots' in number of elements
         *
         * @return The found roots
         */
        unsigned int findRootsDeg2(T *outRoots, unsigned int size) const;

        /**
         * Finds the roots of the polynom for the cubic case
         *
         * @param outRoots Array to receive the found roots
         * @param size The size of 'outRoots' in number of elements
         *
         * @return The found roots
         */
        unsigned int findRootsDeg3(T *outRoots, unsigned int size) const;

        /**
         * Finds the roots of the polynom for the quartic case
         *
         * @param outRoots Array to receive the found roots
         * @param size The size of 'outRoots' in number of elements
         *
         * @return The found roots
         */
        unsigned int findRootsDeg4(T *outRoots, unsigned int size) const;

        /**
         * The D + 1 polynom coefficients
         */
        S coefficients;

        /* Allow access to the coefficients */
        template<class Tf1, unsigned int Df1, class Sf1,
            template<class Tf2, unsigned int Df2, class Sf2> class Cf1>
            friend class AbstractPolynomImpl;

    };


    /*
     * AbstractPolynomImpl<T, D, S, C>::~AbstractPolynomImpl
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    AbstractPolynomImpl<T, D, S, C>::~AbstractPolynomImpl(void) {
        // intentionally empty
    }


    /*
     * AbstractPolynomImpl<T, D, S, C>::Clear
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    void AbstractPolynomImpl<T, D, S, C>::Clear(void) {
        for (unsigned int i = 0; i <= D; i++) {
            this->coefficients[i] = static_cast<T>(0);
        }
    }


    /*
     * AbstractPolynomImpl<T, D, S, C>::Derivative
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    C<T, D - 1, T[D]> AbstractPolynomImpl<T, D, S, C>::Derivative(void) const {
        C<T, D - 1, T[D]> rv;

        for (unsigned int i = 0; i < D; i++) {
            rv.coefficients[i] = this->coefficients[i + 1] *
                static_cast<T>(i + 1);
        }

        return rv;
    }

    /*
     * AbstractPolynomImpl<T, D, S, C>::EffectiveDegree
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    unsigned int AbstractPolynomImpl<T, D, S, C>::EffectiveDegree(void) const {
        for (unsigned int i = D; i > 0; i--) {
            if (!vislib::math::IsEqual(this->coefficients[i],
                    static_cast<T>(0))) {
                return i;
            }
        }
        return 0;
    }


    /*
     * AbstractPolynomImpl<T, D, S, C>::IsZero
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    bool AbstractPolynomImpl<T, D, S, C>::IsZero(void) const {
        for (unsigned int i = 0; i <= D; i++) {
            if (!vislib::math::IsEqual(this->coefficients[i],
                    static_cast<T>(0))) {
                return false;
            }
        }
        return true;
    }


    /*
     * AbstractPolynomImpl<T, D, S, C>::operator()
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    T AbstractPolynomImpl<T, D, S, C>::operator()(T x) {
        T val = this->coefficients[0];
        T xp = static_cast<T>(1);

        for (unsigned int i = 1; i <= D; i++) {
            xp *= x;
            val += this->coefficients[i] * xp;
        }

        return val;
    }


    /*
     * AbstractPolynomImpl<T, D, S, C>::operator=
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    template<class Tp, unsigned int Dp, class Sp,
        template<class Tp, unsigned int Dp, class Sp> class Cp>
    AbstractPolynomImpl<T, D, S, C>&
    AbstractPolynomImpl<T, D, S, C>::operator=(
            const AbstractPolynomImpl<Tp, Dp, Sp, Cp>& rhs) {
        unsigned int rhsed = rhs.EffectiveDegree();
        if (rhsed > D) {
            throw vislib::IllegalParamException("rhs", __FILE__, __LINE__);
        }

        for (unsigned int i = 0; i <= rhsed; i++) {
            this->coefficients[i] = static_cast<T>(rhs.coefficients[i]);
        }
        for (unsigned int i = rhsed + 1; i <= D; i++) {
            this->coefficients[i] = static_cast<T>(0);
        }

        return *this;
    }


    /*
     * AbstractPolynomImpl<T, D, S, C>::operator==
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    template<class Tp, unsigned int Dp, class Sp,
        template<class Tp, unsigned int Dp, class Sp> class Cp>
    bool AbstractPolynomImpl<T, D, S, C>::operator==(
            const AbstractPolynomImpl<Tp, Dp, Sp, Cp>& rhs) const {

        for (unsigned int i = 0; i <= ((D < Dp) ? D : Dp); i++) {
            if (!vislib::math::IsEqual(this->coefficients[i],
                    static_cast<T>(rhs.coefficients[i]))) {
                return false;
            }
        }
        for (unsigned int i = ((D < Dp) ? D : Dp) + 1; i <= D; i++) {
            if (!vislib::math::IsEqual(this->coefficients[i],
                    static_cast<T>(0))) {
                return false;
            }
        }
        for (unsigned int i = ((D < Dp) ? D : Dp) + 1; i <= Dp; i++) {
            if (!vislib::math::IsEqual(rhs.coefficients[i],
                    static_cast<Tp>(0))) {
                return false;
            }
        }

        return true;
    }


    /*
     * AbstractPolynomImpl<T, D, S, C>::findRootsDeg1
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    unsigned int AbstractPolynomImpl<T, D, S, C>::findRootsDeg1(
            T *outRoots, unsigned int size) const {
        ASSERT(!IsEqual(this->coefficients[1], static_cast<T>(0)));
        ASSERT(size > 0);

        outRoots[0] = -this->coefficients[0] / this->coefficients[1];
        return 1;
    }


    /*
     * AbstractPolynomImpl<T, D, S, C>::findRootsDeg2
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    unsigned int AbstractPolynomImpl<T, D, S, C>::findRootsDeg2(
            T *outRoots, unsigned int size) const {
        ASSERT(!IsEqual(this->coefficients[2], static_cast<T>(0)));
        ASSERT(size > 0);

        T a = this->coefficients[2] * static_cast<T>(2);
        T b = this->coefficients[1] * this->coefficients[1]
            - this->coefficients[0] * this->coefficients[2] * static_cast<T>(4);

        if (IsEqual(b, static_cast<T>(0))) {
            // one root
            outRoots[0] = -this->coefficients[1] / a;
            return 1;
        } else if (b > static_cast<T>(0)) {
            // two roots
            outRoots[0] = (-this->coefficients[1] + b) / a;
            if (size > 1) {
                outRoots[1] = (-this->coefficients[1] - b) / a;
                return 2;
            }
            return 1;
        }

        return 0; // no roots
    }


    /*
     * AbstractPolynomImpl<T, D, S, C>::findRootsDeg3
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    unsigned int AbstractPolynomImpl<T, D, S, C>::findRootsDeg3(
            T *outRoots, unsigned int size) const {
        ASSERT(!IsEqual(this->coefficients[3], static_cast<T>(0)));
        ASSERT(size > 0);

        // TODO: Implement

        throw vislib::UnsupportedOperationException("findRootsDeg3",
            __FILE__, __LINE__);
    }


    /*
     * AbstractPolynomImpl<T, D, S, C>::findRootsDeg4
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    unsigned int AbstractPolynomImpl<T, D, S, C>::findRootsDeg4(
            T *outRoots, unsigned int size) const {
        ASSERT(!IsEqual(this->coefficients[4], static_cast<T>(0)));
        ASSERT(size > 0);

        // TODO: Implement

        throw vislib::UnsupportedOperationException("findRootsDeg4",
            __FILE__, __LINE__);
    }


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTPOLYNOMIMPL_H_INCLUDED */

