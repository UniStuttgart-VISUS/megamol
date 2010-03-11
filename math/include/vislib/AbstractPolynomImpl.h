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

#include "vislib/mathfunctions.h"
#include "vislib/assert.h"
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

        ///** 
        // * Typedef for a polynom with "deep storage" class. Objects of this type
        // * are used as return value for methods and operators that must create
        // * and return new instances.
        // */
        //typedef C<T, D, T[D]> DeepStoragePolynom;

        /** Dtor. */
        ~AbstractPolynomImpl(void);

        /**
         * Sets all coefficients to zero.
         */
        void Clear(void);

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
         */
        AbstractPolynomImpl<T, D, S, C>& operator=(
            const AbstractPolynomImpl<T, D, S, C>& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return true if this and rhs are equal
         */
        bool operator==(const AbstractPolynomImpl<T, D, S, C>& rhs) const;

        /**
         * Test for inequality
         *
         * @param rhs The right hand side operand
         *
         * @return false if this and rhs are equal
         */
        inline bool operator!=(const AbstractPolynomImpl<T, D, S, C>& rhs)
                const {
            return !(*this == rhs);
        }

    protected:

        /** Ctor. */
        inline AbstractPolynomImpl(void) { };

        ///**
        // * Finds the roots of a polynom of degree 'Dp'
        // *
        // * @param outRoots The array to receive the roots
        // * @param size The size of 'outRoots' in number of elements
        // *
        // * @return The number of roots written to 'outRoots'
        // */
        //template<unsigned int Dp>
        //unsigned int findRoots(T *outRoots, unsigned int size) const; // {

        //    // TODO: Implement some newton

        //    throw vislib::UnsupportedOperationException("findRoots", __FILE__, __LINE__);

        //    return 0;
        //}

        ///**
        // * Finds the roots of a polynom of degree 'Dp'
        // *
        // * @param outRoots The array to receive the roots
        // * @param size The size of 'outRoots' in number of elements
        // *
        // * @return The number of roots written to 'outRoots'
        // */
        //template<>
        //unsigned int findRoots<1>(T *outRoots, unsigned int size) const {
        //    return 0;
        //}

        ///**
        // * Finds the roots of a polynom of degree 'Dp'
        // *
        // * @param outRoots The array to receive the roots
        // * @param size The size of 'outRoots' in number of elements
        // *
        // * @return The number of roots written to 'outRoots'
        // */
        //template<>
        //unsigned int findRoots<2>(T *outRoots, unsigned int size) const {
        //    return 0;
        //}

        ///**
        // * Finds the roots of a polynom of degree 'Dp'
        // *
        // * @param outRoots The array to receive the roots
        // * @param size The size of 'outRoots' in number of elements
        // *
        // * @return The number of roots written to 'outRoots'
        // */
        //template<>
        //unsigned int findRoots<3>(T *outRoots, unsigned int size) const {
        //    return 0;
        //}

        ///**
        // * Finds the roots of a polynom of degree 'Dp'
        // *
        // * @param outRoots The array to receive the roots
        // * @param size The size of 'outRoots' in number of elements
        // *
        // * @return The number of roots written to 'outRoots'
        // */
        //template<>
        //unsigned int findRoots<4>(T *outRoots, unsigned int size) const {
        //    return 0;
        //}

        /**
         * The polynom coefficients.
         */
        S coefficients;

    };


    /*
     * AbstractPolynomImpl<T, D, S, C>::~AbstractPolynomImpl
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    AbstractPolynomImpl<T, D, S, C>::~AbstractPolynomImpl(void) {
        // intentionally empty
    }

    //template<class T, unsigned int D, class S,
    //    template<class T, unsigned int D, class S> class C>
    //template<unsigned int Dp>
    //unsigned int AbstractPolynomImpl<T, D, S, C>::findRoots(T* outRoots, unsigned int size) const {

    //    return 0;
    //}


    /*
     * AbstractPolynomImpl<T, D, S, C>::Clear
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    void AbstractPolynomImpl<T, D, S, C>::Clear(void) {
        for (unsigned int i = 0; i < D; i++) {
            this->coefficients[i] = static_cast<T>(0);
        }
    }


    /*
     * AbstractPolynomImpl<T, D, S, C>::IsZero
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    bool AbstractPolynomImpl<T, D, S, C>::IsZero(void) const {
        for (unsigned int i = 0; i < D; i++) {
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

        for (unsigned int i = 1; i < D; i++) {
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
    AbstractPolynomImpl<T, D, S, C>&
    AbstractPolynomImpl<T, D, S, C>::operator=(
            const AbstractPolynomImpl<T, D, S, C>& rhs) {
        for (unsigned int i = 0; i < D; i++) {
            this->coefficients[i] = rhs.coefficients[i];
        }
        return *this;
    }


    /*
     * AbstractPolynomImpl<T, D, S, C>::operator==
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    bool AbstractPolynomImpl<T, D, S, C>::operator==(
            const AbstractPolynomImpl<T, D, S, C>& rhs) const {
        for (unsigned int i = 0; i < D; i++) {
            if (!vislib::math::IsEqual(this->coefficients[i],
                    rhs.coefficients[i])) {
                return false;
            }
        }
        return true;
    }


    ///*
    // * AbstractPolynomImpl<T, D, S, C>::findRootsDeg1
    // */
    //template<class T, unsigned int D, class S,
    //    template<class T, unsigned int D, class S> class C>
    //unsigned int AbstractPolynomImpl<T, D, S, C>::findRootsDeg1(
    //        T *outRoots, unsigned int size) const {

    //    if (size < 1) return 0; // not enough size to store

    //    if (IsEqual(this->coefficients[1], static_cast<T>(0))) return 0;
    //        // line is parallel to x-axis

    //    outRoots[0] = -this->coefficients[0] / this->coefficients[1];
    //    return 1;

    //    throw vislib::UnsupportedOperationException("FindRoots",
    //        __FILE__, __LINE__);
    //}


    ///*
    // * AbstractPolynomImpl<T, D, S, C>::findRootsDeg2
    // */
    //template<class T, unsigned int D, class S,
    //    template<class T, unsigned int D, class S> class C>
    //unsigned int AbstractPolynomImpl<T, D, S, C>::findRootsDeg2(
    //        T *outRoots, unsigned int size) const {
    //    throw vislib::UnsupportedOperationException("FindRoots",
    //        __FILE__, __LINE__);
    //}


    ///*
    // * AbstractPolynomImpl<T, D, S, C>::findRootsDeg3
    // */
    //template<class T, unsigned int D, class S,
    //    template<class T, unsigned int D, class S> class C>
    //unsigned int AbstractPolynomImpl<T, D, S, C>::findRootsDeg3(
    //        T *outRoots, unsigned int size) const {
    //    throw vislib::UnsupportedOperationException("FindRoots",
    //        __FILE__, __LINE__);
    //}


    ///*
    // * AbstractPolynomImpl<T, D, S, C>::findRootsDeg4
    // */
    //template<class T, unsigned int D, class S,
    //    template<class T, unsigned int D, class S> class C>
    //unsigned int AbstractPolynomImpl<T, D, S, C>::findRootsDeg4(
    //        T *outRoots, unsigned int size) const {
    //    throw vislib::UnsupportedOperationException("FindRoots",
    //        __FILE__, __LINE__);
    //}


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTPOLYNOMIMPL_H_INCLUDED */

