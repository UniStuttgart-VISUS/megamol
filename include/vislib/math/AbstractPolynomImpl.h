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
#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif /* !M_PI */


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
        template<class Tp, unsigned int Dp, class Sp>
        inline bool CanAssign(const C<Tp, Dp, Sp>& src) const {
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
         * Addition on each coefficient (y-shift)
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        inline AbstractPolynomImpl<T, D, S, C>& operator+=(const T& rhs) {
            for (unsigned int i = 0; i <= D; i++) {
                this->coefficients[i] += rhs;
            }
            return *this;
        }

        /**
         * Subtraction on each coefficient (y-shift)
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        inline AbstractPolynomImpl<T, D, S, C>& operator-=(const T& rhs) {
            for (unsigned int i = 0; i <= D; i++) {
                this->coefficients[i] -= rhs;
            }
            return *this;
        }

        /**
         * Scalar multiplication
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        inline AbstractPolynomImpl<T, D, S, C>& operator*=(const T& rhs) {
            for (unsigned int i = 0; i <= D; i++) {
                this->coefficients[i] *= rhs;
            }
            return *this;
        }

        /**
         * Scalar division
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        inline AbstractPolynomImpl<T, D, S, C>& operator/=(const T& rhs) {
            for (unsigned int i = 0; i <= D; i++) {
                this->coefficients[i] /= rhs;
            }
            return *this;
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
        template<class Tp, unsigned int Dp, class Sp>
        AbstractPolynomImpl<T, D, S, C>& operator=(const C<Tp, Dp, Sp>& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return true if this and rhs are equal
         */
        template<class Tp, unsigned int Dp, class Sp>
        bool operator==(const C<Tp, Dp, Sp>& rhs) const;

        /**
         * Test for inequality
         *
         * @param rhs The right hand side operand
         *
         * @return false if this and rhs are equal
         */
        template<class Tp, unsigned int Dp, class Sp>
        inline bool operator!=(const C<Tp, Dp, Sp>& rhs)
                const {
            return !(*this == rhs);
        }

    protected:

        /**
         * Finds the roots of the polynom for the linear case
         *
         * @param a0 The const coefficient
         * @param a1 The linear coefficient
         * @param outRoots Array to receive the found roots
         * @param size The size of 'outRoots' in number of elements
         *
         * @return The found roots
         */
        static unsigned int findRootsDeg1(const T& a0, const T& a1,
            T *outRoots, unsigned int size);

        /**
         * Finds the roots of the polynom for the quadratic case
         *
         * @param a0 The const coefficient
         * @param a1 The linear coefficient
         * @param a2 The quadratic coefficient
         * @param outRoots Array to receive the found roots
         * @param size The size of 'outRoots' in number of elements
         *
         * @return The found roots
         */
        static unsigned int findRootsDeg2(const T& a0, const T& a1,
            const T& a2, T *outRoots, unsigned int size);

        /**
         * Finds the roots of the polynom for the cubic case
         *
         * @param a0 The const coefficient
         * @param a1 The linear coefficient
         * @param a2 The quadratic coefficient
         * @param a3 The cubic coefficient
         * @param outRoots Array to receive the found roots
         * @param size The size of 'outRoots' in number of elements
         *
         * @return The found roots
         */
        static unsigned int findRootsDeg3(const T& a0, const T& a1,
            const T& a2, const T& a3, T *outRoots, unsigned int size);

        /**
         * Finds the roots of the polynom for the quartic case
         *
         * @param a0 The const coefficient
         * @param a1 The linear coefficient
         * @param a2 The quadratic coefficient
         * @param a3 The cubic coefficient
         * @param a4 The quartic coefficient
         * @param outRoots Array to receive the found roots
         * @param size The size of 'outRoots' in number of elements
         *
         * @return The found roots
         */
        static unsigned int findRootsDeg4(const T& a0, const T& a1,
            const T& a2, const T& a3, const T& a4, T *outRoots,
            unsigned int size);

        /**
         * Removed double roots from the array 'outRoots'
         *
         * @param outRoots Pointer to the array of found roots
         * @param size The number of valid entries in 'outRoots'
         *
         * @return The numer of unique roots now in 'outRoots'
         */
        static unsigned int uniqueRoots(T *outRoots, unsigned int size);

        /** Ctor. */
        inline AbstractPolynomImpl(void) { };

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
    template<class Tp, unsigned int Dp, class Sp>
    AbstractPolynomImpl<T, D, S, C>&
    AbstractPolynomImpl<T, D, S, C>::operator=(const C<Tp, Dp, Sp>& rhs) {
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
    template<class Tp, unsigned int Dp, class Sp>
    bool AbstractPolynomImpl<T, D, S, C>::operator==(
            const C<Tp, Dp, Sp>& rhs) const {

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
    unsigned int AbstractPolynomImpl<T, D, S, C>::findRootsDeg1(const T& a0,
            const T& a1, T *outRoots, unsigned int size) {
        ASSERT(!IsEqual(a1, static_cast<T>(0)));
        ASSERT(size > 0);

        outRoots[0] = -a0 / a1;

        return 1;
    }


    /*
     * AbstractPolynomImpl<T, D, S, C>::findRootsDeg2
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    unsigned int AbstractPolynomImpl<T, D, S, C>::findRootsDeg2(const T& a0,
            const T& a1, const T& a2, T *outRoots, unsigned int size) {
        ASSERT(!IsEqual(a2, static_cast<T>(0)));
        ASSERT(size > 0);

        T a = a2 * static_cast<T>(2);
        T b = a1 * a1 - a0 * a2 * static_cast<T>(4);

        if (IsEqual(b, static_cast<T>(0))) {
            // one root
            outRoots[0] = -a1 / a;
            return 1;
        } else if (b > static_cast<T>(0)) {
            // two roots
            b = static_cast<T>(::sqrt(static_cast<double>(b)));
            outRoots[0] = (-a1 + b) / a;
            if (size > 1) {
                outRoots[1] = (-a1 - b) / a;
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
    unsigned int AbstractPolynomImpl<T, D, S, C>::findRootsDeg3(const T& a0,
            const T& a1, const T& a2, const T& a3, T *outRoots,
            unsigned int size) {
        ASSERT(!IsEqual(a3, static_cast<T>(0)));
        ASSERT(size > 0);
        // calculation following description at
        // http://www.mathe.tu-freiberg.de/~hebisch/cafe/kubisch.html
        // (14.03.2010)

        T p = (static_cast<T>(3) * a3 * a1) - (a2 * a2);
        T q = (static_cast<T>(27) * a3 * a3 * a0)
            - (static_cast<T>(9) * a3 * a2 * a1)
            + (static_cast<T>(2) * a2 * a2 * a2);

        if (IsEqual(p, static_cast<T>(0))) { // special case
            // y1 per double calculation; not nice, but ok for now
            T y1 = static_cast<T>(::pow(static_cast<double>(-q), 1.0 / 3.0));
            outRoots[0] = (y1 - a2) / (static_cast<T>(3) * a3);
            if (size > 1) {
                // quadratic polynom through polynom division
                T y23[2];
                T qa0 = (static_cast<T>(3) * p) - (y1 * y1);
                T qa1 = -y1;
                T qa2 = static_cast<T>(1);
                unsigned int qrc = findRootsDeg2(qa0, qa1, qa2, y23, 2);
                if (qrc > 0) {
                    outRoots[1] = (y23[0] - a2) / (static_cast<T>(3) * a3);
                }
                if ((qrc > 1) && (size > 2)) {
                    outRoots[2] = (y23[1] - a2) / (static_cast<T>(3) * a3);
                    return uniqueRoots(outRoots, 3);
                }
                return uniqueRoots(outRoots, 2);
            }
            return 1;
        }
        // p != 0

        T dis = (q * q) + (static_cast<T>(4) * p * p * p);
        if (dis < static_cast<T>(0)) {
            // casus irreducibilis
            ASSERT(p < static_cast<T>(0)); // or square-root would be complex

            double cosphi = static_cast<double>(-q)
                / (2.0 * ::sqrt(-static_cast<double>(p * p * p)));
            double phi = ::acos(cosphi);
            double sqrtNegP = ::sqrt(static_cast<double>(-p));
            double phiThird = phi / 3.0;
            double piThird = M_PI / 3.0;

            T y1 = static_cast<T>(2.0 * sqrtNegP * ::cos(phiThird));
            T y2 = static_cast<T>(-2.0 * sqrtNegP * ::cos(phiThird + piThird));
            T y3 = static_cast<T>(-2.0 * sqrtNegP * ::cos(phiThird - piThird));

            outRoots[0] = (y1 - a2) / (static_cast<T>(3) * a3);
            if (size > 1) {
                outRoots[1] = (y2 - a2) / (static_cast<T>(3) * a3);
                if (size > 2) {
                    outRoots[2] = (y3 - a2) / (static_cast<T>(3) * a3);
                    return uniqueRoots(outRoots, 3);
                }
                return uniqueRoots(outRoots, 2);
            }
            return 1;
        }

        double sqrtDis = ::sqrt(static_cast<double>(dis));
        T u = static_cast<T>(0.5) * static_cast<T>(
            ::pow(-4.0 * static_cast<double>(q) + 4.0 * sqrtDis,
                1.0 / 3.0));
        T v = static_cast<T>(0.5) * static_cast<T>(
            ::pow(-4.0 * static_cast<double>(q) - 4.0 * sqrtDis,
                1.0 / 3.0));
        T y1 = u + v;
        outRoots[0] = (y1 - a2) / (static_cast<T>(3) * a3);

        if (size > 1) {
            if (IsEqual(u, v)) {
                T y2 = static_cast<T>(-0.5) * (u + v);
                outRoots[1] = (y2 - a2) / (static_cast<T>(3) * a3);
                return uniqueRoots(outRoots, 2);
            } // else second and thrid root are complex
              // (we only return real roots)
        }

        return 1;
    }


    /*
     * AbstractPolynomImpl<T, D, S, C>::findRootsDeg4
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    unsigned int AbstractPolynomImpl<T, D, S, C>::findRootsDeg4(const T& a0,
            const T& a1, const T& a2, const T& a3, const T& a4, T *outRoots,
            unsigned int size) {
        ASSERT(!IsEqual(a4, static_cast<T>(0)));
        ASSERT(size > 0);

        // Implementation of Ferrari-Lagrange method for solving
        //  x^4 + ax^3 + bx^2 + cx + d = 0
        T a = a3 / a4;
        T b = a2 / a4;
        T c = a1 / a4;
        T d = a0 / a4;

        T asq = a * a;
        T p = b;
        T q = a * c - static_cast<T>(4) * d;
        T r = (asq - static_cast<T>(4) * b) * d + c * c;
        T y;
        { // finds the smallest x for cubic polynom x^3 + px^2 + qx + r = 0;
            T cr[3];
            unsigned int crc
                = findRootsDeg3(r, q, p, static_cast<T>(1), cr, 3);
            ASSERT(crc > 0);
            if (crc == 3) y = Min(Min(cr[0], cr[1]), cr[2]);
            else if (crc == 2) y = Min(cr[0], cr[1]);
            else y = cr[0];
        }

        T esq = static_cast<T>(0.25) * asq - b - y;
        if (IsEqual(esq, static_cast<T>(0))) return 0;

        T fsq = static_cast<T>(0.25) * y * y - d;
        if (IsEqual(fsq, static_cast<T>(0))) return 0;

        T ef = -(static_cast<T>(0.25) * a * y + static_cast<T>(0.5) * c);
        T e, f;

        if (((a > static_cast<T>(0)) && (y > static_cast<T>(0))
                    && (c > static_cast<T>(0)))
                || ((a > static_cast<T>(0)) && (y < static_cast<T>(0))
                    && (c < static_cast<T>(0)))
                || ((a < static_cast<T>(0)) && (y < static_cast<T>(0))
                    && (c > static_cast<T>(0)))
                || ((a < static_cast<T>(0)) && (y > static_cast<T>(0))
                    && (c < static_cast<T>(0)))
                || IsEqual(a, static_cast<T>(0))
                || IsEqual(y, static_cast<T>(0))
                || IsEqual(c, static_cast<T>(0))) {
            /* use ef - */

            if ((b < static_cast<T>(0)) && (y < static_cast<T>(0))
                    && (esq > static_cast<T>(0))) {
                e = static_cast<T>(::sqrt(static_cast<double>(esq)));
                f = ef / e;
            } else if ((d < static_cast<T>(0)) && (fsq > static_cast<T>(0))) {
                f = static_cast<T>(::sqrt(static_cast<double>(fsq)));
                e = ef / f;
            } else {
                e = static_cast<T>(::sqrt(static_cast<double>(esq)));
                f = static_cast<T>(::sqrt(static_cast<double>(fsq)));
                if (ef < static_cast<T>(0)) f = -f;
            }
        } else {
            e = static_cast<T>(::sqrt(static_cast<double>(esq)));
            f = static_cast<T>(::sqrt(static_cast<double>(fsq)));
            if (ef < static_cast<T>(0)) f = -f;
        }

        /* note that e >= nought */
        T ainv2 = a * static_cast<T>(0.5);
        T g = ainv2 - e;
        T gg = ainv2 + e;

        if (((b > static_cast<T>(0)) && (y > static_cast<T>(0)))
                || ((b < static_cast<T>(0)) && (y < static_cast<T>(0)))) {
            if ((a > static_cast<T>(0)) && !IsEqual(e, static_cast<T>(0))) {
                g = (b + y) / gg;
            } else if (!IsEqual(e, static_cast<T>(0))) {
                gg = (b + y) / g;
            }
        }

        T h, hh;
        if (IsEqual(y, static_cast<T>(0)) && IsEqual(f, static_cast<T>(0))) {
            h = hh = static_cast<T>(0);
        } else if (((f > static_cast<T>(0)) && (y < static_cast<T>(0)))
                || ((f < static_cast<T>(0)) && (y > static_cast<T>(0)))) {
            hh = static_cast<T>(-0.5) * y + f;
            h = d / hh;
        } else {
            h = static_cast<T>(-0.5) * y - f;
            hh = d / h;
        }

        unsigned int cnt = findRootsDeg2(hh, gg, static_cast<T>(1), outRoots, size);
        if ((size - cnt) > 0) {
            cnt += findRootsDeg2(h, g, static_cast<T>(1), outRoots + cnt, size - cnt);
        }

        return uniqueRoots(outRoots, cnt);
    }


    /*
     * AbstractPolynomImpl<T, D, S, C>::uniqueRoots
     */
    template<class T, unsigned int D, class S,
        template<class T, unsigned int D, class S> class C>
    unsigned int AbstractPolynomImpl<T, D, S, C>::uniqueRoots(T *outRoots,
            unsigned int size) {
        if (size <= 1) return size;
        if (size == 2) return (IsEqual(outRoots[0], outRoots[1])) ? 1 : 2;

        // O(n^2) search to keep the implementation simple
        // change this if the degree of the polynom gets LARGE
        bool found;
        unsigned int cnt = 1;
        for (unsigned int i = 1; i < size; i++) {
            found = false;
            for (unsigned int j = 0; j < cnt; j++) {
                if (IsEqual(outRoots[i], outRoots[j])) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                outRoots[cnt++] = outRoots[i];
            }
        }

        return cnt;
    }


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTPOLYNOMIMPL_H_INCLUDED */

