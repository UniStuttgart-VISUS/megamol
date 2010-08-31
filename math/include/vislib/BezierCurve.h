/*
 * BezierCurve.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Copyright (C) 2006 - 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_BEZIERCURVE_H_INCLUDED
#define VISLIB_BEZIERCURVE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/forceinline.h"
#include "vislib/mathfunctions.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Point.h"
#include "vislib/Vector.h"


namespace vislib {
namespace math {


    /**
     * Stores and evaluates a bézier curve of degree E. The curve is defined
     * by E + 1 control points (type T) forming the control polygon.
     *
     * Template parameters:
     *  T type (e. g. Point<float, 3>); Must support the method
     *    'Interpolate(rhs, float t)' with 0 <= t <= 1 for linear
     *    interpolation
     *  E Degree
     */
    template <class T, unsigned int E>
    class BezierCurve {
    public:

        /**
         * Ctor.
         * Creates an empty curve with all control points placed in the
         * origin.
         */
        BezierCurve(void);

        /** Dtor. */
        ~BezierCurve(void);

        /*
         * Evaluates the position on the bézier curve for the
         * interpolation parameter t based on the algorithm of de Casteljau.
         *
         * @param outPt The variable to be set to the calculated point.
         * @param t The interpolation parameter (0..1); Results for values
         *          outside this range are undefined.
         *
         * @return The same reference as 'outPt'
         */
        T& CalcPoint(T& outPt, float t) const;

        /**
         * Calculates the tangent along the curve for the interpolation
         * value t.
         *
         * @param outVec The variable to be set. The type must be compatible
         *               with the result type of the 'operator -' of the
         *               template type T.
         * @param t The interpolation parameter (0..1); Results for values
         *          outside this range are undefined.
         *
         * @return The same reference as 'outVec'
         */
        template<class Tp>
        Tp& CalcTangent(Tp& outVec, float t) const;

        /**
         * Accesses the idx-th control point
         *
         * @param idx The index of the control point to be set (0..E)
         *
         * @return A reference to the idx-th control point
         */
        T& ControlPoint(unsigned int idx);

        /**
         * Accesses the idx-th control point read-only
         *
         * @param idx The index of the control point to be set (0..E)
         *
         * @return A const reference to the idx-th control point
         */
        const T& ControlPoint(unsigned int idx) const;

        /**
         * Evaluates the position on the bézier curve for the
         * interpolation parameter t based on the algorithm of de Casteljau.
         *
         * @param t The interpolation parameter (0..1); Results for values
         *          outside this range are undefined.
         *
         * @return The position on the bézier curve
         */
        VISLIB_FORCEINLINE T Evaluate(float t) const {
            T rv;
            return this->CalcPoint(rv, t);
        }

        /**
         * Sets the idx-th control point
         *
         * @param idx The index of the control point to be set (0..E)
         * @param p The new value for the control point
         */
        VISLIB_FORCEINLINE void SetControlPoint(unsigned int idx, T& p) {
            this->ControlPoint(idx) = p;
        }

        /**
         * Evaluates the position on the bézier curve for the
         * interpolation parameter t based on the algorithm of de Casteljau.
         *
         * @param t The interpolation parameter (0..1); Results for values
         *          outside this range are undefined.
         *
         * @return The position on the bézier curve
         */
        VISLIB_FORCEINLINE T operator()(float t) const {
            T rv;
            return this->CalcPoint(rv, t);
        }

        /**
         * Accesses the idx-th control point
         *
         * @param idx The index of the control point to be set (0..E)
         *
         * @return A reference to the idx-th control point
         */
        VISLIB_FORCEINLINE T& operator[](unsigned int idx) {
            return this->ControlPoint(idx);
        }

        /**
         * Accesses the idx-th control point read-only
         *
         * @param idx The index of the control point to be set (0..E)
         *
         * @return A const reference to the idx-th control point
         */
        VISLIB_FORCEINLINE const T& operator[](unsigned int idx) const {
            return this->ControlPoint(idx);
        }

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand
         *
         * @return 'true' if 'this' and 'rhs' are equal, 'false' if not.
         */
        bool operator==(const BezierCurve<T, E>& rhs) const;

    private:

        /** control points */
        T cp[E + 1];

    };


    /*
     * BezierCurve<T, E>::BezierCurve
     */
    template<class T, unsigned int E>
    BezierCurve<T, E>::BezierCurve(void) {
        // intentionally empty
    }


    /*
     * BezierCurve<T, E>::~BezierCurve
     */
    template<class T, unsigned int E>
    BezierCurve<T, E>::~BezierCurve(void) {
        // intentionally empty
    }


    /*
     * BezierCurve<T, E>::CalcPoint
     */
    template<class T, unsigned int E>
    T& BezierCurve<T, E>::CalcPoint(T& outPt, float t) const {
        // algorithm of de casteljau
        T bp[E];
        unsigned int cnt = E;
        // first iteration on the control points
        for (unsigned int i = 0; i < cnt; i++) {
            bp[i] = this->cp[i].Interpolate(this->cp[i + 1], t);
        }
        // subsequent iterations on temp array
        while (cnt > 1) {
            cnt--;
            for (unsigned int i = 0; i < cnt; i++) {
                bp[i] = bp[i].Interpolate(bp[i + 1], t);
            }
        }
        // final result
        outPt = bp[0];
        return outPt;
    }


    /*
     * BezierCurve<T, E>::CalcTangent
     */
    template<class T, unsigned int E>
    template<class Tp>
    Tp& BezierCurve<T, E>::CalcTangent(Tp& outVec, float t) const {
        // algorithm of de casteljau
        T bp[E];
        unsigned int cnt = E;
        // first iteration on the control points
        for (unsigned int i = 0; i < cnt; i++) {
            bp[i] = this->cp[i].Interpolate(this->cp[i + 1], t);
        }
        // subsequent iterations on temp array, 
        while (cnt > 2) {
            cnt--;
            for (unsigned int i = 0; i < cnt; i++) {
                bp[i] = bp[i].Interpolate(bp[i + 1], t);
            }
        }
        // final result
        outVec = bp[1] - bp[0];

        if (outVec.IsNull()) {
            // failed: try it again:
            outVec.SetNull();
            for (unsigned int i = 0; i < E; i++) {
                outVec += (this->cp[i + 1] - this->cp[i]);
            }
            if (outVec.IsNull()) {
                // still could be a loop
                bool allEqual = true;
                for (unsigned int i = 1; i < E; i++) {
                    if (this->cp[0] != this->cp[i]) allEqual = false;
                }
                if (allEqual) return outVec; // there is no hope
            }

            Tp vec;
            cnt = E;
            for (unsigned int i = 0; i < cnt; i++) {
                bp[i] = this->cp[i].Interpolate(this->cp[i + 1], t);
            }

            while (cnt > 2) {
                vec.SetNull();
                for (unsigned int i = 1; i < cnt; i++) {
                    vec += bp[i] - bp[i - 1];
                }
                if (vec.IsNull()) {
                    // still could be a loop
                    bool allEqual = true;
                    for (unsigned int i = 1; i < cnt; i++) {
                        if (bp[0] != bp[i]) allEqual = false;
                    }
                    if (allEqual) return outVec; // there is no hope for any better value
                } else {
                    outVec = vec; // this is a better solution
                }

                cnt--;
                for (unsigned int i = 0; i < cnt; i++) {
                    bp[i] = bp[i].Interpolate(bp[i + 1], t);
                }
            }

            // bp[1] - bp[0] is now Null, so any previous calced vector is better
        }

        return outVec;
    }


    /*
     * BezierCurve<T, E>::ControlPoint
     */
    template<class T, unsigned int E>
    T& BezierCurve<T, E>::ControlPoint(unsigned int idx) {
        if (idx > E) {
            throw vislib::OutOfRangeException(idx, 0, E, __FILE__, __LINE__);
        }
        return this->cp[idx];
    }


    /*
     * BezierCurve<T, E>::ControlPoint
     */
    template<class T, unsigned int E>
    const T& BezierCurve<T, E>::ControlPoint(unsigned int idx) const {
        if (idx > E) {
            throw vislib::OutOfRangeException(idx, 0, E, __FILE__, __LINE__);
        }
        return this->cp[idx];
    }


    /*
     * BezierCurve<T, E>::operator==
     */
    template<class T, unsigned int E>
    bool BezierCurve<T, E>::operator==(const BezierCurve<T, E>& rhs) const {
        for (SIZE_T i = 0; i <= E; i++) {
            if (this->cp[i] != rhs.cp[i]) {
                return false;
            }
        }
        return true;
    }


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_BEZIERCURVE_H_INCLUDED */

