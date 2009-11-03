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
#include "vislib/OutOfRangeException.h"
#include "vislib/Point.h"
#include "vislib/Vector.h"


namespace vislib {
namespace math {


    /**
     * Stores and evaluates a bézier curve of degree E. The curve is defined
     * by E + 1 points (type T and dimension D) forming the control polygon.
     *
     * Template parameters:
     *  T type (float, Double)
     *  D Dimension
     *  E Degree
     */
    template <class T, unsigned int D, unsigned int E>
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
         * @param t The interpolation parameter (0..1); Results for values
         *          outside this range are undefined.
         *
         * @return The position on the bézier curve
         */
        Point<T, D> CalcPoint(float t) const;

        /**
         * Calculates the tangent vector along the curve at the point defined
         * by the interpolation value t.
         *
         * @param t The interpolation parameter (0..1); Results for values
         *          outside this range are undefined.
         *
         * @return The tangent vector along the bézier curve
         */
        Vector<T, D> CalcTangent(float t) const;

        /**
         * Accesses the idx-th control point
         *
         * @param idx The index of the control point to be set (0..E)
         *
         * @return A reference to the idx-th control point
         */
        Point<T, D>& ControlPoint(unsigned int idx);

        /**
         * Accesses the idx-th control point read-only
         *
         * @param idx The index of the control point to be set (0..E)
         *
         * @return A const reference to the idx-th control point
         */
        const Point<T, D>& ControlPoint(unsigned int idx) const;

        /**
         * Evaluates the position on the bézier curve for the
         * interpolation parameter t based on the algorithm of de Casteljau.
         *
         * @param t The interpolation parameter (0..1); Results for values
         *          outside this range are undefined.
         *
         * @return The position on the bézier curve
         */
        VISLIB_FORCEINLINE Point<T, D> Evaluate(float t) const {
            return this->CalcPoint(t);
        }

        /**
         * Sets the idx-th control point
         *
         * @param idx The index of the control point to be set (0..E)
         * @param p The new value for the control point
         */
        VISLIB_FORCEINLINE void SetControlPoint(unsigned int idx, const Point<T, D>& p) {
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
        VISLIB_FORCEINLINE Point<T, D> operator()(float t) const {
            return this->CalcPoint(t);
        }

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand
         *
         * @return 'true' if 'this' and 'rhs' are equal, 'false' if not.
         */
        bool operator==(const BezierCurve<T, D, E>& rhs) const;

    private:

        /** control points */
        Point<T, D> cp[E + 1];

    };


    /*
     * BezierCurve<T, D, E>::BezierCurve
     */
    template<class T, unsigned int D, unsigned int E>
    BezierCurve<T, D, E>::BezierCurve(void) {
        // intentionally empty
    }


    /*
     * BezierCurve<T, D, E>::~BezierCurve
     */
    template<class T, unsigned int D, unsigned int E>
    BezierCurve<T, D, E>::~BezierCurve(void) {
        // intentionally empty
    }


    /*
     * BezierCurve<T, D, E>::CalcPoint
     */
    template<class T, unsigned int D, unsigned int E>
    Point<T, D> BezierCurve<T, D, E>::CalcPoint(float t) const {
        // algorithm of de casteljau
        Point<T, D> bp[E];
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
        return bp[0];
    }


    /*
     * BezierCurve<T, D, E>::CalcTangent
     */
    template<class T, unsigned int D, unsigned int E>
    Vector<T, D> BezierCurve<T, D, E>::CalcTangent(float t) const {
        // algorithm of de casteljau
        Point<T, D> bp[E];
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

        return bp[1] - bp[0];
    }


    /*
     * BezierCurve<T, D, E>::ControlPoint
     */
    template<class T, unsigned int D, unsigned int E>
    Point<T, D>& BezierCurve<T, D, E>::ControlPoint(unsigned int idx) {
        if (idx > E) {
            throw vislib::OutOfRangeException(idx, 0, E, __FILE__, __LINE__);
        }
        return this->cp[idx];
    }


    /*
     * BezierCurve<T, D, E>::ControlPoint
     */
    template<class T, unsigned int D, unsigned int E>
    const Point<T, D>& BezierCurve<T, D, E>::ControlPoint(
            unsigned int idx) const {
        if (idx > E) {
            throw vislib::OutOfRangeException(idx, 0, E, __FILE__, __LINE__);
        }
        return this->cp[idx];
    }


    /*
     * BezierCurve<T, D, E>::ControlPoint
     */
    template<class T, unsigned int D, unsigned int E>
    bool BezierCurve<T, D, E>::operator==(const BezierCurve<T, D, E>& rhs) const {
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

