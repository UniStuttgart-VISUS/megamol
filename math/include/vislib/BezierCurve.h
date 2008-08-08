/*
 * BezierCurve.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
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


#include "vislib/Point.h"
#include "vislib/OutOfRangeException.h"


namespace vislib {
namespace math {


    /**
     * TODO: comment class
     *
     * Template parameters:
     *  T type (float, Double)
     *  D Dimension
     *  E Degree
     */
    template <class T, unsigned int D, unsigned int E>
    class BezierCurve {
    public:

        /** Ctor. */
        BezierCurve(void);

        /** Dtor. */
        ~BezierCurve(void);

        /**
         * TODO: Document
         */
        void SetControlPoint(unsigned int idx, const Point<T, D>& p);

        /**
         * TODO: Document
         */
        Point<T, D> Evaluate(float t);

    private:

        /** control points */
        Point<T, D> cp[E + 1];

    };


    /*
     * BezierCurve<T, D, E>::BezierCurve
     */
    template<class T, unsigned int D, unsigned int E>
    BezierCurve<T, D, E>::BezierCurve(void) {
    }


    /*
     * BezierCurve<T, D, E>::~BezierCurve
     */
    template<class T, unsigned int D, unsigned int E>
    BezierCurve<T, D, E>::~BezierCurve(void) {
    }


    /*
     * BezierCurve<T, D, E>::SetControlPoint
     */
    template<class T, unsigned int D, unsigned int E>
    void BezierCurve<T, D, E>::SetControlPoint(unsigned int idx, const Point<T, D>& p) {
        if (idx > E) {
            throw vislib::OutOfRangeException(idx, 0, E, __FILE__, __LINE__);
        }
        this->cp[idx] = p;
    }


    /*
     * BezierCurve<T, D, E>::Evaluate
     */
    template<class T, unsigned int D, unsigned int E>
    Point<T, D> BezierCurve<T, D, E>::Evaluate(float t) {
        Point<T, D> bp[E];
        unsigned int cnt = E;

        for (unsigned int i = 0; i < cnt; i++) {
            bp[i] = this->cp[i].Interpolate(this->cp[i + 1], t);
        }
        while (cnt > 1) {
            cnt--;
            for (unsigned int i = 0; i < cnt; i++) {
                bp[i] = bp[i].Interpolate(bp[i + 1], t);
            }
        }

        return bp[0];
    }


    ///*
    // * BezierCurve<T, D, 0>::Evaluate
    // */
    //template<class T, unsigned int D, 0>
    //Point<T, D> BezierCurve<T, D, 0>::Evaluate(float t) {
    //    return this->cp[0];
    //}


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_BEZIERCURVE_H_INCLUDED */

