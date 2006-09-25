/*
 * Point.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_POINT_H_INCLUDED
#define VISLIB_POINT_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/AbstractPoint.h"


namespace vislib {
namespace math {

    /**
     * This is the implementation of an AbstractPoint that uses its own memory 
     * in a statically allocated array of dimension D. Usually, you want to use
     * this point class or derived classes.
     *
     * See documentation of AbstractPoint for further information about the 
     * vector classes.
     */
    template<class T, unsigned int D> 
    class Point : virtual public AbstractPoint<T, D, T[D]> {

    public:

        /**
         * Create a point in the coordinate origin.
         */
        Point(void);

        /**
         * Create a new point initialised with 'coordinates'. 'coordinates' must
         * not be a NULL pointer. 
         *
         * @param coordinates The initial coordinates of the point.
         */
        explicit inline Point(const T *coordinates) {
            ASSERT(coordinates != NULL);
            ::memcpy(this->coordinates, coordinates, D * sizeof(T));
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Point(const Point& rhs) {
            ::memcpy(this->coordinates, rhs.coordinates, D * sizeof(T));
        }

        /**
         * Create a copy of 'rhs'. This ctor allows for arbitrary point to
         * point conversions.
         *
         * @param rhs The vector to be cloned.
         */
        template<class Tp, unsigned int Dp, class Sp>
        Point(const AbstractPoint<Tp, Dp, Sp>& rhs);

        /** Dtor. */
        ~Point(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline Point& operator =(const Point& rhs) {
            AbstractPoint<T, D, T[D]>::operator =(rhs);
            return *this;
        }

        /**
         * Assigment for arbitrary points. A valid static_cast between T and Tp
         * is a precondition for instantiating this template.
         *
         * This operation does <b>not</b> create aliases. 
         *
         * If the two operands have different dimensions, the behaviour is as 
         * follows: If the left hand side operand has lower dimension, the 
         * highest (Dp - D) dimensions are discarded. If the left hand side
         * operand has higher dimension, the missing dimensions are filled with 
         * zero components.
         *
         * Subclasses must ensure that sufficient memory for the 'coordinates'
         * member has been allocated before calling this operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        template<class Tp, unsigned int Dp, class Sp>
        inline Point& operator =(const AbstractPoint<Tp, Dp, Sp>& rhs) {
            AbstractPoint<T, D, T[D]>::operator =(rhs);
            return *this;
        }
    };


    /*
     * vislib::math::Point<T, D>::Point
     */
    template<class T, unsigned int D>
    Point<T, D>::Point(void) {
        for (unsigned int d = 0; d < D; d++) {
            this->coordinates[d] = static_cast<T>(0);
        }
    }


    /*
     * vislib::math::Point<T, D>::Point
     */
    template<class T, unsigned int D>
    template<class Tp, unsigned int Dp, class Sp>
    Point<T, D>::Point(const AbstractPoint<Tp, Dp, Sp>& rhs) {
        for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
            this->components[d] = static_cast<T>(rhs[d]);
        }
        for (unsigned int d = Dp; d < D; d++) {
            this->components[d] = static_cast<T>(0);
        }
    }


    /*
     * vislib::math::Point<T, D>::~Point
     */
    template<class T, unsigned int D>
    Point<T, D>::~Point(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_POINT_H_INCLUDED */
