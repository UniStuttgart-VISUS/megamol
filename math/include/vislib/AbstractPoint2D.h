/*
 * AbstractPoint2D.h  19.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTPOINT2D_H_INCLUDED
#define VISLIB_ABSTRACTPOINT2D_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/AbstractPoint.h"


namespace vislib {
namespace math {

    /**
     * Specialisation for a two-dimensional point. See AbstractPoint for 
     * additional remarks.
     */
    template<class T, class E, class S> 
    class AbstractPoint2D : virtual public AbstractPoint<T, 2, E, S> {

    public:

        /** Dtor. */
        ~AbstractPoint2D(void);

        /**
         * Answer the x-coordinate of the point.
         *
         * @return The x-coordinate of the point.
         */
        inline const T& GetX(void) const {
            return this->coordinates[0];
        }

        /**
         * Answer the y-coordinate of the point.
         *
         * @return The y-coordinate of the point.
         */
        inline const T& GetY(void) const {
            return this->coordinates[1];
        }

        /**
         * Set the x-coordinate of the point.
         *
         * @param x The new x-coordinate.
         */
        inline void SetX(const T& x) {
            this->coordinates[0] = x;
        }

        /**
         * Set the y-coordinate of the point.
         *
         * @param y The new y-coordinate.
         */
        inline void SetY(const T& y) {
            this->coordinates[1] = y;
        }

        /**
         * Answer the x-coordinate of the point.
         *
         * @return The x-coordinate of the point.
         */
        inline const T& X(void) const {
            return this->coordinates[0];
        }

        /**
         * Answer the y-component of the point.
         *
         * @return The y-component of the point.
         */
        inline const T& Y(void) const {
            return this->coordinates[1];
        }

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline AbstractPoint2D& operator =(const AbstractPoint2D& rhs) {
            AbstractPoint<T, 2, E, S>::operator =(rhs);
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
        template<class Tp, unsigned int Dp, class Ep, class Sp>
        inline AbstractPoint2D& operator =(
                const AbstractPoint<Tp, Dp, Ep, Sp>& rhs) {
            AbstractPoint<T, 2, E, S>::operator =(rhs);
            return *this;
        }

    protected:

        /**
         * Disallow instances of this class.
         */
        inline AbstractPoint2D(void) : AbstractPoint<T, 2, E, S>() {};

    };


    /*
     * vislib::math::AbstractPoint2D<T, E, S>::~AbstractPoint2D
     */
    template<class T, class E, class S>
    AbstractPoint2D<T, E, S>::~AbstractPoint2D(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_ABSTRACTPOINT2D_H_INCLUDED */
