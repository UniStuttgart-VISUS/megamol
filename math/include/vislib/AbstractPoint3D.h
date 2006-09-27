/*
 * AbstractPoint3D.h  19.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTPOINT3D_H_INCLUDED
#define VISLIB_ABSTRACTPOINT3D_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/AbstractPoint.h"


namespace vislib {
namespace math {

    /**
     * Specialisation for a three-dimensional point. See AbstractPoint for 
     * additional remarks.
     */
    template<class T, class S> 
    class AbstractPoint3D : public AbstractPoint<T, 3, S> {

    public:

        /** Dtor. */
        ~AbstractPoint3D(void);

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
         * Answer the z-coordinate of the point.
         *
         * @return The z-coordinate of the point.
         */
        inline const T& GetZ(void) const {
            return this->coordinates[2];
        }

        /**
         * Set the coordinates ot the point.
         *
         * @param x The x-coordinate of the point.
         * @param y The y-coordinate of the point.
         * @param z The z-coordinate of the point.
         */
        inline void Set(const T& x, const T& y, const T& z) {
            this->coordinates[0] = x;
            this->coordinates[1] = y;
            this->coordinates[2] = z;
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
         * Set the z-coordinate of the point.
         *
         * @param z The new z-coordinate.
         */
        inline void SetZ(const T& z) {
            this->coordinates[2] = z;
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
         * Answer the y-coordinate of the point.
         *
         * @return The y-coordinate of the point.
         */
        inline const T& Y(void) const {
            return this->coordinates[1];
        }

        /**
         * Answer the z-coordinate of the point.
         *
         * @return The z-coordinate of the point.
         */
        inline const T& Z(void) const {
            return this->coordinates[2];
        }

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline AbstractPoint3D& operator =(const AbstractPoint3D& rhs) {
            AbstractPoint<T, 3, S>::operator =(rhs);
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
        inline AbstractPoint3D& operator =(
                const AbstractPoint<Tp, Dp, Sp>& rhs) {
            AbstractPoint<T, 3, S>::operator =(rhs);
            return *this;
        }

    protected:

        /**
         * Disallow instances of this class. This ctor does nothing!
         */
        inline AbstractPoint3D(void) : AbstractPoint<T, 3, S>() {};

    };


    /*
     * vislib::math::AbstractPoint3D<T, S>::~AbstractPoint3D
     */
    template<class T, class S>
    AbstractPoint3D<T, S>::~AbstractPoint3D(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_ABSTRACTPOINT3D_H_INCLUDED */
