/*
 * AbstractVector2D.h  19.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTVECTOR2D_H_INCLUDED
#define VISLIB_ABSTRACTVECTOR2D_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/Vector.h"


namespace vislib {
namespace math {

    /**
     * Specialisation for a two-dimensional vector. See AbstractVector for 
     * additional remarks.
     */
    template<class T, class E, class S> 
    class AbstractVector2D : public AbstractVector<T, 2, E, S> {

    public:

        /** Dtor. */
        ~AbstractVector2D(void);

        /**
         * Answer the x-component of the vector.
         *
         * @return The x-component of the vector.
         */
        inline const T& GetX(void) const {
            return this->components[0];
        }

        /**
         * Answer the y-component of the vector.
         *
         * @return The y-component of the vector.
         */
        inline const T& GetY(void) const {
            return this->components[1];
        }

        /**
         * Set the x-component of the vector.
         *
         * @param x The new x-component.
         */
        inline void SetX(const T& x) {
            this->components[0] = x;
        }

        /**
         * Set the y-component of the vector.
         *
         * @param y The new y-component.
         */
        inline void SetY(const T& y) {
            this->components[1] = y;
        }

        /**
         * Answer the x-component of the vector.
         *
         * @return The x-component of the vector.
         */
        inline const T& X(void) const {
            return this->components[0];
        }

        /**
         * Answer the y-component of the vector.
         *
         * @return The y-component of the vector.
         */
        inline const T& Y(void) const {
            return this->components[1];
        }

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline AbstractVector2D& operator =(const AbstractVector2D& rhs) {
            AbstractVector<T, 2, E, S>::operator =(rhs);
            return *this;
        }

        /**
         * Assigment for arbitrary vectors. A valid static_cast between T and Tp
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
         * Subclasses must ensure that sufficient memory for the 'components' 
         * member has been allocated before calling this operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        template<class Tp, unsigned int Dp, class Ep, class Sp>
        inline AbstractVector2D& operator =(
                const AbstractVector<Tp, Dp, Ep, Sp>& rhs) {
            AbstractVector<T, 2, E, S>::operator =(rhs);
            return *this;
        }

    protected:

        /**
         * Disallow instances of this class. This ctor does nothing!
         */
        inline AbstractVector2D(void) {};

    };


    /*
     * vislib::math::AbstractVector2D<T, E, S>::~AbstractVector2D
     */
    template<class T, class E, class S>
    AbstractVector2D<T, E, S>::~AbstractVector2D(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_ABSTRACTVECTOR2D_H_INCLUDED */
