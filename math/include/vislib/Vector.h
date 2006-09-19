/*
 * Vector.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_VECTOR_H_INCLUDED
#define VISLIB_VECTOR_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/AbstractVector.h"


namespace vislib {
namespace math {

    /**

     */
    template<class T, unsigned int D, class E = EqualFunc<T> > 
    class Vector : virtual public AbstractVector<T, D, E, T[D]> {

    public:

        /**
         * Create a null vector.
         */
        Vector(void);

        /**
         * Create a new vector initialised with 'components'. 'components' must
         * not be a NULL pointer. 
         *
         * @param components The initial vector components.
         */
        explicit inline Vector(const T *components) {
            ASSERT(components != NULL);
            ::memcpy(this->components, components, D * sizeof(T));
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Vector(const Vector& rhs) {
            ::memcpy(this->components, rhs.components, D * sizeof(T));
        }

        /**
         * Create a copy of 'vector'. This ctor allows for arbitrary vector to
         * vector conversions.
         *
         * @param rhs The vector to be cloned.
         */
        template<class Tp, unsigned int Dp, class Ep, class Sp>
        Vector(const AbstractVector<Tp, Dp, Ep, Sp>& rhs);

        /** Dtor. */
        ~Vector(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline Vector& operator =(const Vector& rhs) {
            return AbstractVector::operator =(rhs);
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
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        template<class Tp, unsigned int Dp, class Ep, class Sp>
        inline Vector& operator =(const AbstractVector<Tp, Dp, Ep, Sp>& rhs) {
            AbstractVector::operator =(rhs);
            return *this;
        }
    };


    /*
     * vislib::math::Vector<T, D, E, S>::Vector
     */
    template<class T, unsigned int D, class E>
    Vector<T, D, E>::Vector(void) {
        for (unsigned int d = 0; d < D; d++) {
            this->components[d] = static_cast<T>(0);
        }
    }


    /*
     * vislib::math::Vector<T, D, E, S>::Vector
     */
    template<class T, unsigned int D, class E>
    template<class Tp, unsigned int Dp, class Ep, class Sp>
    Vector<T, D, E>::Vector(const AbstractVector<Tp, Dp, Ep, Sp>& rhs) {
        for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
            this->components[d] = static_cast<T>(rhs[d]);
        }
        for (unsigned int d = Dp; d < D; d++) {
            this->components[d] = static_cast<T>(0);
        }
    }


    /*
     * vislib::math::Vector<T, D, E>::~Vector
     */
    template<class T, unsigned int D, class E>
    Vector<T, D, E>::~Vector(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_VECTOR_H_INCLUDED */
