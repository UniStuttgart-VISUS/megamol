/*
 * ShallowVector.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHALLOWVECTOR_H_INCLUDED
#define VISLIB_SHALLOWVECTOR_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/AbstractVector.h"


namespace vislib {
namespace math {

    /**

     */
    template<class T, unsigned int D, class E = EqualFunc<T> > 
    class ShallowVector : public AbstractVector<T, D, E, T *> {

    public:

        /**
         * Create a new vector initialised using 'components' as data. The
         * vector will operate on these data.
         *
         * @param components The initial vector components.
         */
        explicit inline ShallowVector(const T *components) 
                : components(components) {
            ASSERT(components != NULL);
        }

        /**
         * Clone 'rhs'. This operation will create an alias of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline ShallowVector(const ShallowVector& rhs) 
            : components(rhs.components) {}

        /** Dtor. */
        ~ShallowVector(void);

        /**
         * Assignment.
         *
         * This operation does <b>not</b> create aliases. 
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline ShallowVector& operator =(const ShallowVector& rhs) {
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
        inline ShallowVector& operator =(
                const AbstractVector<Tp, Dp, Ep, Sp>& rhs) {
            AbstractVector::operator =(rhs);
            return *this;
        }

    private:

        /** 
         * Forbidden ctor. A default ctor would be inherently unsafe for
         * shallow vectors.
         */
        inline ShallowVector(void) {}
    };


    /*
     * ShallowVector<T, D, E>::~ShallowVector
     */
    template<class T, unsigned int D, class E>
    ShallowVector<T, D, E>::~ShallowVector(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_SHALLOWVECTOR_H_INCLUDED */
