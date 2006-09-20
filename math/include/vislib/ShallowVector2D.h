/*
 * ShallowVector2D.h  14.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHALLOWVECTOR2D_H_INCLUDED
#define VISLIB_SHALLOWVECTOR2D_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/AbstractVector2D.h"
#include "vislib/ShallowVector.h"


namespace vislib {
namespace math {

    /**
     * Specialisation for a two-dimensional vector. See Vector for additional
     * remarks.
     */
    template<class T> 
    class ShallowVector2D : public AbstractVector2D<T, T *>, 
            public ShallowVector<T, 2> {

    public:

        /** A typedef for the super class having the storage related info. */
        typedef ShallowVector<T, 2> Super;

        /**
         * Create a new vector initialised using 'components' as data. The
         * vector will operate on these data. The caller is responsible that
         * the memory designated by 'components' lives as long as the object
         * and all its aliases exist.
         *
         * @param components The initial vector memory. This must not be a NULL
         *                   pointer.
         */
        explicit inline ShallowVector2D(T *components) : Super(components) {}

        /**
         * Clone 'rhs'. This operation will create an alias of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline ShallowVector2D(const ShallowVector2D& rhs) : Super(rhs) {}

        /** Dtor. */
        virtual ~ShallowVector2D(void);

        /**
         * Assignment.
         *
         * This operation does <b>not</b> create aliases. 
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline ShallowVector2D& operator =(const ShallowVector2D& rhs) {
            Super::operator =(rhs);
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
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        template<class Tp, unsigned int Dp, class Sp>
        inline ShallowVector2D& operator =(
                const AbstractVector<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    private:

        /** 
         * Forbidden ctor. A default ctor would be inherently unsafe for
         * shallow vectors.
         */
        inline ShallowVector2D(void) {}
    };


    /*
     * vislib::math::ShallowVector2D<T>::~ShallowVector2D
     */
    template<class T>
    ShallowVector2D<T>::~ShallowVector2D(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_SHALLOWVECTOR2D_H_INCLUDED */
