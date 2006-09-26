/*
 * ShallowVector3D.h  14.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHALLOWVECTOR3D_H_INCLUDED
#define VISLIB_SHALLOWVECTOR3D_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/AbstractVector3D.h"
#include "vislib/ShallowVector.h"


namespace vislib {
namespace math {

    /**
     * Specialisation for a three-dimensional vector. See Vector for additional
     * remarks.
     */
    template<class T> class ShallowVector3D : public AbstractVector3D<T, T *> {

    public:

        /** A typedef for the super class. */
        typedef AbstractVector3D<T, T *> Super;

        /**
         * Create a new vector initialised using 'components' as data. The
         * vector will operate on these data. The caller is responsible that
         * the memory designated by 'components' lives as long as the object
         * and all its aliases exist.
         *
         * @param components The initial vector memory. This must not be a NULL
         *                   pointer.
         */
        explicit inline ShallowVector3D(T *components) {
            ASSERT(components != NULL);
            this->components = components;
        }

        /**
         * Clone 'rhs'. This operation will create an alias of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline ShallowVector3D(const ShallowVector3D& rhs) {
            this->components = rhs.components;
        }

        /** Dtor. */
        ~ShallowVector3D(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline ShallowVector3D& operator =(const ShallowVector3D& rhs) {
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
        template<class Tp, unsigned int Dp,class Sp>
        inline ShallowVector3D& operator =(
                const AbstractVector<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    private:

        /** 
         * Forbidden ctor. A default ctor would be inherently unsafe for
         * shallow vectors.
         */
        inline ShallowVector3D(void) {}
    };


    /*
     * vislib::math::ShallowVector3D<T>::~ShallowVector3D
     */
    template<class T>
    ShallowVector3D<T>::~ShallowVector3D(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_SHALLOWVECTOR3D_H_INCLUDED */
