/*
 * ShallowVector.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHALLOWVECTOR_H_INCLUDED
#define VISLIB_SHALLOWVECTOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractVector.h"


namespace vislib {
namespace math {

    /**
     * This is the implementation of an AbstractVector that uses its a 
     * user-provided memory block for its data. The user-provided memory
     * block is interpreted as vector components. Note, that this class has
     * the inherent problem of producing aliases. You should only use it, if you
     * really require vector operations on memory external to the vector
     * objects.
     *
     * See documentation of AbstractVector for further information about the 
     * vector classes.
     */
    template<class T, unsigned int D> 
    class ShallowVector : public AbstractVector<T, D, T *> {

    public:

        /**
         * Create a new vector initialised using 'components' as data. The
         * vector will operate on these data. The caller is responsible that
         * the memory designated by 'components' lives as long as the object
         * and all its aliases exist.
         *
         * @param components The initial vector memory. This must not be a NULL
         *                   pointer.
         */
        explicit inline ShallowVector(T *components) {
            ASSERT(components != NULL);
            this->components = components;
        }

        /**
         * Clone 'rhs'. This operation will create an alias of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline ShallowVector(const ShallowVector& rhs) {
            this->components = rhs.components;
        }

        /** Dtor. */
        ~ShallowVector(void);

        /**
         * Set a new component pointer. The vector uses from this point the
         * memory designated by 'components' instead of the memory passed to
         * the ctor or in previous calls to this method for its operations.
         * The caller remains owner of the memory designated by 'components'
         * and must ensure that it lives as long as this object and all its
         * aliases live.
         *
         * @param components The new vector component memory. This must not be
         *                   a NULL vector.
         */
//#pragma deprecated(SetComponents)
//        inline void SetComponents(T *components) {
//            ASSERT(components != NULL);
//            this->components = components;
//        }

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
            AbstractVector<T, D, T *>::operator =(rhs);
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
        inline ShallowVector& operator =(
                const AbstractVector<Tp, Dp, Sp>& rhs) {
            AbstractVector<T, D, T *>::operator =(rhs);
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
     * ShallowVector<T, D>::~ShallowVector
     */
    template<class T, unsigned int D>
    ShallowVector<T, D>::~ShallowVector(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHALLOWVECTOR_H_INCLUDED */
