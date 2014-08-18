/*
 * ShallowMatrix.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHALLOWMATRIX_H_INCLUDED
#define VISLIB_SHALLOWMATRIX_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractMatrix.h"
#include "vislib/assert.h"


namespace vislib {
namespace math {


    /**
     * This specialisation implements a "shallow matrix", i. e. a matrix object
     * that does not own the memory of its components. You can use this class
     * e. g. to reinterpret memory received over a network as a matrix without
     * copying the data. You are responsible for providing the data memory as
     * long as the object lives. If you just want to have a easy-to-use matrix 
     * that manages its memory itself, consider using the Matrix class.
     */
    template<class T, unsigned int D, MatrixLayout L> 
    class ShallowMatrix : public AbstractMatrix<T, D, L, T *> {

    public:

        /**
         * Create a new matrix initialised using 'components' as data. The
         * matrix will operate on these data. The caller is responsible that
         * the memory designated by 'components' lives as long as the object
         * and all its aliases exist.
         *
         * @param components The initial matrix memory. This must not be a NULL
         *                   pointer.
         */
        explicit inline ShallowMatrix(T *components) {
            ASSERT(components != NULL);
            this->components = components;
        }

        /**
         * Clone 'rhs'. This operation will create an alias of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline ShallowMatrix(const ShallowMatrix& rhs) {
            this->components = rhs.components;
        }

        /** Dtor. */
        ~ShallowMatrix(void);

        /**
         * Assignment operator.
         *
         * This operation does <b>not</b> create aliases.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline ShallowMatrix& operator =(const ShallowMatrix& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Assigment for arbitrary matrices. A valid static_cast between T and 
         * Tp is a precondition for instantiating this template.
         *
         * This operation does <b>not</b> create aliases. 
         *
         * If the two operands have different dimensions, the behaviour is as 
         * follows: If the left hand side operand has lower dimension, the 
         * highest (Dp - D) dimensions are discarded. If the left hand side
         * operand has higher dimension, the missing dimensions are filled with 
         * parts of the identity matrix.
         *
         * Subclasses must ensure that sufficient memory for the 'coordinates'
         * member has been allocated before calling this operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        template<class Tp, unsigned int Dp, MatrixLayout Lp, class Sp>
        inline ShallowMatrix& operator =(
                const AbstractMatrix<Tp, Dp, Lp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }


    protected:

        /** A typedef for the super class. */
        typedef AbstractMatrix<T, D, L, T *> Super;

        /** 
         * Forbidden default ctor. 
         */
        inline ShallowMatrix(void) {}

    };

    /*
     * vislib::math::ShallowMatrix<T, D, L>::~ShallowMatrix
     */
    template<class T, unsigned int D, MatrixLayout L>
    ShallowMatrix<T, D, L>::~ShallowMatrix(void) {
    }
    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHALLOWMATRIX_H_INCLUDED */

