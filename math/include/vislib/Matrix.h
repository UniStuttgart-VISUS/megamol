/*
 * Matrix.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MATRIX_H_INCLUDED
#define VISLIB_MATRIX_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractMatrix.h"
#include "vislib/assert.h"


namespace vislib {
namespace math {


    /**
     * This class implements matrices that use their own memory on the stack
     * to store their components.
     */
    template<class T, unsigned int D, MatrixLayout L> 
    class Matrix : public AbstractMatrix<T, D, L, T[D * D]> {

    public:

        /** 
         * Create the identity matrix.
         */
        inline Matrix(void) : Super() {
            this->SetIdentity();
        }

        /**
         * Create which has the same value for all components.
         *
         * @param value The initial value of all components.
         */
        Matrix(const T& value);

        /**
         * Create a matrix using the specified components.
         *
         * @param components (D * D) components of the matrix. This must not be
         *                   NULL and according to the matrix layout L.
         */
        inline Matrix(const T *components) : Super() {
            ASSERT(components != NULL);
            ::memcpy(this->components, component, CNT_COMPONENTS * sizeof(T));
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Matrix(const Matrix& rhs) : Super() {
            ::memcpy(this->components, rhs.component, 
                CNT_COMPONENTS * sizeof(T));
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        template<class Tp, unsigned int Dp, MatrixLayout Lp, class Sp>
        inline Matrix(const AbstractMatrix<Tp, Dp, Lp, Sp>& rhs) : Super() {
            this->assign(rhs);
        }

        /** Dtor. */
        ~Matrix(void);

        /**
         * Assignment operator.
         *
         * This operation does <b>not</b> create aliases.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline Matrix& operator =(const Matrix& rhs) {
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
        inline Matrix& operator =(const AbstractMatrix<Tp, Dp, Lp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** A typedef for the super class. */
        typedef AbstractMatrix<T, D, L, T[D * D]> Super;

    };


    /*
     * vislib::math::Matrix<T, D, L>::Matrix
     */
    template<class T, unsigned int D, MatrixLayout L>
    Matrix<T, D, L>::Matrix(const T& value) {
        for (unsigned int i = 0; i < D; i++) {
            this->components[i] = value;
        }
    }


    /*
     * vislib::math::Matrix<T, D, L>::~Matrix
     */
    template<class T, unsigned int D, MatrixLayout L>
    Matrix<T, D, L>::~Matrix(void) {
    }
    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_MATRIX_H_INCLUDED */

