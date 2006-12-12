/*
 * AbstractMatrix.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTMATRIX_H_INCLUDED
#define VISLIB_ABSTRACTMATRIX_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/AbstractMatrixImpl.h"


namespace vislib {
namespace math {


    /**
     * All matrix implementations must inherit from this class. Do not inherit
     * directly from AbstractMatrixImpl as the abstraction layer of 
     * AbstractMatrix ensures that the implementation can work correctly and
     * instantiate derived classes.
     */
    template<class T, unsigned int D, MatrixLayout L, class S>
    class AbstractMatrix 
            : public AbstractMatrixImpl<T, D, L, S, AbstractMatrix> {

    public:

        /** Dtor. */
        ~AbstractMatrix(void);

        /**
         * Assignment operator.
         *
         * This operation does <b>not</b> create aliases.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline AbstractMatrix& operator =(const AbstractMatrix& rhs) {
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
        inline AbstractMatrix& operator =(
                const AbstractMatrix<Tp, Dp, Lp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** A typedef for the super class. */
        typedef AbstractMatrixImpl<T, D, L, S, vislib::math::AbstractMatrix>
            Super;

        /**
         * Disallow instances of this class. 
         */
        inline AbstractMatrix(void) : Super() {}

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, MatrixLayout Lf1, class Sf1,
            template<class Tf2, unsigned int Df2, MatrixLayout Lf2, class Sf2> 
            class Cf>
            friend class AbstractMatrixImpl;

    };


    /*
     * vislib::math::AbstractMatrix<T, D, L, S>::~AbstractMatrix
     */
    template<class T, unsigned int D, MatrixLayout L, class S>
    AbstractMatrix<T, D, L, S>::~AbstractMatrix(void) {
    }
    
} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_ABSTRACTMATRIX_H_INCLUDED */

