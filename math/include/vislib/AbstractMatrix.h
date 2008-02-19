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
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractMatrixImpl.h"
#include "vislib/Quaternion.h"


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

    private:

        /**
         * Allow AbstractMatrixImpl to assign from itself to the AbstractMatrix
         * subclass. This is required for implementing serveral arithmetic 
         * operations in AbstractMatrixImpl, which must initialise their return
         * value by copying themselves.
         *
         * This ctor is private as it should only be used on deep-storage 
         * instantiations. Shallow storage instantiations MUST NEVER EXPOSE OR 
         * USE this ctor.
         *
         * @param rhs The object to be cloned.
         */
        template<class S1>
        inline AbstractMatrix(const AbstractMatrixImpl<T, D, L, S1, 
                vislib::math::AbstractMatrix>& rhs) : Super() {
            ::memcpy(this->components, rhs.PeekComponents(), 
                Super::CNT_COMPONENTS * sizeof(T));
        }

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


    /**
     * Partial template specialisation for 4x4 matrices.
     */
    template<class T, MatrixLayout L, class S>
    class AbstractMatrix<T, 4, L, S>
            : public AbstractMatrixImpl<T, 4, L, S, AbstractMatrix> {

    public:

        /** Dtor. */
        ~AbstractMatrix(void);

        //T Determinant(void) const;

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

        /**
         * Make this matrix represent the quaterion 'rhs'.
         *
         * @param rhs The quaterion to be converted to a rotation matrix.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        AbstractMatrix& operator =(const AbstractQuaternion<Tp, Sp>& rhs);

    protected:

        /** A typedef for the super class. */
        typedef AbstractMatrixImpl<T, 4, L, S, vislib::math::AbstractMatrix>
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
     * vislib::math::AbstractMatrix<T, 4, L, S>::~AbstractMatrix
     */
    template<class T, MatrixLayout L, class S>
    AbstractMatrix<T, 4, L, S>::~AbstractMatrix(void) {
    }


    ///*
    // * vislib::math::AbstractMatrix<T, 4, L, S>::Determinant
    // */
    //template<class T, MatrixLayout L, class S>
    //T AbstractMatrix<T, 4, L, S>::Determinant(void) const {
    //// Note: Fast, but matrix size is fixed! Uses Laplace and Sarrus.

    //    return (this->elements[0] 
    //        * (this->elements[5] * this->elements[10] * this->elements[15] 
    //        + this->elements[9] * this->elements[14] * this->elements[7] 
    //        + this->elements[13] * this->elements[6] * this->elements[11]
    //        - this->elements[13] * this->elements[10] * this->elements[7]
    //        - this->elements[5] * this->elements[14] * this->elements[11]
    //        - this->elements[9] * this->elements[6] * this->elements[15])

    //        - this->elements[4]
    //        * (this->elements[1] * this->elements[10] * this->elements[15] 
    //        + this->elements[9] * this->elements[14] * this->elements[3] 
    //        + this->elements[13] * this->elements[2] * this->elements[11]
    //        - this->elements[13] * this->elements[10] * this->elements[3]
    //        - this->elements[1] * this->elements[14] * this->elements[11]
    //        - this->elements[9] * this->elements[2] * this->elements[15])

    //        + this->elements[8]
    //        * (this->elements[1] * this->elements[6] * this->elements[15] 
    //        + this->elements[5] * this->elements[14] * this->elements[3] 
    //        + this->elements[13] * this->elements[2] * this->elements[7]
    //        - this->elements[13] * this->elements[6] * this->elements[3]
    //        - this->elements[1] * this->elements[14] * this->elements[7]
    //        - this->elements[5] * this->elements[2] * this->elements[15])

    //        - this->elements[12] 
    //        * (this->elements[1] * this->elements[6] * this->elements[11] 
    //        + this->elements[5] * this->elements[10] * this->elements[3] 
    //        + this->elements[9] * this->elements[2] * this->elements[7]
    //        - this->elements[9] * this->elements[6] * this->elements[3]
    //        - this->elements[1] * this->elements[10] * this->elements[7]
    //        - this->elements[5] * this->elements[2] * this->elements[11]));
    //}


    /*
     * AbstractMatrix<T, 4, L, S>::operator =
     */
    template<class T, MatrixLayout L, class S>
    template<class Tp, class Sp>
    AbstractMatrix<T, 4, L, S>& AbstractMatrix<T, 4, L, S>::operator =(
            const AbstractQuaternion<Tp, Sp>& rhs) {
        Quaternion<T> q(rhs);
        q.Normalise();

        this->components[Super::indexOf(0, 0)] 
            = Sqr(q.W()) + Sqr(q.X()) - Sqr(q.Y()) - Sqr(q.Z());
        this->components[Super::indexOf(0, 1)] 
            = static_cast<T>(2) * (q.X() * q.Y() - q.W() * q.Z());
        this->components[Super::indexOf(0, 2)] 
            = static_cast<T>(2) * (q.W() * q.Y() + q.X() * q.Z()); 
        this->components[Super::indexOf(0, 3)] = static_cast<T>(0);

        this->components[Super::indexOf(1, 0)] 
            = static_cast<T>(2) * (q.W() * q.Z() + q.X() * q.Y());
        this->components[Super::indexOf(1, 1)] 
            = Sqr(q.W()) - Sqr(q.X()) + Sqr(q.Y()) - Sqr(q.Z());
        this->components[Super::indexOf(1, 2)] 
            = static_cast<T>(2) * (q.Y() * q.Z() - q.W() * q.X());
        this->components[Super::indexOf(1, 3)] = static_cast<T>(0);

        this->components[Super::indexOf(2, 0)] 
            = static_cast<T>(2) * (q.X() * q.Z() - q.W() * q.Y());
        this->components[Super::indexOf(2, 1)] 
            = static_cast<T>(2) * (q.W() * q.X() - q.Y() * q.Z());
        this->components[Super::indexOf(2, 2)] 
            = Sqr(q.W()) - Sqr(q.X()) - Sqr(q.Y()) + Sqr(q.Z());
        this->components[Super::indexOf(2, 3)] = static_cast<T>(0);

        this->components[Super::indexOf(3, 0)] = static_cast<T>(0);
        this->components[Super::indexOf(3, 1)] = static_cast<T>(0);
        this->components[Super::indexOf(3, 2)] = static_cast<T>(0);
        this->components[Super::indexOf(3, 3)] = static_cast<T>(1);

        return *this;
    }
    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTMATRIX_H_INCLUDED */
