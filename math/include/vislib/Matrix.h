/*
 * Matrix.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MATRIX_H_INCLUDED
#define VISLIB_MATRIX_H_INCLUDED
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
            ::memcpy(this->components, components, Super::CNT_COMPONENTS
                * sizeof(T));
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Matrix(const Matrix& rhs) : Super() {
            ::memcpy(this->components, rhs.component, 
                Super::CNT_COMPONENTS * sizeof(T));
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
    Matrix<T, D, L>::Matrix(const T& value) : Super() {
        for (unsigned int i = 0; i < D * D; i++) {
            this->components[i] = value;
        }
    }


    /*
     * vislib::math::Matrix<T, D, L>::~Matrix
     */
    template<class T, unsigned int D, MatrixLayout L>
    Matrix<T, D, L>::~Matrix(void) {
        // intentionally empty
    }


    /**
     * This class implements matrices that use their own memory on the stack
     * to store their components.
     * Partial template specialization for 3x3 matrices.
     */
    template<class T, MatrixLayout L> 
    class Matrix<T, 3, L> : public AbstractMatrix<T, 3, L, T[3 * 3]> {

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
            ::memcpy(this->components, components, Super::CNT_COMPONENTS
                * sizeof(T));
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Matrix(const Matrix& rhs) : Super() {
            ::memcpy(this->components, rhs.component, 
                Super::CNT_COMPONENTS * sizeof(T));
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


        /**
         * Create a matrix using the specified components. Note, that the first
         * number of the component is the row, the second the column.
         *
         * @param m11 Element in row 1, column 1.
         * @param m12 Element in row 1, column 2.
         * @param m13 Element in row 1, column 3.
         * @param m21 Element in row 2, column 1.
         * @param m22 Element in row 2, column 2.
         * @param m23 Element in row 2, column 3.
         * @param m31 Element in row 3, column 1.
         * @param m32 Element in row 3, column 2.
         * @param m33 Element in row 3, column 3.
         */
        inline Matrix(const T& m11, const T& m12, const T& m13,
                const T& m21, const T& m22, const T& m23,
                const T& m31, const T& m32, const T& m33) : Super() {
            this->components[Super::indexOf(0, 0)] = m11;
            this->components[Super::indexOf(0, 1)] = m12;
            this->components[Super::indexOf(0, 2)] = m13;
            this->components[Super::indexOf(1, 0)] = m21;
            this->components[Super::indexOf(1, 1)] = m22;
            this->components[Super::indexOf(1, 2)] = m23;
            this->components[Super::indexOf(2, 0)] = m31;
            this->components[Super::indexOf(2, 1)] = m32;
            this->components[Super::indexOf(2, 2)] = m33;
        }

        /**
         * Create a matrix that represents the rotation of the quaternion
         * 'rhs'.
         *
         * @param rhs The quaterion to be converted.
         */
        template<class Tp, class Sp>
        explicit inline Matrix(const AbstractQuaternion<Tp, Sp>& rhs)
                : Super() {
            *this=rhs;
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

        /**
         * Make this matrix represent the quaterion 'rhs'.
         *
         * @param rhs The quaterion to be converted to a rotation matrix.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        inline Matrix& operator =(const AbstractQuaternion<Tp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** A typedef for the super class. */
        typedef AbstractMatrix<T, 3, L, T[3 * 3]> Super;

    };


    /*
     * vislib::math::Matrix<T, 3, L>::Matrix
     */
    template<class T, MatrixLayout L>
    Matrix<T, 3, L>::Matrix(const T& value) : Super() {
        for (unsigned int i = 0; i < 3 * 3; i++) {
            this->components[i] = value;
        }
    }


    /*
     * vislib::math::Matrix<T, 3, L>::~Matrix
     */
    template<class T, MatrixLayout L>
    Matrix<T, 3, L>::~Matrix(void) {
        // intentionally empty
    }


    /**
     * This class implements matrices that use their own memory on the stack
     * to store their components.
     * Partial template specialization for 4x4 matrices.
     */
    template<class T, MatrixLayout L> 
    class Matrix<T, 4, L> : public AbstractMatrix<T, 4, L, T[4 * 4]> {

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
            ::memcpy(this->components, components, Super::CNT_COMPONENTS
                * sizeof(T));
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Matrix(const Matrix& rhs) : Super() {
            ::memcpy(this->components, rhs.components, 
                Super::CNT_COMPONENTS * sizeof(T));
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


        /**
         * Create a matrix using the specified components. Note, that the first
         * number of the component is the row, the second the column.
         *
         * @param m11 Element in row 1, column 1.
         * @param m12 Element in row 1, column 2.
         * @param m13 Element in row 1, column 3.
         * @param m14 Element in row 1, column 4.
         * @param m21 Element in row 2, column 1.
         * @param m22 Element in row 2, column 2.
         * @param m23 Element in row 2, column 3.
         * @param m24 Element in row 2, column 4.
         * @param m31 Element in row 3, column 1.
         * @param m32 Element in row 3, column 2.
         * @param m33 Element in row 3, column 3.
         * @param m34 Element in row 3, column 4.
         * @param m41 Element in row 4, column 1.
         * @param m42 Element in row 4, column 2.
         * @param m43 Element in row 4, column 3.
         * @param m44 Element in row 4, column 4.
         */
        Matrix(const T& m11, const T& m12, const T& m13, const T& m14,
                const T& m21, const T& m22, const T& m23, const T& m24,
                const T& m31, const T& m32, const T& m33, const T& m34,
                const T& m41, const T& m42, const T& m43, const T& m44)
                    : Super() {
            this->components[Super::indexOf(0, 0)] = m11;
            this->components[Super::indexOf(0, 1)] = m12;
            this->components[Super::indexOf(0, 2)] = m13;
            this->components[Super::indexOf(0, 3)] = m14;
            this->components[Super::indexOf(1, 0)] = m21;
            this->components[Super::indexOf(1, 1)] = m22;
            this->components[Super::indexOf(1, 2)] = m23;
            this->components[Super::indexOf(1, 3)] = m24;
            this->components[Super::indexOf(2, 0)] = m31;
            this->components[Super::indexOf(2, 1)] = m32;
            this->components[Super::indexOf(2, 2)] = m33;
            this->components[Super::indexOf(2, 3)] = m34;
            this->components[Super::indexOf(3, 0)] = m41;
            this->components[Super::indexOf(3, 1)] = m42;
            this->components[Super::indexOf(3, 2)] = m43;
            this->components[Super::indexOf(3, 3)] = m44;
        }

        /**
         * Create a matrix that represents the rotation of the quaternion
         * 'rhs'.
         *
         * @param rhs The quaterion to be converted.
         */
        template<class Tp, class Sp>
        explicit inline Matrix(const AbstractQuaternion<Tp, Sp>& rhs)
                : Super() {
            *this=rhs;
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

        /**
         * Make this matrix represent the quaterion 'rhs'.
         *
         * @param rhs The quaterion to be converted to a rotation matrix.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        inline Matrix& operator =(const AbstractQuaternion<Tp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** A typedef for the super class. */
        typedef AbstractMatrix<T, 4, L, T[4 * 4]> Super;

    };


    /*
     * vislib::math::Matrix<T, 4, L>::Matrix
     */
    template<class T, MatrixLayout L>
    Matrix<T, 4, L>::Matrix(const T& value) : Super() {
        for (unsigned int i = 0; i < 4 * 4; i++) {
            this->components[i] = value;
        }
    }


    /*
     * vislib::math::Matrix<T, 4, L>::~Matrix
     */
    template<class T, MatrixLayout L>
    Matrix<T, 4, L>::~Matrix(void) {
        // intentionally empty
    }


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_MATRIX_H_INCLUDED */

