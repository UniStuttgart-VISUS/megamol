/*
 * AbstractMatrix.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTMATRIX_H_INCLUDED
#define VISLIB_ABSTRACTMATRIX_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractMatrixImpl.h"
#include "vislib/IllegalStateException.h"
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
         * Calculates the characteristic polynom of the matrix
         *
         * @return The characteristic polynom of the matrix
         *
         * @throw Exception if the calculation of the polynom fails.
         */
        inline Polynom<T, D> CharacteristicPolynom(void) const {
            return Super::characteristicPolynom();
        }

        /**
         * Answer the determinant of this matrix.
         *
         * Note that the implementation uses a Gaussian elimination and is 
         * therefore very slow.
         *
         * @return The determinant of the matrix.
         */
        T Determinant(void) const;

        /**
         * Calculates eigenvalues and eigenvectors of the matrix. The order of
         * eigenvalues is undefined. The eigenvectors will be ordered like the
         * eigenvalues. At most 'size' results will be written to the output.
         * A DxD matrix has a most D unique real eigenvalues.
         *
         * @param outEigenvalues Pointer to the array receiving the found
         *                       eigenvalues. If null, no eigenvalues will be
         *                       stored.
         * @param outEigenvectors Pointer to the array receiving the found
         *                        eigenvectors. If null, no eigenvectors will
         *                        be stored.
         * @param size The size of 'outEigenvalues' and 'outEigenvectors' in
         *             number of elements.
         *
         * @return The number of results written to the output arrays.
         */
        unsigned int FindEigenvalues(T *outEigenvalues,
            Vector<T, D> *outEigenvectors, unsigned int size) const;

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


    /*
     * AbstractMatrix<T, D, L, S>::FindEigenvalues
     */
    template<class T, unsigned int D, MatrixLayout L, class S>
    unsigned int AbstractMatrix<T, D, L, S>::FindEigenvalues(
            T *outEigenvalues, Vector<T, D> *outEigenvectors,
            unsigned int size) const {
        if (this->IsSymmetric()) {
            return this->findEigenvaluesSym(outEigenvalues,
                outEigenvectors, size);
        }
        if (outEigenvectors == NULL) {
            return this->CharacteristicPolynom().FindRoots(
                outEigenvalues, size);
        }

        // TODO: Implement something better

        throw vislib::UnsupportedOperationException("FindEigenvalues",
            __FILE__, __LINE__);

        return 0;
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::Determinant
     */
    template<class T, unsigned int D, MatrixLayout L, class S>
    T AbstractMatrix<T, D, L, S>::Determinant(void) const {
#define A(r, c) a[(r) * D + (c)]
        double a[D * D];                    // input matrix for algorithm
        double f;                           // Multiplication factor.
        double max;                         // Row pivotising.
        unsigned int pRow;                  // Pivot row.
        unsigned int s;                     // Current eliminination step.
        T retval = static_cast<T>(1);       // The resulting determinant.

        /*
         * Create double precision row-major matrix copy as well-defined basis
         * for Gauﬂ elimination. 
         */
        for (unsigned int r = 0; r < D; r++) {
            for (unsigned int c = 0; c < D; c++) {
                A(r, c) = static_cast<double>(
                    this->components[Super::indexOf(r, c)]);
            }
        }

        /* Gauﬂ elimination. */
        s = 0;
        do {

            /* Pivotising. */
            max = ::fabs(A(s, s));
            pRow = s; 
            for (unsigned int r = s + 1; r < D; r++) {
                if (::fabs(A(r, s)) > max) {
                    max = ::fabs(A(r, s));
                    pRow = r;
                }
            }

            if (max < DOUBLE_EPSILON) {
                /*
                 * Matrix is not invertable, because the column cannot be 
                 * deleted. The determinant is zero, iff the matrix is not
                 * invertable.
                 */
                return static_cast<T>(0);
            }

            if (pRow != s) {
                // if necessary, exchange the row
                double h;

                for (unsigned int c = s ; c < D; c++) {
                    h = A(s, c);
                    A(s, c) = A(pRow, c);
                    A(pRow, c) = h;
                }

                retval *= -1.0; // Exchaning rows changes sign.
            } 

            /* Elimination. */
            for (unsigned int r = s + 1; r < D; r++ ) {
                f = -A(r, s) / A(s, s);
                for (unsigned int c = s; c < D; c++) {
                    A(r, c) += f * A(s, c);
                } 
            }

            s++;
        } while (s < D);

        /* Compute determinant as product of the diagonal. */
        ASSERT(D > 0);
        ASSERT(::fabs(retval) == 1.0);
        for (unsigned int i = 0; i < D; i++) {
            retval *= A(i, i);
        }

        return retval;
#undef A
    }


    /**
     * Partial template specialisation for 2x2 matrices.
     */
    template<class T, MatrixLayout L, class S>
    class AbstractMatrix<T, 2, L, S>
            : public AbstractMatrixImpl<T, 2, L, S, AbstractMatrix> {

    public:

        /** Dtor. */
        ~AbstractMatrix(void);

        /**
         * Calculates the characteristic polynom of the matrix
         *
         * @return The characteristic polynom of the matrix
         *
         * @throw Exception if the calculation of the polynom fails.
         */
        inline Polynom<T, 2> CharacteristicPolynom(void) const {
            Polynom<T, 2> rv;
            // x^2 - trace(A)x + det(A)
            rv[0] = Super::determinant2x2(
                this->components[Super::indexOf(0, 0)],
                this->components[Super::indexOf(1, 0)],
                this->components[Super::indexOf(0, 1)],
                this->components[Super::indexOf(1, 1)]);
            rv[1] = -(this->components[Super::indexOf(0, 0)]
                + this->components[Super::indexOf(1, 1)]);
            rv[2] = static_cast<T>(1);
            return rv;
        }

        /**
         * Answer the determinant of this matrix.
         *
         * @return The determinant of the matrix.
         */
        inline T Determinant(void) const {
            return Super::determinant2x2(
                this->components[Super::indexOf(0, 0)],
                this->components[Super::indexOf(1, 0)],
                this->components[Super::indexOf(0, 1)],
                this->components[Super::indexOf(1, 1)]);
        }

        /**
         * Calculates eigenvalues and eigenvectors of the matrix. The order of
         * eigenvalues is undefined. The eigenvectors will be ordered like the
         * eigenvalues. At most 'size' results will be written to the output.
         * A DxD matrix has a most D unique real eigenvalues.
         *
         * @param outEigenvalues Pointer to the array receiving the found
         *                       eigenvalues. If null, no eigenvalues will be
         *                       stored.
         * @param outEigenvectors Pointer to the array receiving the found
         *                        eigenvectors. If null, no eigenvectors will
         *                        be stored.
         * @param size The size of 'outEigenvalues' and 'outEigenvectors' in
         *             number of elements.
         *
         * @return The number of results written to the output arrays.
         */
        unsigned int FindEigenvalues(T *outEigenvalues,
            Vector<T, 2> *outEigenvectors, unsigned int size) const;

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
        typedef AbstractMatrixImpl<T, 2, L, S, vislib::math::AbstractMatrix>
            Super;

        /**
         * Disallow instances of this class. 
         */
        inline AbstractMatrix(void) : Super() {}

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
        inline AbstractMatrix(const AbstractMatrixImpl<T, 2, L, S1, 
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
     * vislib::math::AbstractMatrix<T, 2, L, S>::~AbstractMatrix
     */
    template<class T, MatrixLayout L, class S>
    AbstractMatrix<T, 2, L, S>::~AbstractMatrix(void) {
    }


    /*
     * AbstractMatrix<T, 2, L, S, C>::FindEigenvalues
     */
    template<class T, MatrixLayout L, class S>
    unsigned int AbstractMatrix<T, 2, L, S>::FindEigenvalues(
            T *outEigenvalues, Vector<T, 2> *outEigenvectors,
            unsigned int size) const {
        if (((outEigenvalues == NULL) && (outEigenvectors == NULL))
            || (size == 0)) return 0;

        // implementation based on:
        // http://www.iazd.uni-hannover.de/~erne/Mathematik1/dateien/maple/
        //    MB_5_2.html

        T ev[2];
        T evv[2][2];
        unsigned int evc = this->CharacteristicPolynom().FindRoots(ev, 2);
        if (evc == 0) return 0; // no eigenvalues
        if (evc == 1) ev[1] = ev[0];

        if (IsEqual(this->components[Super::indexof(1, 0)],
                static_cast<T>(0))) {
            if (IsEqual(this->components[Super::indexof(0, 1)],
                    static_cast<T>(0))) {
                evv[0][0] = static_cast<T>(1);
                evv[0][1] = static_cast<T>(0);
                evv[1][0] = static_cast<T>(0);
                evv[1][1] = static_cast<T>(1);
            } else {
                evv[0][0] = ev[0] - this->components[Super::indexof(1, 1)];
                evv[0][1] = this->components[Super::indexof(1, 0)];
                evv[1][0] = ev[1] - this->components[Super::indexof(1, 1)];
                evv[1][1] = this->components[Super::indexof(1, 0)];
            }
        } else {
            evv[0][0] = this->components[Super::indexof(0, 1)];
            evv[0][1] = ev[0] - this->components[Super::indexof(0, 0)];
            evv[1][0] = this->components[Super::indexof(0, 1)];
            evv[1][1] = ev[1] - this->components[Super::indexof(0, 0)];
        }

        if (outEigenvalues != NULL) {
            outEigenvalues[0] = ev[0];
            if (size > 1) {
                outEigenvalues[1] = ev[1];
            }
        }

        if (outEigenvectors != NULL) {
            outEigenvectors[0].Set(evv[0][0], evv[0][1]);
            if (size > 1) {
                outEigenvectors[1].Set(evv[1][0], evv[1][1]);
            }
        }

        return (size > 1) ? 2 : 1;
    }


    /**
     * Partial template specialisation for 3x3 matrices.
     */
    template<class T, MatrixLayout L, class S>
    class AbstractMatrix<T, 3, L, S>
            : public AbstractMatrixImpl<T, 3, L, S, AbstractMatrix> {

    public:

        /** Dtor. */
        ~AbstractMatrix(void);

        /**
         * Calculates the characteristic polynom of the matrix
         *
         * @return The characteristic polynom of the matrix
         *
         * @throw Exception if the calculation of the polynom fails.
         */
        inline Polynom<T, 3> CharacteristicPolynom(void) const {
            Polynom<T, 3> p;
            p[0] = this->Determinant();
            p[1] = -(
                Super::determinant2x2(
                    this->components[Super::indexOf(1, 1)],
                    this->components[Super::indexOf(2, 1)],
                    this->components[Super::indexOf(1, 2)],
                    this->components[Super::indexOf(2, 2)])
                + Super::determinant2x2(
                    this->components[Super::indexOf(0, 0)],
                    this->components[Super::indexOf(2, 0)],
                    this->components[Super::indexOf(0, 2)],
                    this->components[Super::indexOf(2, 2)])
                + Super::determinant2x2(
                    this->components[Super::indexOf(0, 0)],
                    this->components[Super::indexOf(1, 0)],
                    this->components[Super::indexOf(0, 1)],
                    this->components[Super::indexOf(1, 1)]));
            p[2] = this->Trace();
            p[3] = static_cast<T>(-1);
            return p;
        }

        /**
         * Answer the determinant of this matrix.
         *
         * @return The determinant of the matrix.
         */
        inline T Determinant(void) const {
            return Super::determinant3x3(
                this->components[Super::indexOf(0, 0)],
                this->components[Super::indexOf(1, 0)],
                this->components[Super::indexOf(2, 0)],
                this->components[Super::indexOf(0, 1)],
                this->components[Super::indexOf(1, 1)],
                this->components[Super::indexOf(2, 1)],
                this->components[Super::indexOf(0, 2)],
                this->components[Super::indexOf(1, 2)],
                this->components[Super::indexOf(2, 2)]);
        }

        /**
         * Calculates eigenvalues and eigenvectors of the matrix. The order of
         * eigenvalues is undefined. The eigenvectors will be ordered like the
         * eigenvalues. At most 'size' results will be written to the output.
         * A DxD matrix has a most D unique real eigenvalues.
         *
         * @param outEigenvalues Pointer to the array receiving the found
         *                       eigenvalues. If null, no eigenvalues will be
         *                       stored.
         * @param outEigenvectors Pointer to the array receiving the found
         *                        eigenvectors. If null, no eigenvectors will
         *                        be stored.
         * @param size The size of 'outEigenvalues' and 'outEigenvectors' in
         *             number of elements.
         *
         * @return The number of results written to the output arrays.
         */
        unsigned int FindEigenvalues(T *outEigenvalues,
            Vector<T, 3> *outEigenvectors, unsigned int size) const;

        /**
         * Answer if this matrix describes a pure rotation. This is the case
         * if the matrix is orthogonal and has a determinant of one.
         *
         * @return true, if this matrix describes a pure rotation.
         */
        bool IsRotation(void) const;

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

        /**
         * Answer the quaternion representing the rotation of this matrix.
         *
         * @return A quaternion representing the rotation.
         *
         * @throw IllegalStateException if the matrix is not a rotation-only
         *                              matrix.
         */
        operator Quaternion<T>(void) const;

    protected:

        /** A typedef for the super class. */
        typedef AbstractMatrixImpl<T, 3, L, S, vislib::math::AbstractMatrix>
            Super;

        /**
         * Disallow instances of this class. 
         */
        inline AbstractMatrix(void) : Super() {}

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
        inline AbstractMatrix(const AbstractMatrixImpl<T, 3, L, S1, 
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
     * vislib::math::AbstractMatrix<T, 3, L, S>::~AbstractMatrix
     */
    template<class T, MatrixLayout L, class S>
    AbstractMatrix<T, 3, L, S>::~AbstractMatrix(void) {
    }


    /*
     * AbstractMatrix<T, 3, L, S>::FindEigenvalues
     */
    template<class T, MatrixLayout L, class S>
    unsigned int AbstractMatrix<T, 3, L, S>::FindEigenvalues(
            T *outEigenvalues, Vector<T, 3> *outEigenvectors,
            unsigned int size) const {
        if (this->IsSymmetric()) {
            return this->findEigenvaluesSym(outEigenvalues,
                outEigenvectors, size);
        }
        if (outEigenvectors == NULL) {
            return this->CharacteristicPolynom().FindRoots(
                outEigenvalues, size);
        }

        // TODO: Implement something better

        throw vislib::UnsupportedOperationException("FindEigenvalues",
            __FILE__, __LINE__);

        return 0;
    }


    /*
     * AbstractMatrix<T, 3, L, S>::IsRotation
     */
    template<class T, MatrixLayout L, class S>
    bool AbstractMatrix<T, 3, L, S>::IsRotation(void) const {
        return IsEqual(Super::determinant3x3(
                this->components[Super::indexOf(0, 0)],
                this->components[Super::indexOf(1, 0)],
                this->components[Super::indexOf(2, 0)],
                this->components[Super::indexOf(0, 1)],
                this->components[Super::indexOf(1, 1)],
                this->components[Super::indexOf(2, 1)],
                this->components[Super::indexOf(0, 2)],
                this->components[Super::indexOf(1, 2)],
                this->components[Super::indexOf(2, 2)]), static_cast<T>(1))
            && IsEqual((this->components[Super::indexOf(0, 0)]
                    *   this->components[Super::indexOf(0, 0)])
                + (     this->components[Super::indexOf(1, 0)]
                    *   this->components[Super::indexOf(1, 0)])
                + (     this->components[Super::indexOf(2, 0)]
                    *   this->components[Super::indexOf(2, 0)]),
                static_cast<T>(1))
            && IsEqual((this->components[Super::indexOf(0, 0)]
                    *   this->components[Super::indexOf(0, 1)])
                + (     this->components[Super::indexOf(1, 0)]
                    *   this->components[Super::indexOf(1, 1)])
                + (     this->components[Super::indexOf(2, 0)]
                    *   this->components[Super::indexOf(2, 1)]),
                static_cast<T>(0))
            && IsEqual((this->components[Super::indexOf(0, 0)]
                    *   this->components[Super::indexOf(0, 2)])
                + (     this->components[Super::indexOf(1, 0)]
                    *   this->components[Super::indexOf(1, 2)])
                + (     this->components[Super::indexOf(2, 0)]
                    *   this->components[Super::indexOf(2, 2)]),
                static_cast<T>(0))
            && IsEqual((this->components[Super::indexOf(0, 1)]
                    *   this->components[Super::indexOf(0, 0)])
                + (     this->components[Super::indexOf(1, 1)]
                    *   this->components[Super::indexOf(1, 0)])
                + (     this->components[Super::indexOf(2, 1)]
                    *   this->components[Super::indexOf(2, 0)]),
                static_cast<T>(0))
            && IsEqual((this->components[Super::indexOf(0, 1)]
                    *   this->components[Super::indexOf(0, 1)])
                + (     this->components[Super::indexOf(1, 1)]
                    *   this->components[Super::indexOf(1, 1)])
                + (     this->components[Super::indexOf(2, 1)]
                    *   this->components[Super::indexOf(2, 1)]),
                static_cast<T>(1))
            && IsEqual((this->components[Super::indexOf(0, 1)]
                    *   this->components[Super::indexOf(0, 2)])
                + (     this->components[Super::indexOf(1, 1)]
                    *   this->components[Super::indexOf(1, 2)])
                + (     this->components[Super::indexOf(2, 1)]
                    *   this->components[Super::indexOf(2, 2)]),
                static_cast<T>(0))
            && IsEqual((this->components[Super::indexOf(0, 2)]
                    *   this->components[Super::indexOf(0, 0)])
                + (     this->components[Super::indexOf(1, 2)]
                    *   this->components[Super::indexOf(1, 0)])
                + (     this->components[Super::indexOf(2, 2)]
                    *   this->components[Super::indexOf(2, 0)]),
                static_cast<T>(0))
            && IsEqual((this->components[Super::indexOf(0, 2)]
                    *   this->components[Super::indexOf(0, 1)])
                + (     this->components[Super::indexOf(1, 2)]
                    *   this->components[Super::indexOf(1, 1)])
                + (     this->components[Super::indexOf(2, 2)]
                    *   this->components[Super::indexOf(2, 1)]),
                static_cast<T>(0))
            && IsEqual((this->components[Super::indexOf(0, 2)]
                    *   this->components[Super::indexOf(0, 2)])
                + (     this->components[Super::indexOf(1, 2)]
                    *   this->components[Super::indexOf(1, 2)])
                + (     this->components[Super::indexOf(2, 2)]
                    *   this->components[Super::indexOf(2, 2)]),
                static_cast<T>(1));
    }


    /*
     * AbstractMatrix<T, 3, L, S>::operator =
     */
    template<class T, MatrixLayout L, class S>
    template<class Tp, class Sp>
    AbstractMatrix<T, 3, L, S>& AbstractMatrix<T, 3, L, S>::operator =(
            const AbstractQuaternion<Tp, Sp>& rhs) {
        Quaternion<T> q(rhs);
        q.Normalise();

        this->components[Super::indexOf(0, 0)] 
            = Sqr(q.W()) + Sqr(q.X()) - Sqr(q.Y()) - Sqr(q.Z());
        this->components[Super::indexOf(0, 1)] 
            = static_cast<T>(2) * (q.X() * q.Y() - q.W() * q.Z());
        this->components[Super::indexOf(0, 2)] 
            = static_cast<T>(2) * (q.W() * q.Y() + q.X() * q.Z()); 

        this->components[Super::indexOf(1, 0)] 
            = static_cast<T>(2) * (q.W() * q.Z() + q.X() * q.Y());
        this->components[Super::indexOf(1, 1)] 
            = Sqr(q.W()) - Sqr(q.X()) + Sqr(q.Y()) - Sqr(q.Z());
        this->components[Super::indexOf(1, 2)] 
            = static_cast<T>(2) * (q.Y() * q.Z() - q.W() * q.X());

        this->components[Super::indexOf(2, 0)] 
            = static_cast<T>(2) * (q.X() * q.Z() - q.W() * q.Y());
        this->components[Super::indexOf(2, 1)] 
            = static_cast<T>(2) * (q.W() * q.X() + q.Y() * q.Z());
        this->components[Super::indexOf(2, 2)] 
            = Sqr(q.W()) - Sqr(q.X()) - Sqr(q.Y()) + Sqr(q.Z());

        return *this;
    }


    /*
     * vislib::math::AbstractMatrix<T, 3, L, S>::operator Quaternion<T>
     */
    template<class T, MatrixLayout L, class S>
    AbstractMatrix<T, 3, L, S>::operator Quaternion<T>(void) const {
        Quaternion<T> q;
        if (!this->IsRotation()) {
            throw IllegalStateException("Matrix is not rotation-only", __FILE__, __LINE__);
        }
        try {
            q.SetFromRotationMatrix(this->components[Super::indexOf(0, 0)],
                this->components[Super::indexOf(0, 1)],
                this->components[Super::indexOf(0, 2)],
                this->components[Super::indexOf(1, 0)],
                this->components[Super::indexOf(1, 1)],
                this->components[Super::indexOf(1, 2)],
                this->components[Super::indexOf(2, 0)],
                this->components[Super::indexOf(2, 1)],
                this->components[Super::indexOf(2, 2)]);
        } catch(...) {
            throw IllegalStateException("Matrix is not rotation-only", __FILE__, __LINE__);
        }
        return q;
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

        /**
         * Calculates the characteristic polynom of the matrix
         *
         * @return The characteristic polynom of the matrix
         *
         * @throw Exception if the calculation of the polynom fails.
         */
        inline Polynom<T, 4> CharacteristicPolynom(void) const {

            // TODO: Implement something better

            return Super::characteristicPolynom();
        }

        /**
         * Answer the determinant of this matrix.
         *
         * @return The determinant of the matrix.
         */
        inline T Determinant(void) const {
            return Super::determinant4x4(
                this->components[Super::indexOf(0, 0)],
                this->components[Super::indexOf(1, 0)],
                this->components[Super::indexOf(2, 0)],
                this->components[Super::indexOf(3, 0)],
                this->components[Super::indexOf(0, 1)],
                this->components[Super::indexOf(1, 1)],
                this->components[Super::indexOf(2, 1)],
                this->components[Super::indexOf(3, 1)],
                this->components[Super::indexOf(0, 2)],
                this->components[Super::indexOf(1, 2)],
                this->components[Super::indexOf(2, 2)],
                this->components[Super::indexOf(3, 2)],
                this->components[Super::indexOf(0, 3)],
                this->components[Super::indexOf(1, 3)],
                this->components[Super::indexOf(2, 3)],
                this->components[Super::indexOf(3, 3)]);
        }

        /**
         * Calculates eigenvalues and eigenvectors of the matrix. The order of
         * eigenvalues is undefined. The eigenvectors will be ordered like the
         * eigenvalues. At most 'size' results will be written to the output.
         * A DxD matrix has a most D unique real eigenvalues.
         *
         * @param outEigenvalues Pointer to the array receiving the found
         *                       eigenvalues. If null, no eigenvalues will be
         *                       stored.
         * @param outEigenvectors Pointer to the array receiving the found
         *                        eigenvectors. If null, no eigenvectors will
         *                        be stored.
         * @param size The size of 'outEigenvalues' and 'outEigenvectors' in
         *             number of elements.
         *
         * @return The number of results written to the output arrays.
         */
        unsigned int FindEigenvalues(T *outEigenvalues,
            Vector<T, 4> *outEigenvectors, unsigned int size) const;

        /**
         * Answer if this matrix describes a pure rotation. This is the case
         * if the upper left 3x3 matrix is orthogonal and has a determinant of
         * one, and the remaining components are equal to those from the
         * identity matrix.
         *
         * @return true, if this matrix describes a pure rotation.
         */
        bool IsRotation(void) const;

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

        /**
         * Answer the quaternion representing the rotation of this matrix.
         *
         * @return A quaternion representing the rotation.
         *
         * @throw IllegalStateException if the matrix is not a rotation-only
         *                              matrix.
         */
        operator Quaternion<T>(void) const;

    protected:

        /** A typedef for the super class. */
        typedef AbstractMatrixImpl<T, 4, L, S, vislib::math::AbstractMatrix>
            Super;

        /**
         * Disallow instances of this class. 
         */
        inline AbstractMatrix(void) : Super() {}

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
        inline AbstractMatrix(const AbstractMatrixImpl<T, 4, L, S1, 
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
     * vislib::math::AbstractMatrix<T, 4, L, S>::~AbstractMatrix
     */
    template<class T, MatrixLayout L, class S>
    AbstractMatrix<T, 4, L, S>::~AbstractMatrix(void) {
    }


    /*
     * AbstractMatrix<T, 4, L, S>::FindEigenvalues
     */
    template<class T, MatrixLayout L, class S>
    unsigned int AbstractMatrix<T, 4, L, S>::FindEigenvalues(
            T *outEigenvalues, Vector<T, 4> *outEigenvectors,
            unsigned int size) const {
        if (this->IsSymmetric()) {
            return this->findEigenvaluesSym(outEigenvalues,
                outEigenvectors, size);
        }
        if (outEigenvectors == NULL) {
            return this->CharacteristicPolynom().FindRoots(
                outEigenvalues, size);
        }

        // TODO: Implement something better

        throw vislib::UnsupportedOperationException("FindEigenvalues",
            __FILE__, __LINE__);

        return 0;
    }


    /*
     * AbstractMatrix<T, 4, L, S>::FindEigenvalues
     */
    template<class T, MatrixLayout L, class S>
    bool AbstractMatrix<T, 4, L, S>::IsRotation(void) const {
        return IsEqual(Super::determinant3x3(
                this->components[Super::indexOf(0, 0)],
                this->components[Super::indexOf(1, 0)],
                this->components[Super::indexOf(2, 0)],
                this->components[Super::indexOf(0, 1)],
                this->components[Super::indexOf(1, 1)],
                this->components[Super::indexOf(2, 1)],
                this->components[Super::indexOf(0, 2)],
                this->components[Super::indexOf(1, 2)],
                this->components[Super::indexOf(2, 2)]), static_cast<T>(1))
            && IsEqual((this->components[Super::indexOf(0, 0)]
                    *   this->components[Super::indexOf(0, 0)])
                + (     this->components[Super::indexOf(1, 0)]
                    *   this->components[Super::indexOf(1, 0)])
                + (     this->components[Super::indexOf(2, 0)]
                    *   this->components[Super::indexOf(2, 0)]),
                static_cast<T>(1))
            && IsEqual((this->components[Super::indexOf(0, 0)]
                    *   this->components[Super::indexOf(0, 1)])
                + (     this->components[Super::indexOf(1, 0)]
                    *   this->components[Super::indexOf(1, 1)])
                + (     this->components[Super::indexOf(2, 0)]
                    *   this->components[Super::indexOf(2, 1)]),
                static_cast<T>(0))
            && IsEqual((this->components[Super::indexOf(0, 0)]
                    *   this->components[Super::indexOf(0, 2)])
                + (     this->components[Super::indexOf(1, 0)]
                    *   this->components[Super::indexOf(1, 2)])
                + (     this->components[Super::indexOf(2, 0)]
                    *   this->components[Super::indexOf(2, 2)]),
                static_cast<T>(0))
            && IsEqual((this->components[Super::indexOf(0, 1)]
                    *   this->components[Super::indexOf(0, 0)])
                + (     this->components[Super::indexOf(1, 1)]
                    *   this->components[Super::indexOf(1, 0)])
                + (     this->components[Super::indexOf(2, 1)]
                    *   this->components[Super::indexOf(2, 0)]),
                static_cast<T>(0))
            && IsEqual((this->components[Super::indexOf(0, 1)]
                    *   this->components[Super::indexOf(0, 1)])
                + (     this->components[Super::indexOf(1, 1)]
                    *   this->components[Super::indexOf(1, 1)])
                + (     this->components[Super::indexOf(2, 1)]
                    *   this->components[Super::indexOf(2, 1)]),
                static_cast<T>(1))
            && IsEqual((this->components[Super::indexOf(0, 1)]
                    *   this->components[Super::indexOf(0, 2)])
                + (     this->components[Super::indexOf(1, 1)]
                    *   this->components[Super::indexOf(1, 2)])
                + (     this->components[Super::indexOf(2, 1)]
                    *   this->components[Super::indexOf(2, 2)]),
                static_cast<T>(0))
            && IsEqual((this->components[Super::indexOf(0, 2)]
                    *   this->components[Super::indexOf(0, 0)])
                + (     this->components[Super::indexOf(1, 2)]
                    *   this->components[Super::indexOf(1, 0)])
                + (     this->components[Super::indexOf(2, 2)]
                    *   this->components[Super::indexOf(2, 0)]),
                static_cast<T>(0))
            && IsEqual((this->components[Super::indexOf(0, 2)]
                    *   this->components[Super::indexOf(0, 1)])
                + (     this->components[Super::indexOf(1, 2)]
                    *   this->components[Super::indexOf(1, 1)])
                + (     this->components[Super::indexOf(2, 2)]
                    *   this->components[Super::indexOf(2, 1)]),
                static_cast<T>(0))
            && IsEqual((this->components[Super::indexOf(0, 2)]
                    *   this->components[Super::indexOf(0, 2)])
                + (     this->components[Super::indexOf(1, 2)]
                    *   this->components[Super::indexOf(1, 2)])
                + (     this->components[Super::indexOf(2, 2)]
                    *   this->components[Super::indexOf(2, 2)]),
                static_cast<T>(1))
            && IsEqual(this->components[Super::indexOf(3, 0)],
                static_cast<T>(0))
            && IsEqual(this->components[Super::indexOf(3, 1)],
                static_cast<T>(0))
            && IsEqual(this->components[Super::indexOf(3, 2)],
                static_cast<T>(0))
            && IsEqual(this->components[Super::indexOf(3, 3)],
                static_cast<T>(1))
            && IsEqual(this->components[Super::indexOf(2, 3)],
                static_cast<T>(0))
            && IsEqual(this->components[Super::indexOf(1, 3)],
                static_cast<T>(0))
            && IsEqual(this->components[Super::indexOf(0, 3)],
                static_cast<T>(0));
    }


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
            = static_cast<T>(2) * (q.W() * q.X() + q.Y() * q.Z());
        this->components[Super::indexOf(2, 2)] 
            = Sqr(q.W()) - Sqr(q.X()) - Sqr(q.Y()) + Sqr(q.Z());
        this->components[Super::indexOf(2, 3)] = static_cast<T>(0);

        this->components[Super::indexOf(3, 0)] = static_cast<T>(0);
        this->components[Super::indexOf(3, 1)] = static_cast<T>(0);
        this->components[Super::indexOf(3, 2)] = static_cast<T>(0);
        this->components[Super::indexOf(3, 3)] = static_cast<T>(1);

        return *this;
    }

    /*
     * vislib::math::AbstractMatrix<T, 4, L, S>::operator Quaternion<T>
     */
    template<class T, MatrixLayout L, class S>
    AbstractMatrix<T, 4, L, S>::operator Quaternion<T>(void) const {
        Quaternion<T> q;
        if (!this->IsRotation()) {
            throw IllegalStateException("Matrix is not rotation-only", __FILE__, __LINE__);
        }
        try {
            q.SetFromRotationMatrix(this->components[Super::indexOf(0, 0)],
                this->components[Super::indexOf(0, 1)],
                this->components[Super::indexOf(0, 2)],
                this->components[Super::indexOf(1, 0)],
                this->components[Super::indexOf(1, 1)],
                this->components[Super::indexOf(1, 2)],
                this->components[Super::indexOf(2, 0)],
                this->components[Super::indexOf(2, 1)],
                this->components[Super::indexOf(2, 2)]);
        } catch(...) {
            throw IllegalStateException("Matrix is not rotation-only", __FILE__, __LINE__);
        }
        return q;
    }

} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTMATRIX_H_INCLUDED */
