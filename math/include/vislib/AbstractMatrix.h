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
         * Computes 'sqrt(a * a + b * b)' without destructive underflow or
         * overflow.
         *
         * @param a The first operand
         * @param b The second operand
         *
         * @return The result
         */
        static inline double pythag(double a, double b) {
            double absa = ::fabs(a);
            double absb = ::fabs(b);
            if (absa > absb) {
                absb /= absa;
                absb *= absb;
                return absa * sqrt(1.0 + absb);
            }
            if (IsEqual(absb, 0.0)) return 0.0;
            absa /= absb;
            absa *= absa;
            return absb * sqrt(1.0 + absa);
        }

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

                //
                // THIS Implementation is completely wrong
                //
                // It only works for SYMMETRIC matrices
                //

//        if (((outEigenvalues == NULL) && (outEigenvectors == NULL))
//                || (size == 0)) return 0;
//#define A(r, c) a[(r) * D + (c)]
//        double a[D * D];                    // input matrix for algorithm
//        double d[D];                        // diagonal elements
//        double e[D];                        // off-diagonal elements
//
//        for (unsigned int r = 0; r < D; r++) {
//            for (unsigned int c = 0; c < D; c++) {
//                A(r, c) = static_cast<double>(this->components[indexOf(r, c)]);
//            }
//        }
//
//        // 1. Householder reduction.
//        int l, k, j, i;
//        double scale, hh, h, g, f;
//
//        for (i = D; i >= 2; i--) {
//            l = i - 1;
//            h = scale = 0.0;
//            if (l > 1) {
//                for (k = 1; k <= l; k++) {
//                    scale += ::fabs(A(i - 1, k - 1));
//                }
//                if (IsEqual(scale, 0.0)) {
//                    e[i - 1] = A(i - 1, l - 1);
//                } else {
//                    for (k = 1; k <= l; k++) {
//                        A(i - 1, k - 1) /= scale;
//                        h += A(i - 1, k - 1) * A(i - 1, k - 1);
//                    }
//                    f = A(i - 1, l - 1);
//                    g = ((f >= 0.0) ? -::sqrt(h) : ::sqrt(h));
//                    e[i - 1] = scale * g;
//                    h -= f * g;
//                    A(i - 1, l - 1) = f - g;
//                    f = 0.0;
//                    for (j = 1; j <= l; j++) {
//                        A(j - 1, i - 1) = A(i - 1, j - 1) / h;
//                        g = 0.0;
//                        for (k = 1; k <= j; k++) {
//                            g += A(j - 1, k - 1) * A(i - 1, k - 1);
//                        }
//                        for (k = j + 1; k <= l; k++) {
//                            g += A(k - 1, j - 1) * A(i - 1, k - 1);
//                        }
//                        e[j - 1] = g / h;
//                        f += e[j - 1] * A(i - 1, j - 1);
//                    }
//                    hh = f / (h + h);
//                    for (j = 1; j <= l; j++) {
//                        f = A(i - 1, j - 1);
//                        e[j - 1] = g = e[j - 1] - hh * f;
//                        for (k = 1; k <= j; k++) {
//                            A(j - 1, k - 1) -= (f * e[k - 1]
//                                + g * A(i - 1, k - 1));
//                        }
//                    }
//                }
//            } else {
//                e[i - 1] = A(i - 1, l - 1);
//            }
//            d[i - 1] = h;
//        }
//        d[0] = 0.0;
//        e[0] = 0.0;
//        for (i = 1; i <= D; i++) {
//            l = i - 1;
//            if (!IsEqual(d[i - 1], 0.0)) {
//                for (j = 1; j <= l ; j++) {
//                    g = 0.0;
//                    for (k = 1; k <= l; k++) {
//                        g += A(i - 1, k - 1) * A(k - 1, j - 1);
//                    }
//                    for (k = 1; k <= l; k++) {
//                        A(k - 1, j - 1) -= g * A(k - 1, i - 1);
//                    }
//                }
//            }
//            d[i - 1] = A(i - 1, i - 1);
//            A(i - 1, i - 1) = 1.0;
//            for (j = 1; j <= l; j++) {
//                A(j - 1, i - 1) = A(i - 1, j - 1) = 0.0;
//            }
//        }
//
//        // 2. Calculation von eigenvalues and eigenvectors (QL algorithm)
//#ifdef SIGN
//#error SIGN macro already in use! Code rewrite required!
//#endif
//#define SIGN(a, b) ((b) >= 0.0 ? ::fabs(a) : -::fabs(a))
//
//        int m, iter;
//        double s, r, p, dd, c, b;
//        const int MAX_ITER = 30;
//
//        for (i = 2; i <= D; i++) {
//            e[i - 2] = e[i - 1];
//        }
//        e[D - 1] = 0.0;
//
//        for (l = 1; l <= D; l++) {
//            iter = 0;
//            do {
//                for (m = l; m <= D - 1; m++) {
//                    dd = ::fabs(d[m - 1]) + ::fabs(d[m - 1 + 1]);
//                    if (IsEqual(::fabs(e[m - 1]) + dd, dd)) break;
//                }
//                if (m != l) {
//                    if (iter++ == MAX_ITER) {
//                        throw vislib::Exception(
//                            "Too many iterations in FindEigenvalues",
//                            __FILE__, __LINE__);
//                    }
//                    g = (d[l - 1 + 1] - d[l - 1]) / (2.0 * e[l - 1]);
//                    r = pythag(g, 1.0);
//                    g = d[m - 1] - d[l - 1] + e [l - 1] / (g + SIGN(r, g));
//                    s = c = 1.0;
//                    p = 0.0;
//                    for (i = m - 1; i >= l ; i--) {
//                        f = s * e[i - 1];
//                        b = c * e[i - 1];
//                        e[i - 1 + 1] = r = pythag(f, g);
//                        if (IsEqual(r, 0.0)) {
//                            d[i - 1 + 1] -= p;
//                            e[m - 1] = 0.0;
//                            break;
//                        }
//                        s = f / r;
//                        c = g / r;
//                        g = d[i - 1 + 1] - p;
//                        r = (d[i - 1] - g) * s + 2.0 * c * b;
//                        d[i - 1 + 1] = g + (p = s * r);
//                        g = c * r - b;
//                        for (k = 1; k <= D; k++) {
//                            f = A(k - 1, i - 1 + 1);
//                            A(k - 1, i - 1 + 1) = s * A(k - 1, i - 1) + c * f;
//                            A(k - 1, i - 1) = c * A(k - 1, i - 1) - s * f;
//                        }
//                    }
//                    if (IsEqual(r, 0.0) && (i >= l)) continue;
//                    d[l - 1] -= p;
//                    e[l - 1] = g;
//                    e[m - 1] = 0.0;
//                }
//            } while (m != l);
//        }
//#undef SIGN
//
//        // 3. output
//        if (outEigenvalues != NULL){
//            for (i = 0; i < static_cast<int>(Min(D, size)); i++) {
//                outEigenvalues[i] = static_cast<T>(d[i]);
//            }
//        }
//        if (outEigenvectors != NULL){
//            for (i = 0; i < static_cast<int>(Min(D, size)); i++) {
//                for (j = 0; j < static_cast<int>(D); j++) {
//                    outEigenvectors[i][j] = static_cast<T>(A(i, j));
//                }
//            }
//        }
//
//#undef A
//        return Min(D, size);

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
            return Super::characteristicPolynom();
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

        throw vislib::UnsupportedOperationException("FindEigenvalues",
            __FILE__, __LINE__);

        return 0;
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
            return Super::characteristicPolynom();
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
         */
        //operator Quaternion<T>(void) const;

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

        throw vislib::UnsupportedOperationException("FindEigenvalues",
            __FILE__, __LINE__);

        return 0;
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
            = static_cast<T>(2) * (q.W() * q.X() - q.Y() * q.Z());
        this->components[Super::indexOf(2, 2)] 
            = Sqr(q.W()) - Sqr(q.X()) - Sqr(q.Y()) + Sqr(q.Z());

        return *this;
    }


    /*
     * vislib::math::AbstractMatrix<T, 3, L, S>::operator Quaternion<T>
     */
    //template<class T, MatrixLayout L, class S>
    //AbstractMatrix<T, 3, L, S>::operator Quaternion<T>(void) const {
    //    double r = Sqrt(1.0 + this->components[Super::indexOf(0, 0)]
    //        + this->components[Super::indexOf(1, 1)]
    //        + this->components[Super::indexOf(2, 2]);
    //    TODO: Implement
    //}


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
         */
        //operator Quaternion<T>(void) const;

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

        throw vislib::UnsupportedOperationException("FindEigenvalues",
            __FILE__, __LINE__);

        return 0;
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

    /*
     * vislib::math::AbstractMatrix<T, 4, L, S>::operator Quaternion<T>
     */
    //template<class T, MatrixLayout L, class S>
    //AbstractMatrix<T, 4, L, S>::operator Quaternion<T>(void) const {
    //    double r = Sqrt(1.0 + this->components[Super::indexOf(0, 0)]
    //        + this->components[Super::indexOf(1, 1)]
    //        + this->components[Super::indexOf(2, 2]);
    //    TODO: Implement
    //}

} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTMATRIX_H_INCLUDED */
