/*
 * AbstractMatrixImpl.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTMATRIXIMPL_H_INCLUDED
#define VISLIB_ABSTRACTMATRIXIMPL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <iomanip>
#include <ios>
#include <iostream>

#include "vislib/assert.h"
#include "vislib/Exception.h"
#include "vislib/forceinline.h"
#include "vislib/mathfunctions.h"
#include "vislib/memutils.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Point.h"
#include "vislib/Polynom.h"
#include "vislib/ShallowVector.h"
#include "vislib/String.h"
#include "vislib/Vector.h"


namespace vislib {
namespace math {

    /**
     * Possible matrix memory layouts.
     *
     * COLUMN_MAJOR means that the columns are stored one after another, i. e.
     * the rows are stored contiguously. This is memory layout is used by
     * OpenGL.
     *
     * ROW_MAJOR means that the rows are stored one after another, i. e. the
     * columns are stored contiguosly. This layout is used for multidimensional
     * arrays in C and in Direct3D.
     */
    enum MatrixLayout {
        COLUMN_MAJOR = 1,
        ROW_MAJOR
    };

    /**
     * Implementation of matrix behaviour. Do not use this class directly. It is
     * used to implement the same inheritance pattern as for vectors, points 
     * etc. See the documentation of AbstractVectorImpl for further details.
     *
     * T scalar type.
     * D Dimension of the matrix
     * L Matrix memory layout
     * S Matrix storage
     * C Deriving subclass.
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
            template<class T, unsigned int D, MatrixLayout L, class S> class C>
    class AbstractMatrixImpl {

    public:

        /** 
         * Typedef for a matrix with "deep storage" class. Objects of this type
         * are used as return value for methods and operators that must create
         * and return new instances.
         */
        typedef C<T, D, L, T[D * D]> DeepStorageMatrix;

        /** Dtor. */
        ~AbstractMatrixImpl(void);

        /**
         * Dump the matrix to the specified stream.
         *
         * @param out The stream to dump the matrix to.
         */
        void Dump(std::ostream& out) const;

        /**
         * Get the matrix component at the specified position.
         *
         * @param row The row to return.
         * @param col The column to return.
         *
         * @return The matrix value at 'row', 'col'.
         *
         * @throws OutOfRangeException If 'row' and 'col' does not designate a 
         *         valid matrix component within [0, D[.
         */
        inline T GetAt(const int row, const int col) const {
            return (*this)(row, col);
        }

        /**
         * Get the matrix component at the specified position.
         *
         * @param row The row to return.
         * @param col The column to return.
         *
         * @return A reference to 'row', 'col'.
         *
         * @throws OutOfRangeException If 'row' and 'col' does not designate a 
         *                             valid matrix component within [0, D[.
         */
        inline T& GetAt(const int row, const int col) {
            return (*this)(row, col);
        }

        /**
         * Answer the 'col'th column vector.
         *
         * @param col The index of the column within [0, D[.
         *
         * @return The requested vector.
         *
         * @throws OutOfRangeException If 'col' is not within [0, D[.
         */
        Vector<T, D> GetColumn(const int col) const;

        /**
         * Answer the 'row'th row vector.
         *
         * @param row The index of the row within [0, D[.
         *
         * @return The requested vector.
         *
         * @throws OutOfRangeException If 'row' is not within [0, D[.
         */
        Vector<T, D> GetRow(const int row) const;

        /** 
         * Inverts the matrix.
         *
         * Note that the implementation uses a Gaussian elimination and is 
         * therefore very slow.
         *
         * @return true, if the matrix was inverted, false, if the matrix is not
         *         invertable.
         */
        bool Invert(void);

        /**
         * Answer, whether the matrix is the identity matrix.
         *
         * @return true, if the matrix is the identity matrix, false otherwise.
         */
        bool IsIdentity(void) const;

        /**
         * Answer, whether the matrix is the null matrix.
         *
         * @return true, if the matrix is the null matrix, false otherwise.
         */
        bool IsNull(void) const;

        /**
         * Answer, whether the matrix is orthogonal.
         * Note: O(n^3) matrix multiplication
         *
         * @return true, if the matrix is orthogonal, false otherwise.
         */
        bool IsOrthogonal(void) const;

        /**
         * Answer, whether the matrix is symmetric.
         *
         * @return true, if the matrix is symmetric, false otherwise.
         */
        bool IsSymmetric(void) const;

        /**
         * Direct access to the matrix components. The object remains owner
         * of the memory designated by the returned pointer.
         *
         * Note that the layout of the matrix components is dependent on the
         * template parameter L.
         *
         * @return A pointer to the matrix components.
         */
        inline const T *PeekComponents(void) const {
            return this->components;
        }

        /**
         * Direct access to the matrix components. The object remains owner
         * of the memory designated by the returned pointer.
         *
         * Note that the layout of the matrix components is dependent on the
         * template parameter L.
         *
         * @return A pointer to the matrix components.
         */
        inline T *PeekComponents(void) {
            return this->components;
        }

        /**
         * Set a new value for a matrix component.
         *
         * @param row   The row to set.
         * @param col   The column to set.
         * @param value The new value at 'row', 'col'.
         *
         * @return The previous value at the updated position.
         */
        void SetAt(const int row, const int col, const T value);

        /**
         * Make this matrix the identity matrix.
         */
        void SetIdentity(void);

        /**
         * Make this matrix a null matrix.
         */
        void SetNull(void);

        /**
         * Answer the trace (sum over the diagonal) of the matrix.
         *
         * @return The trace of the matrix.
         */
        T Trace(void) const;

        /**
         * Transposes the matrix.
         */
        void Transpose(void);

        /**
         * Componentwise add of this matrix and 'rhs'. The result is assigned
         * to this matrix.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Sp>
        AbstractMatrixImpl<T, D, L, S, C>& operator +=(
            const C<T, D, L, Sp>& rhs);

        /**
         * Componentwise add of this matrix and 'rhs'. 
         *
         * @param rhs The right hand side operand.
         *
         * @return The resulting matrix holding the sum of this one and 'rhs'.
         */
        template<class Sp>
        inline DeepStorageMatrix operator +(const C<T, D, L, Sp>& rhs) const {
            DeepStorageMatrix retval = *this;
            retval += rhs;
            return retval;
        }

        /**
         * Componentwise subtract of 'rhs' from this matrix. The result is 
         * assigned to this matrix.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Sp>
        AbstractMatrixImpl<T, D, L, S, C>& operator -=(
            const C<T, D, L, Sp>& rhs);

        /**
         * Componentwise subtract of 'rhs' from this matrix.
         *
         * @param rhs The right hand side operand.
         *
         * @return The resulting matrix holding the difference.
         */
        template<class Sp>
        inline DeepStorageMatrix operator -(const C<T, D, L, S>& rhs) const {
            DeepStorageMatrix retval = *this;
            retval -= rhs;
            return retval;
        }

        /**
         * Matrix multiplication. 'rhs' is multiplied with this matrix from
         * the right and the result is assigned to this matrix.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Sp>
        inline AbstractMatrixImpl<T, D, L, S, C>& operator *=(
                const C<T, D, L, Sp>& rhs) {
            return (*this = *this * rhs);
        }

        /**
         * Matrix multiplication. 'rhs' is multiplied with this matrix from the
         * right side.
         *
         * @param rhs The right hand side operand.
         *
         * @return The product of this matrix and 'rhs'.
         */
        template<class Sp>
        DeepStorageMatrix operator *(const C<T, D, L, Sp>& rhs) const;

        /**
         * Scalar multiplication. The result is assigned to this object.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractMatrixImpl<T, D, L, S, C>& operator *=(const T rhs);

        /**
         * Scalar multiplication.
         *
         * @param rhs The right hand side operand.
         *
         * @return 'rhs' * *this.
         */
        inline DeepStorageMatrix operator *(const T rhs) const {
            DeepStorageMatrix retval = *this;
            retval *= rhs;
            return retval;
        }

        /**
         * Componentwise division. The result is assigned to this object.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractMatrixImpl<T, D, L, S, C>& operator /=(const T rhs);

        /**
         * Componentwise division.
         *
         * @param rhs The right hand side operand.
         *
         * @return This matrix having all components divided by 'rhs'.
         */
        inline DeepStorageMatrix operator /(const T rhs) const {
            DeepStorageMatrix retval = *this;
            retval /= rhs;
            return retval;
        }

        /**
         * Multiplies this matrix with the vector 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The transformed vector.
         */
        template<class Sp>
        Vector<T, D> operator *(const AbstractVector<T, D, Sp>& rhs) const;

        /**
         * Extends 'rhs' to be a homogenous vector and multiplies the matrix 
         * with this vector. Before returning the (D - 1) dimensional result,
         * a perspective divide is performed to normalise the last component
         * to 1. This component is discarded in the result.
         *
         * @param rhs The right hand side operand.
         *
         * @return The transformed vector.
         */
        template<class Sp>
        Vector<T, D - 1> operator *(
            const AbstractVector<T, D - 1, Sp>& rhs) const;

        /**
         * Multiplies this matrix with the point 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The transformed point.
         */
        template<class Sp>
        inline Point<T, D> operator *(const AbstractPoint<T, D, Sp>& rhs) const {
            ShallowVector<T, D> v(const_cast<double *>(rhs.PeekCoordinates()));
            return Point<T, D>((*this * v).PeekComponents());
        }

        /**
         * Extends 'rhs' to be a point with homogenous coordinates and 
         * multiplies the matrix with this point. Before returning the (D - 1)
         * dimensional result, a perspective divide is performed to normalise 
         * the last component to 1. This component is discarded in the result.
         *
         * @param rhs The right hand side operand.
         *
         * @return The transformed point.
         */
        template<class Sp>
        inline Point<T, D - 1> operator *(
                const AbstractPoint<T, D - 1, Sp>& rhs) const {
            ShallowVector<T, D - 1> v(const_cast<double *>(
                rhs.PeekCoordinates()));
            return Point<T, D - 1>((*this * v).PeekComponents());
        }

        /**
         * Assignment operator.
         *
         * This operation does <b>not</b> create aliases.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractMatrixImpl<T, D, L, S, C>& operator =(const C<T, D, L, S>& rhs);

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
        inline AbstractMatrixImpl<T, D, L, S, C>& operator =(
                const C<Tp, Dp, Lp, Sp>& rhs) {
            if (static_cast<void *>(this) != static_cast<const void *>(&rhs)) {
                this->assign(rhs);
            }

            return *this;
        }

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        bool operator ==(const C<T, D, L, S>& rhs) const;

        /**
         * Test for equality of arbitrary matrices. This operation uses the
         * IsEqual function of the left hand side operand. Note that matrices 
         * with different dimensions are never equal.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this point are equal, false otherwise.
         */
        template<class Tp, unsigned int Dp, MatrixLayout Lp, class Sp>
        bool operator ==(const C<Tp, Dp, Lp, Sp>& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const C<T, D, L, S>& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Test for inequality of arbitrary matrices. See operator == for further
         * details.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this vector are not equal, false otherwise.
         */
        template<class Tp, unsigned int Dp, MatrixLayout Lp, class Sp>
        inline bool operator !=(const C<Tp, Dp, Lp, Sp>& rhs) const {
            return !(*this == rhs);
        }

        ///**
        // * Directly access the 'i'th component of the matrix. Note, that the 
        // * semantic of the 'i'th component depends on the matrix memory layout
        // * defined by the template parameter L and the matrix dimension D.
        // *
        // * @param i The index of the component within [0, D * D[.
        // *
        // * @return A reference to the 'i'th component.
        // *
        // * @throws OutOfRangeException, if 'i' is not within [0, D * D[.
        // */
        //T& operator [](const int i);

        ///**
        // * Directly access the 'i'th component of the matrix. Note, that the 
        // * semantic of the 'i'th component depends on the matrix memory layout
        // * defined by the template parameter L and the matrix dimension D.
        // *
        // * @param i The index of the component within [0, D * D[.
        // *
        // * @return Tthe 'i'th component.
        // *
        // * @throws OutOfRangeException, if 'i' is not within [0, D * D[.
        // */
        //T operator [](const int i) const;

        /**
         * Get the matrix component at the specified position.
         *
         * @param row The row to return.
         * @param col The column to return.
         *
         * @return The matrix value at 'row', 'col'.
         *
         * @throws OutOfRangeException If 'row' and 'col' does not designate a 
         *         valid matrix component within [0, D[.
         */
        T operator ()(const int row, const int col) const;

        /**
         * Get the matrix component at the specified position.
         *
         * @param row The row to return.
         * @param col The column to return.
         *
         * @return A reference to 'row', 'col'.
         *
         * @throws OutOfRangeException If 'row' and 'col' does not designate a 
         *         valid matrix component within [0, D[.
         */
        T& operator ()(const int row, const int col);

    protected:

        /**
         * Calculates the characteristic polynom of the matrix
         *
         * @return The characteristic polynom of the matrix
         *
         * @throw Exception if the calculation of the polynom fails.
         */
        Polynom<T, D> characteristicPolynom(void) const;

        /**
         * Calculates the determinant of the 2x2 matrix
         *  a00 a10
         *  a01 a11
         *
         * @param a00 A coefficient of the matrix
         * @param a10 A coefficient of the matrix
         * @param a01 A coefficient of the matrix
         * @param a11 A coefficient of the matrix
         *
         * @return The determinant of the matrix
         */
        static VISLIB_FORCEINLINE T determinant2x2(const T& a00, const T& a10,
            const T& a01, const T& a11);

        /**
         * Calculates the determinant of the 3x3 matrix
         *  a00 a10 a20
         *  a01 a11 a21
         *  a02 a12 a22
         *
         * @param a00 A coefficient of the matrix
         * @param a10 A coefficient of the matrix
         * @param a20 A coefficient of the matrix
         * @param a01 A coefficient of the matrix
         * @param a11 A coefficient of the matrix
         * @param a21 A coefficient of the matrix
         * @param a02 A coefficient of the matrix
         * @param a12 A coefficient of the matrix
         * @param a22 A coefficient of the matrix
         *
         * @return The determinant of the matrix
         */
        static VISLIB_FORCEINLINE T determinant3x3(const T& a00, const T& a10,
            const T& a20, const T& a01, const T& a11, const T& a21,
            const T& a02, const T& a12, const T& a22);

        /**
         * Calculates the determinant of the 4x4 matrix
         *  a00 a10 a20 a30
         *  a01 a11 a21 a31
         *  a02 a12 a22 a32
         *  a03 a13 a23 a33
         *
         * @param a00 A coefficient of the matrix
         * @param a10 A coefficient of the matrix
         * @param a20 A coefficient of the matrix
         * @param a30 A coefficient of the matrix
         * @param a01 A coefficient of the matrix
         * @param a11 A coefficient of the matrix
         * @param a21 A coefficient of the matrix
         * @param a31 A coefficient of the matrix
         * @param a02 A coefficient of the matrix
         * @param a12 A coefficient of the matrix
         * @param a22 A coefficient of the matrix
         * @param a32 A coefficient of the matrix
         * @param a03 A coefficient of the matrix
         * @param a13 A coefficient of the matrix
         * @param a23 A coefficient of the matrix
         * @param a33 A coefficient of the matrix
         *
         * @return The determinant of the matrix
         */
        static VISLIB_FORCEINLINE T determinant4x4(const T& a00, const T& a10,
            const T& a20, const T& a30, const T& a01, const T& a11,
            const T& a21, const T& a31, const T& a02, const T& a12,
            const T& a22, const T& a32, const T& a03, const T& a13,
            const T& a23, const T& a33);

        /**
         * Compute the index of the matrix element at 'row', 'col' depending
         * on the matrix size and layout of the instantiation. No bounds check
         * is done.
         *
         * @param row The zero-based index of the row.
         * @param col The zero-based index of the column.
         *
         * @return The index of the specified element in the components array of
         *         the current instantiation.
         */
        static inline int indexOf(const int row, const int col) {
            return (L == COLUMN_MAJOR) ? (col * D + row) : (row * D + col);
        }

        /**
         * Answer the index of 'idx' in a matrix having the opposite memory
         * layout, but the same dimension, i. e. effectively the index of the 
         * transposed element.
         *
         * @param idx The index of a component when using the memory layout 
         *            defined by the template parameter L.
         *
         * @return The index of 'idx' when using the other memory layout.
         */
        static inline int transposeIndex(const int idx) {
            return (idx % D) * D + (idx / D);
        }

        /** The number of components the matrix consists of. */
        static const unsigned int CNT_COMPONENTS;

        /** Ctor. */
        inline AbstractMatrixImpl(void) {};

        /**
         * Assign the components of 'rhs' to this matrix for the assignment
         * operator and copy ctor.
         */
        template<class Tp, unsigned int Dp, MatrixLayout Lp, class Sp>
        void assign(const C<Tp, Dp, Lp, Sp>& rhs);

        /**
         * Numerically find the eigenvalues of a SYMMETRIC matrix! Except for
         * this limitation this method behaves like 'FindEigenvalues'.
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
        unsigned int findEigenvaluesSym(T *outEigenvalues,
            Vector<T, D> *outEigenvectors, unsigned int size) const;

        /** 
         * The matrix components. Their memory layout is defined by the
         * template parameter L.
         */
        S components;

        ///**
        // *
        // */
        //template<class Tf1, unsigned int Df1, MatrixLayout Lf1, class Sf1,
        //    template<class Tf2, unsigned int Df2, MatrixLayout Lf2, class Sf2> 
        //    class Cf1>
        //friend DeepStorageMatrix operator *(const T lhs, 
        //    Cf1<Tf2, Df2, Lf2, Sf2> rhs);
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

    };


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::~AbstractMatrixImpl
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    AbstractMatrixImpl<T, D, L, S, C>::~AbstractMatrixImpl(void) {
    }


    /*
     * AbstractMatrixImpl<T, D, L, S, C>::Dump
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>   
    void AbstractMatrixImpl<T, D, L, S, C>::Dump(std::ostream& out) const {
        out << std::setiosflags(std::ios::fixed) << std::setprecision(3) 
            << std::setfill(' ');

        for (unsigned int r = 0; r < D; r++) {
            out << "{ ";

            for (unsigned int c = 0; c < D; c++) {
                out << std::setw(7) << this->components[indexOf(r, c)];
                
                if (c < D - 1) {
                    out << ", ";
                }
            }
            out << " }" << std::endl;
        }

        out << std::resetiosflags(std::ios::fixed);
    }


    /*
     * AbstractMatrixImpl<T, D, L, S, C>::GetColumn
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>   
    Vector<T, D> AbstractMatrixImpl<T, D, L, S, C>::GetColumn(
            const int col) const {
        Vector<T, D> retval;

        for (unsigned int r = 0; r < D; r++) {
            retval[r] = this->components[indexOf(r, col)];
        }

        return retval;
    }


    /*
     * AbstractMatrixImpl<T, D, L, S, C>::GetRow
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>   
    Vector<T, D> AbstractMatrixImpl<T, D, L, S, C>::GetRow(
            const int row) const {
        Vector<T, D> retval;

        for (unsigned int c = 0; c < D; c++) {
            retval[c] = this->components[indexOf(row, c)];
        }

        return retval;
    }


    /*
     * AbstractMatrixImpl<T, D, L, S, C>::Invert
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>    
    bool AbstractMatrixImpl<T, D, L, S, C>::Invert(void) {
#define A(r, c) a[(r) * 2 * D + (c)]
        double a[2 * D * D];    // input matrix for algorithm
        double f;               // Multiplication factor.
        double max;             // Row pivotising.
        unsigned int pRow;      // Pivot row.
        unsigned int s;         // Current eliminination step.
        
        /* Create double precision matrix and add identity at the right. */
        for (unsigned int r = 0; r < D; r++) {
            for (unsigned int c = 0; c < D; c++) {
                A(r, c) = static_cast<double>(this->components[indexOf(r, c)]);
            }

            for (unsigned int c = 0; c < D; c++) {
                A(r, c + D) = (r == c) ? 1.0 : 0.0;
            }
        }

        /* Gauss elimination. */
        s = 0;
        do {
            // pivotising avoids unnecessary canceling if a zero is in the 
            // diagonal and increases the precision.
            max = ::fabs(A(s, s));
            pRow = s; 
            for (unsigned int r = s + 1; r < D; r++) {
                if (::fabs(A(r, s)) > max) {
                    max = ::fabs(A(r, s));
                    pRow = r;
                }
            }

            if (max < DOUBLE_EPSILON) {
                return false;   // delete is not possible
            }

            if (pRow != s) {
                // if necessary, exchange the row
                double h;

                for (unsigned int c = s ; c < 2 * D; c++) {
                    h = A(s, c);
                    A(s, c) = A(pRow, c);
                    A(pRow, c) = h;
                }
            } 

            // eliminations row is divided by pivot-coefficient f = a[s][s]
            f = A(s, s);
            for (unsigned int c = s; c < 2 * D; c++) {
                A(s, c) /= f;
            }

            for (unsigned int r = 0; r < D; r++ ) {
                if (r != s) {
                    f = -A(r, s);
                    for (unsigned int c = s; c < 2 * D; c++) {
                        A(r, c) += f * A(s, c);
                    }
                } 
            }

            s++;
        } while (s < D);

        /* Copy identity on the right which is now inverse. */
        for (unsigned int r = 0; r < D; r++) {
            for (unsigned int c = 0; c < D; c++) { 
                this->components[indexOf(r, c)] = static_cast<T>(A(r, D + c));
            }
        }

        return true;
#undef A
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::IsIdentity
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>    
    bool AbstractMatrixImpl<T, D, L, S, C>::IsIdentity(void) const {
        for (unsigned int r = 0; r < D; r++) {
            for (unsigned int c = 0; c < D; c++) {
                if (!IsEqual(this->components[indexOf(r, c)], 
                        static_cast<T>((r == c) ? 1 : 0))) {
                    return false;
                }
            } /* end for (unsigned int c = 0; c < D; c++) */
        } /* end for (unsigned int r = 0; r < D; r++) */

        return true;
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::IsNull
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>   
    bool AbstractMatrixImpl<T, D, L, S, C>::IsNull(void) const {
        for (unsigned int c = 0; c < CNT_COMPONENTS; c++) {
            if (!IsEqual(this->components[c], static_cast<T>(0))) {
                return false;
            }
        }

        return true;
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::IsOrthogonal
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>   
    bool AbstractMatrixImpl<T, D, L, S, C>::IsOrthogonal(void) const {
        // Test: A * A^T = I

        for (unsigned int r = 0; r < D; r++) {
            for (unsigned int c = 0; c < D; c++) {
                T t = static_cast<T>(0);
                for (unsigned int i = 0; i < D; i++) {
                    t += this->components[indexOf(r, i)]
                        * this->components[indexOf(i, c)];
                }
                if (!IsEqual(t, static_cast<T>((r == c) ? 1 : 0))) {
                    return false;
                }
            }
        }

        return true;
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::IsSymmetric
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>   
    bool AbstractMatrixImpl<T, D, L, S, C>::IsSymmetric(void) const {
        for (unsigned int r = 1; r < D; r++) {
            for (unsigned int c = 0; c < r; c++) {
                if (!IsEqual(this->components[indexOf(r, c)], 
                        this->components[indexOf(c, r)])) {
                    return false;
                }
            } /* end for (unsigned int c = 0; c < D; c++) */
        } /* end for (unsigned int r = 0; r < D; r++) */
        return true;
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::SetAt
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    void AbstractMatrixImpl<T, D, L, S, C>::SetAt(const int row, const int col,
            const T value) {

        int idx = indexOf(row, col);

        if ((idx >= 0) && (idx < static_cast<int>(CNT_COMPONENTS))) {
            this->components[idx] = value;

        } else {
            throw OutOfRangeException(idx, 0, CNT_COMPONENTS - 1, __FILE__, 
                __LINE__);
        }        
    }


    /*
     * AbstractMatrixImpl<T, D, L, S, C>::SetIdentity
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    void AbstractMatrixImpl<T, D, L, S, C>::SetIdentity(void) {
        for (unsigned int r = 0; r < D; r++) {
            for (unsigned int c = 0; c < D; c++) {
                this->components[indexOf(r, c)] 
                    = static_cast<T>((r == c) ? 1 : 0);
            }
        }
    }


    /*
     * AbstractMatrixImpl<T, D, L, S, C>::SetNull
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    void AbstractMatrixImpl<T, D, L, S, C>::SetNull(void) {
        for (unsigned int i = 0; i < CNT_COMPONENTS; i++) {
            this->components[i] = static_cast<T>(0);
        }
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::Trace
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>  
    T AbstractMatrixImpl<T, D, L, S, C>::Trace(void) const {
        T retval = static_cast<T>(0);

        for (unsigned int i = 0; i < D; i++) {
            retval += this->components[indexOf(i, i)];
        }

        return retval;
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::Transpose
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>  
    void AbstractMatrixImpl<T, D, L, S, C>::Transpose(void) {
        T tmp;
        int idx1, idx2;

        for (unsigned int r = 0; r < D; r++) {
            for (unsigned int c = r; c < D; c++) {
                idx1 = indexOf(r, c);
                idx2 = indexOf(c, r);
                tmp = this->components[idx1];
                this->components[idx1] = this->components[idx2];
                this->components[idx2] = tmp;
            }
        }
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::operator +=
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>  
    template<class Sp>
    AbstractMatrixImpl<T, D, L, S, C>& 
    AbstractMatrixImpl<T, D, L, S, C>::operator +=(const C<T, D, L, Sp>& rhs) {
        for (unsigned int c = 0; c < CNT_COMPONENTS; c++) {
            this->components[c] += rhs.PeekComponents()[c];
        }

        return *this;
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::operator -=
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>  
    template<class Sp>
    AbstractMatrixImpl<T, D, L, S, C>& 
    AbstractMatrixImpl<T, D, L, S, C>::operator -=(const C<T, D, L, Sp>& rhs) {
        for (unsigned int c = 0; c < CNT_COMPONENTS; c++) {
            this->components[c] -= rhs.PeekComponents()[c];
        }

        return *this;
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::operator *
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>  
    template<class Sp>
    typename AbstractMatrixImpl<T, D, L, S, C>::DeepStorageMatrix
    AbstractMatrixImpl<T, D, L, S, C>::operator *(
            const C<T, D, L, Sp>& rhs) const {
        DeepStorageMatrix retval; 
        const T *rhsComps = rhs.PeekComponents();
        T *retvalComps = retval.PeekComponents();

        // TODO: Not so nice ...
        retval.SetNull();

        for (unsigned int r = 0; r < D; r++) {
            for (unsigned int c = 0; c < D; c++) {
                for (unsigned int i = 0; i < D; i++) {
                    retvalComps[indexOf(r, c)] 
                        += this->components[indexOf(r, i)]
                        * rhsComps[indexOf(i, c)];
                }
            }
        }

        return retval;
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::operator *=
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>  
    AbstractMatrixImpl<T, D, L, S, C>& 
    AbstractMatrixImpl<T, D, L, S, C>::operator *=(const T rhs) {
        for (unsigned int c = 0; c < CNT_COMPONENTS; c++) {
            this->components[c] *= rhs;
        }

        return *this;
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::operator /=
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>  
    AbstractMatrixImpl<T, D, L, S, C>& 
    AbstractMatrixImpl<T, D, L, S, C>::operator /=(const T rhs) {
        for (unsigned int c = 0; c < CNT_COMPONENTS; c++) {
            this->components[c] /= rhs;
        }

        return *this;
    }


    /* 
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::operator *
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>  
    template<class Sp>
    Vector<T, D> AbstractMatrixImpl<T, D, L, S, C>::operator *(
            const AbstractVector<T, D, Sp>& rhs) const {
        Vector<T, D> retval;
        ASSERT(retval.IsNull());

        for (unsigned int r = 0; r < D; r++) {
            for (unsigned int c = 0; c < D; c++) {
                retval[r] += this->components[indexOf(r, c)] * rhs[c];
            }
        }

        return retval;
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::operator *
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    template<class Sp>
    Vector<T, D - 1> AbstractMatrixImpl<T, D, L, S, C>::operator *(
            const AbstractVector<T, D - 1, Sp>& rhs) const {
        Vector<T, D - 1> retval;
        ASSERT(retval.IsNull());

        /* Compute w-component of result. */
        T w = this->components[indexOf(D - 1, D - 1)];
        for (unsigned int i = 0; i < D - 1; i++) {
            w += this->components[indexOf(D - 1, i)] * rhs[i];
        }

        /* Compute normal product of D - 1 vector. */
        for (unsigned int r = 0; r < D - 1; r++) {
            for (unsigned int c = 0; c < D - 1; c++) {
                retval[r] += this->components[indexOf(r, c)] * rhs[c];
            }
        }

        /* Perspective divide. */
        return (retval / w);
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::operator =
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    AbstractMatrixImpl<T, D, L, S, C>& 
    AbstractMatrixImpl<T, D, L, S, C>::operator =(const C<T, D, L, S>& rhs) {

        if (this != &rhs) {
            ::memcpy(this->components, rhs.components, CNT_COMPONENTS 
                * sizeof(T));
        }

        return *this;
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::operator ==
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    bool AbstractMatrixImpl<T, D, L, S, C>::operator ==(
            const C<T, D, L, S>& rhs) const {

        for (unsigned int d = 0; d < CNT_COMPONENTS; d++) {
            if (!IsEqual<T>(this->components[d], rhs.components[d])) {
                return false;
            }
        }

        return true;
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::operator ==
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    template<class Tp, unsigned int Dp, MatrixLayout Lp, class Sp>
    bool AbstractMatrixImpl<T, D, L, S, C>::operator ==(
            const C<Tp, Dp, Lp, Sp>& rhs) const {
        
        if (D != Dp) {
            return false;
        }

        for (unsigned int d = 0; d < CNT_COMPONENTS; d++) {
            if (!IsEqual<T>(this->components[d], (L == Lp) ? rhs.components[d] 
                    : rhs.components[transposeIndex(d)])) {
                return false;
            }
        }

        return true;        
    }


    ///*
    // * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::operator []
    // */
    //template<class T, unsigned int D, MatrixLayout L, class S,
    //    template<class T, unsigned int D, MatrixLayout L, class S> class C>
    //T& AbstractMatrixImpl<T, D, L, S, C>::operator [](const int i) {

    //    if ((i >= 0) && (i < static_cast<int>(CNT_COMPONENTS))) {
    //        return this->components[i];
    //    } else {
    //        throw OutOfRangeException(i, 0, D - 1, __FILE__, __LINE__);
    //    }
    //}


    ///*
    // * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::operator []
    // */
    //template<class T, unsigned int D, MatrixLayout L, class S,
    //    template<class T, unsigned int D, MatrixLayout L, class S> class C>
    //T AbstractMatrixImpl<T, D, L, S, C>::operator [](const int i) const {

    //    if ((i >= 0) && (i < static_cast<int>(CNT_COMPONENTS))) {
    //        return this->components[i];
    //    } else {
    //        throw OutOfRangeException(i, 0, D - 1, __FILE__, __LINE__);
    //    }
    //}

    /*
     * AbstractMatrixImpl<T, D, L, S, C>::operator ()
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    T AbstractMatrixImpl<T, D, L, S, C>::operator ()(const int row, 
            const int col) const {

        int idx = indexOf(row, col);

        if ((idx >= 0) && (idx < static_cast<int>(CNT_COMPONENTS))) {
            return this->components[idx];
        } else {
            throw OutOfRangeException(idx, 0, CNT_COMPONENTS - 1, __FILE__, 
                __LINE__);
        }
    }


    /*
     * AbstractMatrixImpl<T, D, L, S, C>::operator ()
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    T& AbstractMatrixImpl<T, D, L, S, C>::operator ()(const int row, 
            const int col) {
        int idx = indexOf(row, col);

        if ((idx >= 0) && (idx < static_cast<int>(CNT_COMPONENTS))) {
            return this->components[idx];
        } else {
            throw OutOfRangeException(idx, 0, CNT_COMPONENTS - 1, __FILE__, 
                __LINE__);
        }
    }


    /*
     * AbstractMatrixImpl<T, D, L, S, C>::CNT_COMPONENTS
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    const unsigned int AbstractMatrixImpl<T, D, L, S, C>::CNT_COMPONENTS 
        = D * D;


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::assign
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    template<class Tp, unsigned int Dp, MatrixLayout Lp, class Sp>
    void AbstractMatrixImpl<T, D, L, S, C>::assign(
            const C<Tp, Dp, Lp, Sp>& rhs) {

        for (unsigned int r = 0; r < D; r++) {
            for (unsigned int c = 0; c < D; c++) {
                if ((r < Dp) && (c < Dp)) {
                    this->components[indexOf(r, c)] 
                        = static_cast<T>(rhs.GetAt(r, c));
                } else if (r == c) {
                    this->components[indexOf(r, c)] = static_cast<T>(1);
                } else {
                    this->components[indexOf(r, c)] = static_cast<T>(0);
                } /* end if ((r < Dp) && (c < Dp)) */
            } /* end for (unsigned int c = 0; c < D; c++) */
        } /* end for (unsigned int r = 0; r < D; r++) */
    }


    /*
     * AbstractMatrixImpl<T, D, L, S, C>::characteristicPolynom
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    Polynom<T, D>
    AbstractMatrixImpl<T, D, L, S, C>::characteristicPolynom(void) const {
        // method of Faddejew-Leverrier
        // http://de.wikipedia.org/wiki/Algorithmus_von_Faddejew-Leverrier
        Polynom<T, D> c;
        DeepStorageMatrix B[2];

        B[0].SetNull();
        c[D] = static_cast<T>(1);
        B[1].SetNull(); // B1 = A * B0 = A * 0 = 0

        for (unsigned int k = 1; k <= D; k++) {
            unsigned int a = k % 2;
            unsigned int b = 1 - a;

            for (unsigned int i = 0; i < D; i++) {
                B[a](i, i) += c[D - k + 1];
            }

            B[b] = (*this);
            B[b] *= B[a];

            c[D - k] = B[b].Trace() * static_cast<T>(-1) / static_cast<T>(k);
        }

        B[1 - (D % 2)] = (*this);
        B[0] *= B[1];
        for (unsigned int i = 0; i < D; i++) {
            B[0](i, i) += c[0];
        }

        if (!B[0].IsNull()) {
            throw Exception("Characteristic polynom calculation failed",
                __FILE__, __LINE__);
        }

        return c;
    }


    /*
     * AbstractMatrixImpl<T, D, L, S, C>::determinant2x2
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    T AbstractMatrixImpl<T, D, L, S, C>::determinant2x2(const T& a00,
            const T& a10, const T& a01, const T& a11) {
        return a00 * a11 - a01 * a10;
    }


    /*
     * AbstractMatrixImpl<T, D, L, S, C>::determinant3x3
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    T AbstractMatrixImpl<T, D, L, S, C>::determinant3x3(const T& a00,
            const T& a10, const T& a20, const T& a01, const T& a11,
            const T& a21, const T& a02, const T& a12, const T& a22) {
        // rule of sarrus
        return a00 * a11 * a22 + a01 * a12 * a20 + a02 * a10 * a21
            - a02 * a11 * a20 - a01 * a10 * a22 - a00 * a12 * a21;
    }


    /*
     * AbstractMatrixImpl<T, D, L, S, C>::determinant4x4
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    T AbstractMatrixImpl<T, D, L, S, C>::determinant4x4(const T& a00,
            const T& a10, const T& a20, const T& a30, const T& a01,
            const T& a11, const T& a21, const T& a31, const T& a02,
            const T& a12, const T& a22, const T& a32, const T& a03,
            const T& a13, const T& a23, const T& a33) {
        // Method: 1 x laplace + sarrus
        return a02
            * determinant3x3(a10, a20, a30, a11, a21, a31, a13, a23, a33)
            - a12
            * determinant3x3(a00, a20, a30, a01, a21, a31, a03, a23, a33)
            + a22
            * determinant3x3(a00, a10, a30, a01, a11, a31, a03, a13, a33)
            - a32
            * determinant3x3(a00, a10, a20, a01, a11, a21, a03, a13, a23);
    }


    /*
     * AbstractMatrixImpl<T, D, L, S, C>::findEigenvaluesSym
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    unsigned int AbstractMatrixImpl<T, D, L, S, C>::findEigenvaluesSym(T *outEigenvalues,
            Vector<T, D> *outEigenvectors, unsigned int size) const {

        if (((outEigenvalues == NULL) && (outEigenvectors == NULL))
                || (size == 0)) return 0;

#define A(r, c) a[(r) * D + (c)]
        double a[D * D];                    // input matrix for algorithm
        double d[D];                        // diagonal elements
        double e[D];                        // off-diagonal elements

        for (unsigned int r = 0; r < D; r++) {
            for (unsigned int c = 0; c < D; c++) {
                A(r, c) = static_cast<double>(this->components[indexOf(r, c)]);
            }
        }

        // 1. Householder reduction.
        int l, k, j, i;
        double scale, hh, h, g, f;

        for (i = D; i >= 2; i--) {
            l = i - 1;
            h = scale = 0.0;
            if (l > 1) {
                for (k = 1; k <= l; k++) {
                    scale += ::fabs(A(i - 1, k - 1));
                }
                if (IsEqual(scale, 0.0)) {
                    e[i - 1] = A(i - 1, l - 1);
                } else {
                    for (k = 1; k <= l; k++) {
                        A(i - 1, k - 1) /= scale;
                        h += A(i - 1, k - 1) * A(i - 1, k - 1);
                    }
                    f = A(i - 1, l - 1);
                    g = ((f >= 0.0) ? -::sqrt(h) : ::sqrt(h));
                    e[i - 1] = scale * g;
                    h -= f * g;
                    A(i - 1, l - 1) = f - g;
                    f = 0.0;
                    for (j = 1; j <= l; j++) {
                        A(j - 1, i - 1) = A(i - 1, j - 1) / h;
                        g = 0.0;
                        for (k = 1; k <= j; k++) {
                            g += A(j - 1, k - 1) * A(i - 1, k - 1);
                        }
                        for (k = j + 1; k <= l; k++) {
                            g += A(k - 1, j - 1) * A(i - 1, k - 1);
                        }
                        e[j - 1] = g / h;
                        f += e[j - 1] * A(i - 1, j - 1);
                    }
                    hh = f / (h + h);
                    for (j = 1; j <= l; j++) {
                        f = A(i - 1, j - 1);
                        e[j - 1] = g = e[j - 1] - hh * f;
                        for (k = 1; k <= j; k++) {
                            A(j - 1, k - 1) -= (f * e[k - 1]
                                + g * A(i - 1, k - 1));
                        }
                    }
                }
            } else {
                e[i - 1] = A(i - 1, l - 1);
            }
            d[i - 1] = h;
        }
        d[0] = 0.0;
        e[0] = 0.0;
        for (i = 1; i <= static_cast<int>(D); i++) {
            l = i - 1;
            if (!IsEqual(d[i - 1], 0.0)) {
                for (j = 1; j <= l ; j++) {
                    g = 0.0;
                    for (k = 1; k <= l; k++) {
                        g += A(i - 1, k - 1) * A(k - 1, j - 1);
                    }
                    for (k = 1; k <= l; k++) {
                        A(k - 1, j - 1) -= g * A(k - 1, i - 1);
                    }
                }
            }
            d[i - 1] = A(i - 1, i - 1);
            A(i - 1, i - 1) = 1.0;
            for (j = 1; j <= l; j++) {
                A(j - 1, i - 1) = A(i - 1, j - 1) = 0.0;
            }
        }

        // 2. Calculation von eigenvalues and eigenvectors (QL algorithm)
#ifdef SIGN
#error SIGN macro already in use! Code rewrite required!
#endif
#define SIGN(a, b) ((b) >= 0.0 ? ::fabs(a) : -::fabs(a))

        int m, iter;
        double s, r, p, dd, c, b;
        const int MAX_ITER = 30;

        for (i = 2; i <= static_cast<int>(D); i++) {
            e[i - 2] = e[i - 1];
        }
        e[D - 1] = 0.0;

        for (l = 1; l <= static_cast<int>(D); l++) {
            iter = 0;
            do {
                for (m = l; m <= static_cast<int>(D) - 1; m++) {
                    dd = ::fabs(d[m - 1]) + ::fabs(d[m - 1 + 1]);
                    if (IsEqual(::fabs(e[m - 1]) + dd, dd)) break;
                }
                if (m != l) {
                    if (iter++ == MAX_ITER) {
                        throw vislib::Exception(
                            "Too many iterations in FindEigenvalues",
                            __FILE__, __LINE__);
                    }
                    g = (d[l - 1 + 1] - d[l - 1]) / (2.0 * e[l - 1]);
                    r = pythag(g, 1.0);
                    g = d[m - 1] - d[l - 1] + e [l - 1] / (g + SIGN(r, g));
                    s = c = 1.0;
                    p = 0.0;
                    for (i = m - 1; i >= l ; i--) {
                        f = s * e[i - 1];
                        b = c * e[i - 1];
                        e[i - 1 + 1] = r = pythag(f, g);
                        if (IsEqual(r, 0.0)) {
                            d[i - 1 + 1] -= p;
                            e[m - 1] = 0.0;
                            break;
                        }
                        s = f / r;
                        c = g / r;
                        g = d[i - 1 + 1] - p;
                        r = (d[i - 1] - g) * s + 2.0 * c * b;
                        d[i - 1 + 1] = g + (p = s * r);
                        g = c * r - b;
                        for (k = 1; k <= static_cast<int>(D); k++) {
                            f = A(k - 1, i - 1 + 1);
                            A(k - 1, i - 1 + 1) = s * A(k - 1, i - 1) + c * f;
                            A(k - 1, i - 1) = c * A(k - 1, i - 1) - s * f;
                        }
                    }
                    if (IsEqual(r, 0.0) && (i >= l)) continue;
                    d[l - 1] -= p;
                    e[l - 1] = g;
                    e[m - 1] = 0.0;
                }
            } while (m != l);
        }
#undef SIGN

        // 3. output
        if (outEigenvalues != NULL){
            for (i = 0; i < static_cast<int>(Min(D, size)); i++) {
                outEigenvalues[i] = static_cast<T>(d[i]);
            }
        }
        if (outEigenvectors != NULL){
            for (i = 0; i < static_cast<int>(Min(D, size)); i++) {
                for (j = 0; j < static_cast<int>(D); j++) {
                    outEigenvectors[i][j] = static_cast<T>(A(i, j));
                }
            }
        }

#undef A
        return Min(D, size);
    }

} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTMATRIXIMPL_H_INCLUDED */
