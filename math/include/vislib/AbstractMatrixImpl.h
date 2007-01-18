/*
 * AbstractMatrixImpl.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTMATRIXIMPL_H_INCLUDED
#define VISLIB_ABSTRACTMATRIXIMPL_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/assert.h"
#include "vislib/mathfunctions.h"
#include "vislib/memutils.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Point.h"
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
         * Answer the determinant of this matrix.
         *
         * Note that the implementation uses a Gaussian elimination and is 
         * therefore very slow.
         *
         * @return The determinant of the matrix.
         */
        T Determinant(void) const;

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
            DeepStorageMatrix retval = rhs;
            retval += *this;
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
            return Point<T, D>(v.PeekComponents());
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
            return Point<T, D - 1>(v.PeekComponents());
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

    };


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::~AbstractMatrixImpl
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    AbstractMatrixImpl<T, D, L, S, C>::~AbstractMatrixImpl(void) {
    }


    /*
     * vislib::math::AbstractMatrixImpl<T, D, L, S, C>::Determinant
     */
    template<class T, unsigned int D, MatrixLayout L, class S,
        template<class T, unsigned int D, MatrixLayout L, class S> class C>
    T AbstractMatrixImpl<T, D, L, S, C>::Determinant(void) const {
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
                A(r, c) = static_cast<double>(this->components[indexOf(r, c)]);
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
                if (r == c) {
                    if (!IsEqual(this->components[indexOf(r, c)], 
                            static_cast<T>(1))) {
                        return false;
                    }
                } else {
                    if (!IsEqual(this->components[indexOf(r, c)], 
                            static_cast<T>(0))) {
                        return false;
                    }
                } /* end if (r == c) */
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
                retval[r] += this->components[indexOf(r, c)] * rhs[r];
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
                retval[r] += this->components[indexOf(r, c)] * rhs[r];
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
    
} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_ABSTRACTMATRIXIMPL_H_INCLUDED */
