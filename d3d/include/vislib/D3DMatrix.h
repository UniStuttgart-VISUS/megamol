/*
 * D3DMatrix.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_D3DMATRIX_H_INCLUDED
#define VISLIB_D3DMATRIX_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <d3d9.h>
#include <d3dx9math.h>

#include "vislib/AbstractMatrix.h"
#include "vislib/assert.h"


namespace vislib {
namespace graphics {
namespace d3d {


    /**
     * Matrix specialisation that uses a D3DXMATRIX as storage class.
     *
     * Implementation note: We require a D3DXMATRIX rather than a D3DMATRIX
     * because of the cast to FLOAT.
     */
    class D3DMatrix : public math::AbstractMatrix<FLOAT, 4, 
            math::ROW_MAJOR, D3DXMATRIX> {

    private:

        /** Make storage class available. */
        typedef D3DXMATRIX S;

        /** Make 'T' available. */
        typedef FLOAT T;

    public:

        /** 
         * Create the identity matrix.
         */
        inline D3DMatrix(void) : Super() {
            this->SetIdentity();
        }

        /**
         * Create a matrix using the specified components.
         *
         * @param components (D * D) components of the matrix. This must not be
         *                   NULL and according to the matrix layout L.
         */
        inline D3DMatrix(const T *components) : Super() {
            ASSERT(components != NULL);
            ::memcpy(this->components.m, components, Super::CNT_COMPONENTS
                * sizeof(T));
        }

        /**
         * Create which has the same value for all components.
         *
         * @param value The initial value of all components.
         */
        explicit D3DMatrix(const T& value);

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
        D3DMatrix(const T& m11, const T& m12, const T& m13, const T& m14, 
            const T& m21, const T& m22, const T& m23, const T& m24, 
            const T& m31, const T& m32, const T& m33, const T& m34, 
            const T& m41, const T& m42, const T& m43, const T& m44);

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline D3DMatrix(const D3DMatrix& rhs) : Super() {
            ::memcpy(this->components.m, rhs.components.m,
                Super::CNT_COMPONENTS * sizeof(T));
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        template<class Tp, unsigned int Dp, math::MatrixLayout Lp, class Sp>
        inline D3DMatrix(const math::AbstractMatrix<Tp, Dp, Lp, Sp>& rhs) 
                : Super() {
            this->assign(rhs);
        }

        /**
         * Create a matrix that represents the rotation of the quaternion
         * 'rhs'.
         *
         * @param rhs The quaterion to be converted.
         */
        template<class Tp, class Sp>
        explicit inline D3DMatrix(const math::AbstractQuaternion<Tp, Sp>& rhs) {
            // Implementation note: No superclass ctor called
            // Implementation note: quaternion assign does not check for
            // self assignment, so this works here.
            Super::operator =(rhs);
        }

        /** Dtor. */
        ~D3DMatrix(void);

        /**
         * Assignment operator.
         *
         * This operation does <b>not</b> create aliases.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline D3DMatrix& operator =(const D3DMatrix& rhs) {
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
        template<class Tp, unsigned int Dp, math::MatrixLayout Lp, class Sp>
        inline D3DMatrix& operator =(const math::AbstractMatrix<Tp, Dp, Lp, Sp>&
                rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Provide access to the underlying Direct3D matrix structure.
         *
         * @return A pointer to the underlying Direct3D matrix structure.
         */
        inline operator D3DXMATRIX&(void) {
            return this->components;
        }

        /**
         * Provide access to the underlying Direct3D matrix structure.
         *
         * @return A pointer to the underlying Direct3D matrix structure.
         */
        inline operator const D3DXMATRIX&(void) const {
            return this->components;
        }

        /**
         * Provide access to the underlying Direct3D matrix structure.
         *
         * @return A pointer to the underlying Direct3D matrix structure.
         */
        inline operator D3DXMATRIX *(void) {
            return &this->components;
        }

        /**
         * Provide access to the underlying Direct3D matrix structure.
         *
         * @return A pointer to the underlying Direct3D matrix structure.
         */
        inline operator const D3DXMATRIX *(void) const {
            return &this->components;
        }

    protected:

        /** A typedef for the super class. */
        typedef math::AbstractMatrix<T, 4, math::ROW_MAJOR, S> Super;

        /** The number of dimensions. */
        static const unsigned int D;
    };
    
} /* end namespace d3d */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_D3DMATRIX_H_INCLUDED */
