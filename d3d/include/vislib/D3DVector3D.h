/*
 * D3DVector3D.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_D3DVECTOR3D_H_INCLUDED
#define VISLIB_D3DVECTOR3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <d3d9.h>
#include <d3dx9math.h>

#include "vislib/AbstractVector.h"


namespace vislib {
namespace graphics {
namespace d3d {


    /**
     * Vector specialisation using a D3DXVECTOR3 as storage class.
     */
    class D3DVector3D 
            : public vislib::math::AbstractVector<FLOAT, 3, D3DXVECTOR3> {


    private:

        /** Make storage class available. */
        typedef D3DXVECTOR3 S;

        /** Make 'T' available. */
        typedef FLOAT T;

    public:

        /**
         * Create a null vector.
         */
        D3DVector3D(void);

        /**
         * Create a new vector.
         *
         * @param x The x-component.
         * @param y The y-component.
         * @param z The z-component.
         */
        inline D3DVector3D(const T& x, const T& y, const T& z) : Super() {
            this->components[0] = x;
            this->components[1] = y;
            this->components[2] = z;
        }

        /**
         * Create a new vector initialised with 'components'. 'components' must
         * not be a NULL pointer. 
         *
         * @param components The initial vector components.
         */
        explicit inline D3DVector3D(const T *components) : Super() {
            ASSERT(components != NULL);
            ::memcpy(this->components, components, D * sizeof(T));
        }

        /**
         * Create a new vector from a D3DXVECTOR3.
         *
         * @param rhs The object to be cloned.
         */
        explicit inline D3DVector3D(const D3DXVECTOR3& rhs) : Super() {
            ::memcpy(this->components, rhs, sizeof(D3DXVECTOR3));
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline D3DVector3D(const D3DVector3D& rhs) : Super() {
            ::memcpy(this->components, rhs.components, D * sizeof(T));
        }

        /**
         * Create a copy of 'rhs'. This ctor allows for arbitrary vector to
         * vector conversions.
         *
         * @param rhs The vector to be cloned.
         */
        template<class Tp, unsigned int Dp, class Sp>
        D3DVector3D(const AbstractVector<Tp, Dp, Sp>& rhs);

        /** Dtor. */
        ~D3DVector3D(void);

        /**
         * Assignment operator.
         *
         * This operation does <b>not</b> create aliases.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline D3DVector3D& operator =(const D3DVector3D& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Assigment for arbitrary vectors. A valid static_cast between T and Tp
         * is a precondition for instantiating this template.
         *
         * This operation does <b>not</b> create aliases. 
         *
         * If the two operands have different dimensions, the behaviour is as 
         * follows: If the left hand side operand has lower dimension, the 
         * highest (Dp - D) dimensions are discarded. If the left hand side
         * operand has higher dimension, the missing dimensions are filled with 
         * zero components.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        template<class Tp, unsigned int Dp, class Sp>
        inline D3DVector3D& operator =(const AbstractVector<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Provide access to the underlying Direct3D vector structure.
         *
         * @return A pointer to the underlying Direct3D vector structure.
         */
        inline operator D3DXVECTOR3&(void) {
            return this->components;
        }

        /**
         * Provide access to the underlying Direct3D vector structure.
         *
         * @return A pointer to the underlying Direct3D vector structure.
         */
        inline operator const D3DXVECTOR3&(void) const {
            return this->components;
        }

        /**
         * Provide access to the underlying Direct3D vector structure.
         *
         * @return A pointer to the underlying Direct3D vector structure.
         */
        inline operator D3DXVECTOR3 *(void) {
            return &this->components;
        }

        /**
         * Provide access to the underlying Direct3D vector structure.
         *
         * @return A pointer to the underlying Direct3D vector structure.
         */
        inline operator const D3DXVECTOR3 *(void) const {
            return &this->components;
        }

    protected:

        /** A typedef for the super class. */
        typedef vislib::math::AbstractVector<FLOAT, 3, D3DXVECTOR3> Super;

        /** The number of dimensions. */
        static const unsigned int D;

    };


    /*
     * vislib::math::D3DVector3D::D3DVector3D
     */
    template<class Tp, unsigned int Dp, class Sp>
    D3DVector3D::D3DVector3D(const AbstractVector<Tp, Dp, Sp>& rhs) : Super() {
        for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
            this->components[d] = static_cast<T>(rhs[d]);
        }
        for (unsigned int d = Dp; d < D; d++) {
            this->components[d] = static_cast<T>(0);
        }
    }
    
} /* end namespace d3d */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_D3DVECTOR3D_H_INCLUDED */
