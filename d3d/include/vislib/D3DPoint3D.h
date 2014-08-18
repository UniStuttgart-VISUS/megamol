/*
 * D3DPoint3D.h
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_D3DPOINT3D_H_INCLUDED
#define VISLIB_D3DPOINT3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <d3d9.h>
#include <d3dx9math.h>

#include "vislib/AbstractPoint.h"


namespace vislib {
namespace graphics {
namespace d3d {


    /**
     * Point specialisation using a D3DXVECTOR3 as storage class.
     */
    class D3DPoint3D 
                : public vislib::math::AbstractPoint<FLOAT, 3, D3DXVECTOR3> {

    private:

        /** Make storage class available. */
        typedef D3DXVECTOR3 S;

        /** Make 'T' available. */
        typedef FLOAT T;

    public:

        /**
         * Create a point in the origin.
         */
        D3DPoint3D(void);

        /**
         * Create a new point.
         *
         * @param x The x-component.
         * @param y The y-component.
         * @param z The z-component.
         */
        inline D3DPoint3D(const T& x, const T& y, const T& z) : Super() {
            this->coordinates[0] = x;
            this->coordinates[1] = y;
            this->coordinates[2] = z;
        }

        /**
         * Create a new point initialised with 'coordinates'. 'coordinates' must
         * not be a NULL pointer. 
         *
         * @param coordinates The initial point coordinates.
         */
        explicit inline D3DPoint3D(const T *coordinates) : Super() {
            ASSERT(coordinates != NULL);
            ::memcpy(this->coordinates, coordinates, D * sizeof(T));
        }

        /**
         * Create a new point from a D3DXVECTOR3.
         *
         * @param rhs The object to be cloned.
         */
        explicit inline D3DPoint3D(const D3DXVECTOR3& rhs) : Super() {
            ::memcpy(this->coordinates, rhs, sizeof(D3DXVECTOR3));
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline D3DPoint3D(const D3DPoint3D& rhs) : Super() {
            ::memcpy(this->coordinates, rhs.coordinates, D * sizeof(T));
        }

        /**
         * Create a copy of 'rhs'. This ctor allows for arbitrary point to
         * point conversions.
         *
         * @param rhs The vector to be cloned.
         */
        template<class Tp, unsigned int Dp, class Sp>
        D3DPoint3D(const AbstractPoint<Tp, Dp, Sp>& rhs);

        /** Dtor. */
        ~D3DPoint3D(void);

        /**
         * Assignment operator.
         *
         * This operation does <b>not</b> create aliases.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline D3DPoint3D& operator =(const D3DPoint3D& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Assigment for arbitrary points. A valid static_cast between T and Tp
         * is a precondition for instantiating this template.
         *
         * This operation does <b>not</b> create aliases. 
         *
         * If the two operands have different dimensions, the behaviour is as 
         * follows: If the left hand side operand has lower dimension, the 
         * highest (Dp - D) dimensions are discarded. If the left hand side
         * operand has higher dimension, the missing dimensions are filled with 
         * zero coordinates.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        template<class Tp, unsigned int Dp, class Sp>
        inline D3DPoint3D& operator =(const AbstractPoint<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Provide access to the underlying Direct3D vector structure.
         *
         * @return A pointer to the underlying Direct3D vector structure.
         */
        inline operator D3DXVECTOR3&(void) {
            return this->coordinates;
        }

        /**
         * Provide access to the underlying Direct3D vector structure.
         *
         * @return A pointer to the underlying Direct3D vector structure.
         */
        inline operator const D3DXVECTOR3&(void) const {
            return this->coordinates;
        }

        /**
         * Provide access to the underlying Direct3D vector structure.
         *
         * @return A pointer to the underlying Direct3D vector structure.
         */
        inline operator D3DXVECTOR3 *(void) {
            return &this->coordinates;
        }

        /**
         * Provide access to the underlying Direct3D vector structure.
         *
         * @return A pointer to the underlying Direct3D vector structure.
         */
        inline operator const D3DXVECTOR3 *(void) const {
            return &this->coordinates;
        }

    protected:

        /** A typedef for the super class. */
        typedef vislib::math::AbstractPoint<FLOAT, 3, D3DXVECTOR3> Super;

        /** The number of dimensions. */
        static const unsigned int D;

    };


    /*
     * vislib::math::D3DPoint3D::D3DPoint3D
     */
    template<class Tp, unsigned int Dp, class Sp>
    D3DPoint3D::D3DPoint3D(const AbstractPoint<Tp, Dp, Sp>& rhs) : Super() {
        for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
            this->coordinates[d] = static_cast<T>(rhs[d]);
        }
        for (unsigned int d = Dp; d < D; d++) {
            this->coordinates[d] = static_cast<T>(0);
        }
    }
    
} /* end namespace d3d */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_D3DPOINT3D_H_INCLUDED */

