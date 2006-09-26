/*
 * Vector3D.h  14.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_VECTOR3D_H_INCLUDED
#define VISLIB_VECTOR3D_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/AbstractVector3D.h"
#include "vislib/Vector.h"


namespace vislib {
namespace math {

    /**
     * Specialisation for a three-dimensional vector. See Vector for additional
     * remarks.
     */
    template<class T> 
    class Vector3D : public AbstractVector3D<T, T[3]> {

    public:

        /** A typedef for the super class. */
        typedef AbstractVector3D<T, T[3]> Super;

        /**
         * Create a null vector.
         */
        inline Vector3D(void) {
            this->components[0] = static_cast<T>(0);
            this->components[1] = static_cast<T>(0);
            this->components[2] = static_cast<T>(0);
        }

        /**
         * Create a new vector initialised with 'components'. 'components' must
         * not be a NULL pointer. 
         *
         * @param components The initial vector components.
         */
        explicit inline Vector3D(const T *components) {
            ASSERT(components != NULL);
            ::memcpy(this->components, components, 3 * sizeof(T));
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Vector3D(const Vector3D& rhs) {
            ::memcpy(this->components, rhs.components, 3 * sizeof(T));
        }

        /**
         * Create a copy of 'vector'. This ctor allows for arbitrary vector to
         * vector conversions.
         *
         * @param rhs The vector to be cloned.
         */
        template<class Tp, unsigned int Dp, class Sp>
        inline Vector3D(const AbstractVector<Tp, Dp, Sp>& rhs) {
            this->components[0] = (Dp < 1) ? static_cast<T>(0) 
                                           : static_cast<T>(rhs[0]);
            this->components[1] = (Dp < 2) ? static_cast<T>(0) 
                                           : static_cast<T>(rhs[1]);
            this->components[2] = (Dp < 3) ? static_cast<T>(0) 
                                           : static_cast<T>(rhs[2]);
        }

        /**
         * Create a new vector.
         *
         * @param x The x-component of the new vector.
         * @param y The y-component of the new vector.
         * @param z The z-component fo the new vector.
         */
        inline Vector3D(const T& x, const T& y, const T& z) {
            this->components[0] = x;
            this->components[1] = y;
            this->components[2] = z;
        }

        /** Dtor. */
        ~Vector3D(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline Vector3D& operator =(const Vector3D& rhs) {
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
        inline Vector3D& operator =(const AbstractVector<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }
    };

    /*
     * vislib::math::Vector3D<T>::~Vector3D
     */
    template<class T> Vector3D<T>::~Vector3D(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_VECTOR3D_H_INCLUDED */
