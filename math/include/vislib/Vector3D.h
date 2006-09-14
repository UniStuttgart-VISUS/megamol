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


#include "vislib/Vector.h"


namespace vislib {
namespace math {

    /**
     * Specialisation for a three-dimensional vector. See Vector for additional
     * remarks.
     */
    template<class T,class E = EqualFunc<T>, 
            template<class, unsigned int> class S = DeepStorageClass> 
    class Vector3D : public Vector<T, 3, E, S> {

    public:

        inline Vector3D(void) : Vector<T, 3, E, S>() {}

        explicit inline Vector3D(const T *components) 
            : Vector<T, 3, E, S>(components) {}


        inline Vector3D(const Vector3D& rhs)
            : Vector<T, 3, E, S>(rhs) {}

        template<class Tp, unsigned int Dp, class Ep,
            template<class, unsigned int> class Sp>
        inline Vector3D(const Vector<Tp, Dp, Ep, Sp>& vector)
            : Vector<T, 3, E, S>(vector) {}

        inline Vector3D(const T& x, const T& y, const T& z) {
            this->components[0] = x;
            this->components[1] = y;
            this->components[2] = z;
        }

        /** Dtor. */
        virtual ~Vector3D(void);

        /**
         * Answer the cross product of this vector and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The cross product.
         */
        Vector3D Cross(const Vector3D& rhs) const;

        /**
         * Answer the x-component of the vector.
         *
         * @return The x-component of the vector.
         */
        inline const T& GetX(void) const {
            return this->components[0];
        }

        /**
         * Answer the y-component of the vector.
         *
         * @return The y-component of the vector.
         */
        inline const T& GetY(void) const {
            return this->components[1];
        }

        /**
         * Answer the z-component of the vector.
         *
         * @return The z-component of the vector.
         */
        inline const T& GetZ(void) const {
            return this->components[2];
        }

        /**
         * Set the x-component of the vector.
         *
         * @param x The new x-component.
         */
        inline void SetX(const T& x) {
            this->components[0] = x;
        }

        /**
         * Set the y-component of the vector.
         *
         * @param y The new y-component.
         */
        inline void SetY(const T& y) {
            this->components[1] = y;
        }

        /**
         * Set the z-component of the vector.
         *
         * @param z The new z-component.
         */
        inline void SetZ(const T& z) {
            this->components[2] = z;
        }

        /**
         * Answer the x-component of the vector.
         *
         * @return The x-component of the vector.
         */
        inline const T& X(void) const {
            return this->components[0];
        }

        /**
         * Answer the y-component of the vector.
         *
         * @return The y-component of the vector.
         */
        inline const T& Y(void) const {
            return this->components[1];
        }

        /**
         * Answer the z-component of the vector.
         *
         * @return The z-component of the vector.
         */
        inline const T& Z(void) const {
            return this->components[2];
        }

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline Vector3D& operator =(const Vector3D& rhs) {
            Vector<T, 3, E, S>::operator =(rhs);
            return *this;
        }

        template<class Tp, unsigned int Dp, class Ep, 
            template<class, unsigned int> class Sp>
        inline Vector3D& operator =(const Vector<Tp, Dp, Ep, Sp>& rhs) {
            Vector<T, 3, E, S>::operator =(rhs);
            return *this;
        }
    };


    /*
     * vislib::math::Vector3D<T, E, S>::~Vector3D
     */
    template<class T, class E, template<class, unsigned int> class S>
    Vector3D<T, E, S>::~Vector3D(void) {
    }


    /*
     * vislib::math::Vector3D<T, E, S>::Cross
     */
    template<class T, class E, template<class, unsigned int> class S>
    Vector3D<T, E, S> Vector3D<T, E, S>::Cross(const Vector3D& rhs) const {
        Vector3D retval(this->components[1] * rhs.components[2] 
            - this->components[2] * rhs.components[1],
            this->components[2] * rhs.components[0] 
            - this->components[0] * rhs.components[2],
            this->components[0] * rhs.components[1] 
            - this->components[1] * rhs.components[0]);
        return retval;

    }
} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_VECTOR_H_INCLUDED */
