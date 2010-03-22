/*
 * Quaternion.h  28.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#ifndef VISLIB_QUATERNION_H_INCLUDED
#define VISLIB_QUATERNION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractQuaternion.h"


namespace vislib {
namespace math {

    /**
     * A quaternion.
     */
    template<class T> class Quaternion : public AbstractQuaternion<T, T[4]> {

    public:

        /**
         * Create a new quaternion (<0, 0, 0>, 1).
         */
        inline Quaternion(void) {
            this->components[Super::IDX_X] = this->components[Super::IDX_Y] 
                = this->components[Super::IDX_Z] = static_cast<T>(0);
            this->components[Super::IDX_W] = static_cast<T>(1);
        }

        /**
         * Create a new quaternion.
         *
         * @param x The new x-component.
         * @param y The new y-component.
         * @param z The new z-component.
         * @param w The new w-component.
         */
        inline Quaternion(const T& x, const T& y, const T& z, const T& w) 
                : Super(x, y, z, w) {
        }

        /**
         * Create a new quaternion using the specified components. 'components'
         * must be an array of four consecutive elements, ordered x, y, z, and 
         * w. The caller remains owner of the memory, the object creates a
         * deep copy.
         *
         * @param components The components of the quaternion.
         */
        inline explicit Quaternion(const T *components) {
            ::memcpy(this->components, components, 4 * sizeof(T));
        }

        /**
         * Construct a quaternion from an angle and a rotation axis.
         *
         * @param angle The rotation angle in radians.
         * @param axis  The rotation axis.
         */
        template<class Tp, class Sp>
        inline Quaternion(const T& angle, const AbstractVector<Tp, 3, Sp>& axis) {
            this->Set(angle, axis);
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be copied.
         */
        inline Quaternion(const Quaternion& rhs) {
            ::memcpy(this->components, rhs.components, 4 * sizeof(T));
        }

        /**
         * Create a copy of 'rhs'. This ctor allows for arbitrary quaternion to
         * quaternion conversions.
         *
         * @param rhs The quaternion to be cloned.
         */
        template<class Tp, class Sp>
        explicit Quaternion(const AbstractQuaternion<Tp, Sp>& rhs);

        /** Dtor. */
        ~Quaternion(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline Quaternion& operator =(const Quaternion& rhs) {
            AbstractQuaternion<T, T[4]>::operator =(rhs);
            return *this;
        }

        /**
         * Assignment. This operator allows arbitrary quaternion to
         * quaternion conversions.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        inline Quaternion& operator =(const AbstractQuaternion<Tp, Sp>& rhs) {
            AbstractQuaternion<T, T[4]>::operator =(rhs);
            return *this;
        }

    private:

        /** Super class typedef. */
        typedef AbstractQuaternion<T, T[4]> Super;
    };


    /*
     * vislib::math::Quaternion<T>::Quaternion
     */
    template<class T>
    template<class Tp, class Sp>
    Quaternion<T>::Quaternion(const AbstractQuaternion<Tp, Sp>& rhs) {
        this->components[Super::IDX_X] = static_cast<T>(rhs.X());
        this->components[Super::IDX_Y] = static_cast<T>(rhs.Y());
        this->components[Super::IDX_Z] = static_cast<T>(rhs.Z());
        this->components[Super::IDX_W] = static_cast<T>(rhs.W());
    }


    /*
     * vislib::math::Quaternion<T>::~Quaternion
     */
    template<class T>
    Quaternion<T>::~Quaternion(void) {
    }


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_QUATERNION_H_INCLUDED */
