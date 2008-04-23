/*
 * ShallowQuaternion.h  28.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHALLOWQUATERNION_H_INCLUDED
#define VISLIB_SHALLOWQUATERNION_H_INCLUDED
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
    template<class T> class ShallowQuaternion : public AbstractQuaternion<T, T*> {

    public:


        /**
         * Create a new quaternion using the memory designated by 'components'
         * as data. 'components' must hold the x, y, z and w component of the
         * quaternion in consecutive order. The caller remains owner of the 
         * memory, but must ensure that it lives at least as long as the
         * object.
         *
         * @param components The components of the quaternion.
         */
        inline explicit ShallowQuaternion(T *components) {
            this->components = components;
        }


        /**
         * Clone 'rhs'. This operation creates an alias of 'rhs'
         *
         * @param rhs The object to be copied.
         */
        inline ShallowQuaternion(const ShallowQuaternion& rhs) {
            this->components = rhs.components;
        }

        /** Dtor. */
        ~ShallowQuaternion(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline ShallowQuaternion& operator =(const ShallowQuaternion& rhs) {
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
        inline ShallowQuaternion& operator =(
                const AbstractQuaternion<Tp, Sp>& rhs) {
            AbstractQuaternion<T, T[4]>::operator =(rhs);
            return *this;
        }
    };


    /*
     * vislib::math::ShallowQuaternion<T>::~ShallowQuaternion
     */
    template<class T>
    ShallowQuaternion<T>::~ShallowQuaternion(void) {
    }


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHALLOWQUATERNION_H_INCLUDED */
