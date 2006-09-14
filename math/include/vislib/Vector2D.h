/*
 * Vector2D.h  14.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_VECTOR2D_H_INCLUDED
#define VISLIB_VECTOR2D_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/Vector.h"


namespace vislib {
namespace math {

    /**
     * Specialisation for a two-dimensional vector. See Vector for additional
     * remarks.
     */
    template<class T,class E = EqualFunc<T>, 
            template<class, unsigned int> class S = DeepStorageClass> 
    class Vector2D : public Vector<T, 2, E, S> {

    public:

        inline Vector2D(void) : Vector<T, 2, E, S>() {}

        explicit inline Vector2D(const T *components) 
            : Vector<T, 2, E, S>(components) {}

        inline Vector2D(const Vector2D& rhs) 
            : Vector<T, 2, E, S>(rhs) {}

        template<class Tp, unsigned int Dp, class Ep,
            template<class, unsigned int> class Sp>
        inline Vector2D(const Vector<Tp, Dp, Ep, Sp>& vector)
            : Vector<T, 2, E, S>(vector) {}

        inline Vector2D(const T& x, const T& y) {
            this->components[0] = x;
            this->components[1] = y;
        }

        /** Dtor. */
        virtual ~Vector2D(void);

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
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline Vector2D& operator =(const Vector2D& rhs) {
            Vector<T, 2, E, S>::operator =(rhs);
            return *this;
        }

        template<class Tp, unsigned int Dp, class Ep, 
            template<class, unsigned int> class Sp>
        inline Vector2D& operator =(const Vector<Tp, Dp, Ep, Sp>& rhs) {
            Vector<T, 2, E, S>::operator =(rhs);
            return *this;
        }
    };


    /*
     * vislib::math::Vector2D<T, E, S>::~Vector2D
     */
    template<class T, class E, template<class, unsigned int> class S>
    Vector2D<T, E, S>::~Vector2D(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_VECTOR2D_H_INCLUDED */
