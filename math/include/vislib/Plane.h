/*
 * Plane.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_PLANE_H_INCLUDED
#define VISLIB_PLANE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/AbstractPlane.h"


namespace vislib {
namespace math {

    /**
     * Objects of this class represent a plane with cartesian coordinates.
     *
     * @author Christoph Mueller
     */
    template<class T> class Plane : public AbstractPlane<T, T[4]> {

    public:

        /** 
         * Ctor. 
         */
        inline Plane(void) {
            this->parameters[IDX_A] = static_cast<T>(0);
            this->parameters[IDX_B] = static_cast<T>(0);
            this->parameters[IDX_C] = static_cast<T>(0);
            this->parameters[IDX_D] = static_cast<T>(0);
        }

        /** 
         * Create a plane.
         *
         * @param a The parameter a in the equation ax + by + cz + d = 0.
         * @param b The parameter b in the equation ax + by + cz + d = 0.
         * @param c The parameter c in the equation ax + by + cz + d = 0.
         * @param d The parameter d in the equation ax + by + cz + d = 0.
         */
        inline Plane(const T& a, const T& b, const T& c, const T& d) {
            this->Set(a, b, c, d); 
        }

        /**
         * Create a plane through 'point' with the normal vector 'normal'.
         *
         * @param point  An arbitrary point in the plane.
         * @param normal The normal vector of the plane.
         */
        template<class Tp1, class Sp1, class Tp2, class Sp2>
        inline Plane(const AbstractPoint3D<Tp1, Sp2>& point, 
                const AbstractVector3D<Tp2, Sp2>& normal) {
            this->Set(point, normal);
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Plane(const Plane& rhs) {
            ::memcpy(this->parameters, rhs.parameters, 4 * sizeof(T));
        }

        /**
         * Create a copy of 'rhs'. This ctor allows for arbitrary plane to
         * plane conversions. See documentation of Point for additional
         * remarks.
         *
         * @param rhs The object to be cloned.
         */        
        template<class Tp, class Sp>
        Plane(const AbstractPlane<Tp, Sp>& rhs);

        /** Dtor. */
        ~Plane(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        Plane& operator =(const AbstractPlane& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Assignment. This operator allows arbitrary plane to plane 
         * conversions.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        Plane& operator =(const AbstractPlane<Tp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** The super class of this one. */
        typedef AbstractPlane<T, T[4]> Super;

    };


    /*
     * Plane<T>::Plane
     */
    template <class T> 
    template<class Tp, class Sp>
    Plane<T>::Plane(const AbstractPlane<Tp, Sp>& rhs) {
        this->parameters[IDX_A] = static_cast<T>(rhs.A());
        this->parameters[IDX_B] = static_cast<T>(rhs.B());
        this->parameters[IDX_C] = static_cast<T>(rhs.C());
        this->parameters[IDX_D] = static_cast<T>(rhs.D());
    }


    /*
     * Plane<T>::~Plane
     */
    template<class T> Plane<T>::~Plane(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_PLANE_H_INCLUDED */
