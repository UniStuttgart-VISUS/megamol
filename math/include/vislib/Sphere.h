/*
 * Sphere.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SPHERE_H_INCLUDED
#define VISLIB_SPHERE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractSphere.h"


namespace vislib {
namespace math {


    /**
     * This class represents a sphere, which is represented by its center point
     * and radius.
     */
    template<class T> class Sphere : public AbstractSphere<T, T[4]> {

    public:

        /**
         * Ctor. 
         *
         * @param x      The x-coordinate of the sphere's center point.
         * @param y      The y-coordinate of the sphere's center point.
         * @param z      The z-coordinate of the sphere's center point.
         * @param radius The radius of the sphere.
         */
        inline Sphere(const T x = static_cast<T>(0), 
                const T y = static_cast<T>(0), 
                const T z = static_cast<T>(0), 
                const T radius = static_cast<T>(0)) 
                : Super() {
            this->xyzr[Super::IDX_X] = x;
            this->xyzr[Super::IDX_Y] = y;
            this->xyzr[Super::IDX_Z] = z;
            this->xyzr[Super::IDX_RADIUS] = radius;
        }

        /**
         * Ctor.
         *
         * @param center The center point of the sphere.
         * @param radius The radius of the sphere.
         */
        template<class Sp> 
        Sphere(const AbstractPoint<T, 3, Sp>& center, const T radius);

        /**
         * Clone rhs.
         *
         * @param rhs The object to clone.
         */
        inline Sphere(const Sphere& rhs) {
            ::memcpy(this->xyzr, rhs.xyzr, 4 * sizeof(T));
        }

        /**
         * Allow arbitrary sphere to sphere conversions.
         *
         * @param rhs The object to clone.
         */
        template<class Tp, class Sp>
        Sphere(const AbstractSphere<Tp, Sp>& rhs);

        /** Dtor. */
        ~Sphere(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline Sphere& operator =(const Sphere& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Assignment. This operator allows arbitrary sphere to sphere 
         * conversions.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        inline Sphere& operator =(const AbstractSphere<Tp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:
    
        /** The superclass type. */
        typedef AbstractSphere<T, T[4]> Super;

    };


    /*
     * vislib::math::Sphere<T>::Sphere
     */
    template<class T> 
    template<class Sp> 
    Sphere<T>::Sphere(const AbstractPoint<T, 3, Sp>& center, const T radius) {
        this->xyzr[Super::IDX_X] = center.X();
        this->xyzr[Super::IDX_Y] = center.Y();
        this->xyzr[Super::IDX_Z] = center.Z();
        this->xyzr[Super::IDX_RADIUS] = radius;
    }


    /*
     * vislib::math::Sphere<T>::Sphere
     */
    template<class T> 
    template<class Tp, class Sp>
    Sphere<T>::Sphere(const AbstractSphere<Tp, Sp>& rhs) {
        ShallowPoint<T, 3> rhsCenter = rhs.GetCenter();
        this->xyzr[Super::IDX_X] = static_cast<T>(rhsCenter.X());
        this->xyzr[Super::IDX_Y] = static_cast<T>(rhsCenter.Y());
        this->xyzr[Super::IDX_Z] = static_cast<T>(rhsCenter.Z());
        this->xyzr[Super::IDX_RADIUS] = static_cast<T>(rhs.GetRadius());
    }


    /*
     * vislib::math::Sphere<T>::~Sphere
     */
    template<class T> Sphere<T>::~Sphere(void) {
    }
    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SPHERE_H_INCLUDED */

