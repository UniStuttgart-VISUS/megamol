/*
 * AbstractSphere.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTSPHERE_H_INCLUDED
#define VISLIB_ABSTRACTSPHERE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#define _USE_MATH_DEFINES
#include <cmath>

#include "vislib/assert.h"
#include "vislib/mathfunctions.h"
#include "vislib/Point.h"
#include "vislib/ShallowPoint.h"
#include "vislib/types.h"


namespace vislib {
namespace math {


    /**
     * This class represents a sphere, which is represented by its center point
     * and radius.
     */
    template<class T, class S> class AbstractSphere {

    public:

        /** Possible relative locations of a point with regard to a sphere. */
        enum RelativeLocation {
            OUTSIDE_SPHERE = -1,
            ON_SPHERE = 0,
            INSIDE_SPHERE = 1
        };

        /** Dtor. */
        ~AbstractSphere(void);

        /**
         * Answer the relative location of 'point' with regard to the sphere.
         *
         * @return The point to be checked.
         *
         * @return ON_SPHERE if the point lies on the sphere (within the epsilon
         *         range of T),
         *         OUTSIDE_SPHERE if the point lies outside the sphere, 
         *         INSIDE_SPHERE if the point lies inside the sphere.
         */
        template<class Sp>
        RelativeLocation CalcRelativeLocation(
            const AbstractPoint<T, 3, Sp>& point);

        /**
         * Answer the diameter of the sphere.
         *
         * @return The diameter of the sphere.
         */
        inline T Diameter(void) const {
            return static_cast<T>(2) * this->xyzr[IDX_RADIUS];
        }

        /**
         * Answer the center point of the sphere.
         *
         * @return The center point of the sphere.
         */
        const ShallowPoint<T, 3> GetCenter(void) const;

        /**
         * Answer the x-coordinate of the center point of the sphere.
         *
         * @return The x-coordinate of the center point of the sphere.
         */
        inline T GetCenterX(void) const {
            return this->xyzr[IDX_X];
        }

        /**
         * Answer the y-coordinate of the center point of the sphere.
         *
         * @return The y-coordinate of the center point of the sphere.
         */
        inline T GetCenterY(void) const {
            return this->xyzr[IDX_Y];
        }

        /**
         * Answer the z-coordinate of the center point of the sphere.
         *
         * @return The z-coordinate of the center point of the sphere.
         */
        inline T GetCenterZ(void) const {
            return this->xyzr[IDX_Z];
        }

        /**
         * Answer the radius of the sphere.
         *
         * @return The radius of the sphere.
         */
        inline T GetRadius(void) const {
            return this->xyzr[IDX_RADIUS];
        }

        /**
         * Set the center point of the sphere.
         *
         * @param center The center point of the sphere.
         */
        template<class Sp> 
        void SetCenter(const AbstractPoint<T, 3, Sp>& center);

        /**
         * Set the center point of the sphere.
         *
         * @param x The x-coordinate of the center point of the sphere.
         * @param y The y-coordinate of the center point of the sphere.
         * @param z The z-coordinate of the center point of the sphere.
         */
        inline void SetCenter(const T x, const T y, const T z) {
            this->xyzr[IDX_X] = x;
            this->xyzr[IDX_Y] = y;
            this->xyzr[IDX_Z] = z;
        }

        /**
         * Set the x-coordinate of the center point of the sphere.
         *
         * @param centerX The x-coordinate of the center point of the sphere.
         */
        inline void SetCenterX(const T centerX) {
            this->xyzr[IDX_X] = centerX;
        }

        /**
         * Set the y-coordinate of the center point of the sphere.
         *
         * @param centerY The y-coordinate of the center point of the sphere.
         */
        inline void SetCenterY(const T centerY) {
            this->xyzr[IDX_Y] = centerY;
        }

        /**
         * Set the z-coordinate of the center point of the sphere.
         *
         * @param centerZ The z-coordinate of the center point of the sphere.
         */
        inline void SetCenterZ(const T centerZ) {
            this->xyrz[IDX_Z] = centerZ;
        }

        /**
         * Set the radius of the sphere.
         *
         * @param radius The radius of the sphere.
         */
        inline void SetRadius(const T radius) {
            this->xyzr[IDX_RADIUS] = radius;
        }

        /**
         * Answer the surface area of the sphere.
         *
         * @return The surface area of the sphere.
         */
        inline T SurfaceArea(void) const {
            return (static_cast<T>(4.0 * M_PI) * Sqr(this->xyzr[IDX_RADIUS]));
        }

        /**
         * Answer the volume of the sphere.
         *
         * @return The volume of the sphere.
         */
        inline T Volume(void) const {
            return (static_cast<T>(4.0 / 3.0 * M_PI) * this->xyzr[IDX_RADIUS]
                * this->xyzr[IDX_RADIUS] * this->xyzr[IDX_RADIUS]);
        }

        /**
         * Assigment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractSphere& operator =(const AbstractSphere& rhs);

        /**
         * Assigment operator. This operator never creates an alias, even for
         * shallow spheres!
         *
         * This assignment allows for arbitrary sphere to sphere conversions.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        AbstractSphere& operator =(const AbstractSphere<Tp, Sp>& rhs);

        /**
         * Test for equality. The operator uses the IsEqual method for T for 
         * each member.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this sphere are equal, false otherwise.
         */
        bool operator ==(const AbstractSphere& rhs) const;

        /**
         * Test for inequality. The operator uses the IsEqual method for T for 
         * each member.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this sphere are not equal, false otherwise.
         */
        inline bool operator !=(const AbstractSphere& rhs) const {
            return !(*this == rhs);
        }

    protected:

        /** Index of the center point x-coordinate in 'xyzr'. */
        static const UINT_PTR IDX_X;

        /** Index of the center point y-coordinate in 'xyzr'. */
        static const UINT_PTR IDX_Y;

        /** Index of the center point z-coordinate in 'xyzr'. */
        static const UINT_PTR IDX_Z;

        /** Index of the radius in 'xyzr'. */
        static const UINT_PTR IDX_RADIUS;

        /**
         * Forbidden default ctor. This does nothing.
         */
        inline AbstractSphere(void) {}

        /** The center point and radius of the sphere. */
        S xyzr;

    };


    /*
     * vislib::math::AbstractSphere<T, S>::~AbstractSphere
     */
    template<class T, class S> AbstractSphere<T, S>::~AbstractSphere(void) {
    }

    
    /*
     * vislib::math::AbstractSphere<T, S>::CalcRelativeLocation
     */
    template<class T, class S> 
    template<class Sp>
    typename AbstractSphere<T, S>::RelativeLocation 
    AbstractSphere<T, S>::CalcRelativeLocation(
            const AbstractPoint<T, 3, Sp>& point) {
        T d = point.Distance(this->GetCenter());
        if (IsEqual<T>(d, 0)) {
            return ON_SPHERE;
        } else if (d < 0) {
            return INSIDE_SPHERE;
        } else {
            return OUTSIDE_SPHERE;
        }
    }


    /*
     * vislib::math::AbstractSphere<T, S>::GetCenter
     */
    template<class T, class S> 
    const ShallowPoint<T, 3> AbstractSphere<T, S>::GetCenter(void) const {
        ShallowPoint<T, 3> retval(const_cast<T *>(this->xyzr));
        return retval;
    }


    /*
     * vislib::math::AbstractSphere<T, S>::SetCenter
     */
    template<class T, class S> 
    template<class Sp> 
    void AbstractSphere<T, S>::SetCenter(
            const AbstractPoint<T, 3, Sp>& center) {
        this->xyzr[IDX_X] = center.X();
        this->xyzr[IDX_Y] = center.Y();
        this->xyzr[IDX_Z] = center.Z();
    }


    /*
     * vislib::math::AbstractSphere<T, S>::operator =
     */
    template<class T, class S> 
    AbstractSphere<T, S>& AbstractSphere<T, S>::operator =(
            const AbstractSphere& rhs) {
        if (this != &rhs) {
            ::memcpy(this->xyzr, rhs.xyzr, 4 * sizeof(T));
        }

        return *this;
    }


    /*
     * vislib::math::AbstractSphere<T, S>::operator =
     */
    template<class T, class S> 
    template<class Tp, class Sp>
    AbstractSphere<T, S>& AbstractSphere<T, S>::operator =(
            const AbstractSphere<Tp, Sp>& rhs) {
        if (static_cast<void *>(this) != static_cast<const void *>(&rhs)) {
            ShallowPoint<T, 3> rhsCenter = rhs.GetCenter();
            this->xyzr[IDX_X] = static_cast<T>(rhsCenter.X());
            this->xyzr[IDX_Y] = static_cast<T>(rhsCenter.Y());
            this->xyzr[IDX_Z] = static_cast<T>(rhsCenter.Z());
            this->xyzr[IDX_RADIUS] = static_cast<T>(rhs.GetRadius());
        }

        return *this;
    }


    /*
     * vislib::math::AbstractSphere<T, S>::operator ==
     */
    template<class T, class S> 
    bool AbstractSphere<T, S>::operator ==(const AbstractSphere& rhs) const {
        return (IsEqual<T>(this->xyzr[IDX_X], rhs.xyzr[IDX_X])
            && IsEqual<T>(this->xyzr[IDX_Y], rhs.xyzr[IDX_Y])
            && IsEqual<T>(this->xyzr[IDX_Z], rhs.xyzr[IDX_Z])
            && IsEqual<T>(this->xyzr[IDX_RADIUS], rhs.xyzr[IDX_RADIUS]));
    }


    /*
     * vislib::math::AbstractSphere<T, S>::IDX_X
     */
    template<class T, class S> const UINT_PTR AbstractSphere<T, S>::IDX_X = 0;


    /*
     * vislib::math::AbstractSphere<T, S>::IDX_Y
     */
    template<class T, class S> const UINT_PTR AbstractSphere<T, S>::IDX_Y = 1;


    /*
     * vislib::math::AbstractSphere<T, S>::IDX_Z
     */
    template<class T, class S> const UINT_PTR AbstractSphere<T, S>::IDX_Z = 2;


    /*
     * vislib::math::AbstractSphere<T, S>::IDX_RADIUS
     */
    template<class T, class S> 
    const UINT_PTR AbstractSphere<T, S>::IDX_RADIUS = 3;

    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTSPHERE_H_INCLUDED */
