/*
 * AbstractFrustum.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTFRUSTUM_H_INCLUDED
#define VISLIB_ABSTRACTFRUSTUM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/assert.h"
#include "vislib/mathfunctions.h"
#include "vislib/types.h"


namespace vislib {
namespace math {


    /**
     * This is the abstract superclass for shallow storage and deep storage
     * rectangular frustums. This frustum class is intended to represent a view
     * frusum and uses according naming conventions.
     */
    template<class T, class S> class AbstractFrustum {

    public:

        /** Dtor. */
        ~AbstractFrustum(void);

        /**
         * Answer the distance of the bottom plane/near plane intersection
         * from the origin.
         *
         * @return The bottom plane offset.
         */
        inline T Bottom(void) const {
            return this->bounds[IDX_BOTTOM];
        }

        /**
         * Answer the distance of the far plane from the origin.
         *
         * @return The far plane offset.
         */
        inline T Far(void) const {
            return this->bounds[IDX_FAR];
        }

        /**
         * Answer the distance of the left plane/near plane intersection
         * from the origin.
         *
         * @return The left plane offset.
         */
        inline T Left(void) const {
            return this->bounds[IDX_LEFT];
        }

        /**
         * Answer the distance of the near plane from the origin.
         *
         * @return The near plane offset.
         */
        inline T Near(void) const {
            return this->bounds[IDX_NEAR];
        }

        /**
         * Answer the distance of the right plane/near plane intersection
         * from the origin.
         *
         * @return The right plane offset.
         */
        inline T Right(void) const {
            return this->bounds[IDX_RIGHT];
        }

        /**
         * Answer the distance of the top plane/near plane intersection
         * from the origin.
         *
         * @return The top plane offset.
         */
        inline T Top(void) const {
            return this->bounds[IDX_TOP];
        }

        /**
         * Answer the distance of the bottom plane/near plane intersection
         * from the origin.
         *
         * @return The bottom plane offset.
         */
        inline T GetBottom(void) const {
            return this->bounds[IDX_BOTTOM];
        }

        /**
         * Answer the distance of the far plane from the origin.
         *
         * @return The far plane offset.
         */
        inline T GetFar(void) const {
            return this->bounds[IDX_FAR];
        }

        /**
         * Answer the distance of the left plane/near plane intersection
         * from the origin.
         *
         * @return The left plane offset.
         */
        inline T GetLeft(void) const {
            return this->bounds[IDX_LEFT];
        }

        /**
         * Answer the distance of the near plane from the origin.
         *
         * @return The near plane offset.
         */
        inline T GetNear(void) const {
            return this->bounds[IDX_NEAR];
        }

        /**
         * Answer the distance of the right plane/near plane intersection
         * from the origin.
         *
         * @return The right plane offset.
         */
        inline T GetRight(void) const {
            return this->bounds[IDX_RIGHT];
        }

        /**
         * Answer the distance of the top plane/near plane intersection
         * from the origin.
         *
         * @return The top plane offset.
         */
        inline T GetTop(void) const {
            return this->bounds[IDX_TOP];
        }

        /** 
         * Change the frustum bounds.
         *
         * @param left    The offset of the left/near plane intersection from
         *                the origin.
         * @param right   The offset of the right/near plane intersection from
         *                the origin.
         * @param bottom  The offset of the bottom/near plane intersection from
         *                the origin.
         * @param top     The offset of the top/near plane intersection from
         *                the origin.
         * @param zNear   The offset of the near plane from the origin.
         * @param zFar    The offset of the far plane from the origin.
         */
        inline void Set(const T left, const T right, const T bottom, 
                const T top, const T zNear, const T zFar) {
            this->bounds[IDX_BOTTOM] = bottom;
            this->bounds[IDX_TOP] = top;
            this->bounds[IDX_LEFT] = left;
            this->bounds[IDX_RIGHT] = right;
            this->bounds[IDX_NEAR] = zNear;
            this->bounds[IDX_FAR] = zFar;
        }

        /**
         * Change the frustum bounds to represent the view frustum of the given 
         * perspective projection.
         *
         * @param fovy        The field of view angle, in degrees, in the 
         *                    y-direction.
         * @param aspectRatio The aspect ratio that determines the field of 
         *                    view in the x-direction. The aspect ratio is the
         *                    ratio of x (width) to y (height). 
         * @param zNear       The distance from the viewer to the near clipping 
         *                    plane. 
         * @param zFar        The distance from the viewer to the far clipping
         *                    plane. 
         */
        void Set(const T fovy, const double aspectRatio, const T zNear, 
            const T zFar);

        /**
         * Change the distance of the bottom plane/near plane intersection from
         * the origin.
         *
         * @param bottom The new plane offset.
         */
        inline void SetBottom(const T bottom) const {
            this->bounds[IDX_BOTTOM] = bottom;
        }

        /**
         * Change the distance of the far plane  from the origin.
         *
         * @param zFar The new plane offset.
         */
        inline void SetFar(const T zFar) const {
            this->bounds[IDX_FAR] = zFar;
        }

        /**
         * Change the distance of the left plane/near plane intersection from
         * the origin.
         *
         * @param left The new plane offset.
         */
        inline void SetLeft(const T left) const {
            this->bounds[IDX_LEFT] = left;
        }

        /**
         * Change the distance of the near plane from the origin.
         *
         * @param zNear The new plane offset.
         */
        inline void SetNear(const T zNear) const {
            this->bounds[IDX_NEAR] = zNear;
        }

        /**
         * Change the distance of the right plane/near plane intersection from
         * the origin.
         *
         * @param right The new plane offset.
         */
        inline void SetRight(const T right) const {
            this->bounds[IDX_RIGHT] = right;
        }

        /**
         * Change the distance of the top plane/near plane intersection from
         * the origin.
         *
         * @param top The new plane offset.
         */
        inline void SetTop(const T top) const {
            this->bounds[IDX_TOP] = top;
        }


        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractFrustum& operator =(const AbstractFrustum& rhs);

        /**
         * Assignment. This operator allows arbitrary frustum to frustum 
         * conversions.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        AbstractFrustum& operator =(const AbstractFrustum<Tp, Sp>& rhs);

        /**
         * Test for equality. The IsEqual function is used for this.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        bool operator ==(const AbstractFrustum& rhs) const;

        /**
         * Test for equality. This operator allows comparing frustums that
         * have been instantiated for different scalar types. The IsEqual<T>
         * function for the scalar type of the left hand side operand is used
         * as comparison operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        template<class Tp, class Sp>
        bool operator ==(const AbstractFrustum<Tp, Sp>& rhs) const;

        /**
         * Test for inequality. The IsEqual function is used for this.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const AbstractFrustum& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Test for inequality. This operator allows comparing frustums that
         * have been instantiated for different scalar types. The IsEqual<T>
         * function for the scalar type of the left hand side operand is used
         * as comparison operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal, false otherwise.
         */
        template<class Tp, class Sp>
        inline bool operator !=(const AbstractFrustum<Tp, Sp>& rhs) const {
            return !(*this == rhs);
        }


    protected:

        /** 
         * Disallow instances of this class.
         */
        inline AbstractFrustum(void) {}

        /** The index of the bottom plane offset. */
        static const UINT_PTR IDX_BOTTOM;

        /** The index of the far plane offset. */
        static const UINT_PTR IDX_FAR;

        /** The index of the left plane offset. */
        static const UINT_PTR IDX_LEFT;

        /** The index of the near plane offset. */
        static const UINT_PTR IDX_NEAR;

        /** The index of the right plane offset. */
        static const UINT_PTR IDX_RIGHT;

        /** The index of the top plane offset. */
        static const UINT_PTR IDX_TOP;

        /** 
         * The bounds defining the frustum. The indices in the array are 
         * given by the IDX_* constants.
         */
        S bounds;

    };


    /*
     * vislib::math::AbstractFrustum<T, S>::~AbstractFrustum
     */
    template<class T, class S> AbstractFrustum<T, S>::~AbstractFrustum(void) {
        // Must not do anything due to the Crowbar pattern(TM).
    }


    /*
     * vislib::math::AbstractFrustum<T, S>::Set
     */
    template<class T, class S> void AbstractFrustum<T, S>::Set(const T fovy, 
            const double aspectRatio, const T zNear, const T zFar) {
        T height = static_cast<T>(tan(static_cast<double>(fovy) * 0.5));
        T width = static_cast<T>(static_cast<double>(height) * aspectRatio);
        this->Set(-width, width, -height, height, zNear, zFar);
    }


    /*
     * vislib::math::AbstractFrustum<T, S>::operator =
     */
    template<class T, class S>
    AbstractFrustum<T, S>& AbstractFrustum<T, S>::operator =(
            const AbstractFrustum& rhs) {
        if (this != &rhs) {
            ::memcpy(this->bounds, rhs.bounds, 6 * sizeof(T));
        }

        return *this;
    }


    /*
     * vislib::math::AbstractFrustum<T, S>::operator =
     */
    template<class T, class S>
    template<class Tp, class Sp>
    AbstractFrustum<T, S>& AbstractFrustum<T, S>::operator =(
            const AbstractFrustum<Tp, Sp>& rhs) {
        if (static_cast<void *>(this) != static_cast<const void *>(&rhs)) {
            this->bounds[IDX_BOTTOM] = static_cast<T>(rhs.Bottom());
            this->bounds[IDX_TOP] = static_cast<T>(rhs.Top());
            this->bounds[IDX_LEFT] = static_cast<T>(rhs.Left());
            this->bounds[IDX_RIGHT] = static_cast<T>(rhs.Right());
            this->bounds[IDX_NEAR] = static_cast<T>(rhs.Near());
            this->bounds[IDX_FAR] = static_cast<T>(rhs.Far());
        }

        return *this;
    }


    /*
     * vislib::math::AbstractFrustum<T, S>::operator ==
     */
    template<class T, class S>
    bool AbstractFrustum<T, S>::operator ==(const AbstractFrustum& rhs) const {
        return (IsEqual(this->bounds[IDX_BOTTOM], rhs.bounds[IDX_BOTTOM])
            && IsEqual(this->bounds[IDX_TOP], rhs.bounds[IDX_TOP])
            && IsEqual(this->bounds[IDX_LEFT], rhs.bounds[IDX_LEFT])
            && IsEqual(this->bounds[IDX_RIGHT], rhs.bounds[IDX_RIGHT])
            && IsEqual(this->bounds[IDX_NEAR], rhs.bounds[IDX_NEAR])
            && IsEqual(this->bounds[IDX_FAR], rhs.bounds[IDX_FAR]));
    }


    /*
     * vislib::math::AbstractFrustum<T, S>::operator ==
     */
    template<class T, class S>
    template<class Tp, class Sp> bool AbstractFrustum<T, S>::operator ==(
            const AbstractFrustum<Tp, Sp>& rhs) const {
        return (IsEqual<T>(this->bounds[IDX_BOTTOM], rhs.bounds[IDX_BOTTOM])
            && IsEqual<T>(this->bounds[IDX_TOP], rhs.bounds[IDX_TOP])
            && IsEqual<T>(this->bounds[IDX_LEFT], rhs.bounds[IDX_LEFT])
            && IsEqual<T>(this->bounds[IDX_RIGHT], rhs.bounds[IDX_RIGHT])
            && IsEqual<T>(this->bounds[IDX_NEAR], rhs.bounds[IDX_NEAR])
            && IsEqual<T>(this->bounds[IDX_FAR], rhs.bounds[IDX_FAR]));
    }


    /*
     * vislib::math::AbstractFrustum<T, S>::IDX_BOTTOM
     */
    template<class T, class S>
    const UINT_PTR AbstractFrustum<T, S>::IDX_BOTTOM = 0;


    /*
     * vislib::math::AbstractFrustum<T, S>::IDX_FAR
     */
    template<class T, class S>
    const UINT_PTR AbstractFrustum<T, S>::IDX_FAR = 4;


    /*
     * vislib::math::AbstractFrustum<T, S>::IDX_LEFT
     */
    template<class T, class S>
    const UINT_PTR AbstractFrustum<T, S>::IDX_LEFT = 2;


    /*
     * vislib::math::AbstractFrustum<T, S>::IDX_NEAR
     */
    template<class T, class S>
    const UINT_PTR AbstractFrustum<T, S>::IDX_NEAR = 5;


    /*
     * vislib::math::AbstractFrustum<T, S>::IDX_RIGHT
     */    
    template<class T, class S>
    const UINT_PTR AbstractFrustum<T, S>::IDX_RIGHT = 3;


    /*
     * vislib::math::AbstractFrustum<T, S>::IDX_TOP
     */
    template<class T, class S>
    const UINT_PTR AbstractFrustum<T, S>::IDX_TOP = 1;

    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTFRUSTUM_H_INCLUDED */

