/*
 * Frustum.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FRUSTUM_H_INCLUDED
#define VISLIB_FRUSTUM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractFrustum.h"


namespace vislib {
namespace math {


    /**
     * Objects of this class represent a rectangular view frustum.
     */
    template<class T> class Frustum : public AbstractFrustum<T, T[6]> {

    public:

        /** 
         * Create a degenerate frustum that has collapsed to a point.
         */
        inline Frustum(void) {
            this->bounds[Super::IDX_BOTTOM] = static_cast<T>(0);
            this->bounds[Super::IDX_TOP] = static_cast<T>(0);
            this->bounds[Super::IDX_LEFT] = static_cast<T>(0);
            this->bounds[Super::IDX_RIGHT] = static_cast<T>(0);
            this->bounds[Super::IDX_NEAR] = static_cast<T>(0);
            this->bounds[Super::IDX_FAR] = static_cast<T>(0);
        }

        /** 
         * Create a view frustum.
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
        inline Frustum(const T left, const T right, const T bottom, 
                const T top, const T zNear, const T zFar) {
            this->Set(left, right, bottom, top, zNear, zFar); 
        }

        /**
         * Create a view frustum to represent the view frustum of the given 
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
        inline Frustum(const T fovy, const double aspectRatio, const T zNear, 
                const T zFar) {
            this->Set(fovy, aspectRatio, zNear, zFar);
        }


        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Frustum(const Frustum& rhs) {
            ::memcpy(this->bounds, rhs.bounds, 6 * sizeof(T));
        }

        /**
         * Create a copy of 'rhs'. This ctor allows for arbitrary frustum to
         * frustum conversions.
         *
         * @param rhs The object to be cloned.
         */        
        template<class Tp, class Sp>
        Frustum(const AbstractFrustum<Tp, Sp>& rhs);

        /** Dtor. */
        ~Frustum(void);


        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline Frustum& operator =(const Frustum& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Assignment. This operator allows arbitrary frustum to frustum
         * conversions.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        inline Frustum& operator =(const AbstractFrustum<Tp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** The super class of this one. */
        typedef AbstractFrustum<T, T[6]> Super;

    };


    /*
     * vislib::math::Frustum<T>::Frustum
     */
    template <class T> 
    template<class Tp, class Sp>
    Frustum<T>::Frustum(const AbstractFrustum<Tp, Sp>& rhs) {
        this->bounds[Super::IDX_BOTTOM] = static_cast<T>(rhs.Bottom());
        this->bounds[Super::IDX_TOP] = static_cast<T>(rhs.Top());
        this->bounds[Super::IDX_LEFT] = static_cast<T>(rhs.Left());
        this->bounds[Super::IDX_RIGHT] = static_cast<T>(rhs.Right());
        this->bounds[Super::IDX_NEAR] = static_cast<T>(rhs.Near());
        this->bounds[Super::IDX_FAR] = static_cast<T>(rhs.Far());
    }


    /*
     * Frustum<T>::~Frustum
     */
    template<class T> Frustum<T>::~Frustum(void) {
    }
    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_FRUSTUM_H_INCLUDED */

