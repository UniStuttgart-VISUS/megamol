/*
 * ViewFrustum.h
 *
 * Copyright (C) 2009 by Universität Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_VIEWFRUSTUM_H_INCLUDED
#define VISLIB_VIEWFRUSTUM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractViewFrustum.h"


namespace vislib {
namespace math {


    /**
     * Objects of this class represent a rectangular view frustum.
     */
    template<class T> 
    class ViewFrustum : public AbstractViewFrustum<T, T[6]> {

    public:

        /** 
         * Create a degenerate frustum that has collapsed to a point.
         */
        inline ViewFrustum(void) {
            this->offsets[Super::IDX_BOTTOM] = static_cast<T>(0);
            this->offsets[Super::IDX_TOP] = static_cast<T>(0);
            this->offsets[Super::IDX_LEFT] = static_cast<T>(0);
            this->offsets[Super::IDX_RIGHT] = static_cast<T>(0);
            this->offsets[Super::IDX_NEAR] = static_cast<T>(0);
            this->offsets[Super::IDX_FAR] = static_cast<T>(0);
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
        inline ViewFrustum(const T left, const T right, const T bottom, 
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
        inline ViewFrustum(const T fovy, const double aspectRatio, 
                const T zNear, const T zFar) {
            this->Set(fovy, aspectRatio, zNear, zFar);
        }


        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline ViewFrustum(const ViewFrustum& rhs) {
            ::memcpy(this->offsets, rhs.offsets, 6 * sizeof(T));
        }

        /**
         * Create a copy of 'rhs'. This ctor allows for arbitrary frustum to
         * frustum conversions.
         *
         * @param rhs The object to be cloned.
         */        
        template<class Tp, class Sp>
        ViewFrustum(const AbstractViewFrustum<Tp, Sp>& rhs);

        /** Dtor. */
        virtual ~ViewFrustum(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline ViewFrustum& operator =(const ViewFrustum& rhs) {
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
        inline ViewFrustum& operator =(
                const AbstractViewFrustum<Tp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** The super class of this one. */
        typedef AbstractViewFrustum<T, T[6]> Super;

    };


    /*
     * vislib::math::ViewFrustum<T>::ViewFrustum
     */
    template<class T> 
    template<class Tp, class Sp>
    ViewFrustum<T>::ViewFrustum(const AbstractViewFrustum<Tp, Sp>& rhs) {
        this->offsets[Super::IDX_BOTTOM] 
            = static_cast<T>(rhs.GetBottomDistance());
        this->offsets[Super::IDX_TOP] = static_cast<T>(rhs.GetTopDistance());
        this->offsets[Super::IDX_LEFT] = static_cast<T>(rhs.GetLeftDistance());
        this->offsets[Super::IDX_RIGHT] 
            = static_cast<T>(rhs.GetRightDistance());
        this->offsets[Super::IDX_NEAR] = static_cast<T>(rhs.GetNearDistance());
        this->offsets[Super::IDX_FAR] = static_cast<T>(rhs.GetFarDistance());
    }


    /*
     * ViewFrustum<T>::~ViewFrustum
     */
    template<class T> ViewFrustum<T>::~ViewFrustum(void) {
    }
    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_VIEWFRUSTUM_H_INCLUDED */

