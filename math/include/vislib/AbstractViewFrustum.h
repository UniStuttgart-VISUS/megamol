/*
 * AbstractViewFrustum.h
 *
 * Copyright (C) 2009 by Universität Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTVIEWFRUSTUM_H_INCLUDED
#define VISLIB_ABSTRACTVIEWFRUSTUM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <memory.h>

#include "vislib/AbstractPyramidalFrustum.h"


namespace vislib {
namespace math {


    /**
     * This is the abstract superclass for shallow storage and deep storage
     * rectangular eye-space view frustums. This frustum class is intended to 
     * represent a view frustum and has accessors using the respective naming 
     * conventions.
     */
    template<class T, class S> 
    class AbstractViewFrustum : public AbstractPyramidalFrustum<T> {

    public:

        /** Dtor. */
        virtual ~AbstractViewFrustum(void);

        /**
         * Answer the points that form the bottom base of the frustum.
         *
         * @param outPoints An array that will receive the points. All existing 
         *                  content will be replaced.
         */
        virtual void GetBottomBasePoints(
            vislib::Array<Point<T, 3> >& outPoints) const;

        /**
         * Answer the distance of the far bottom from the origin.
         *
         * @return The bottom plane distance.
         */
        inline T GetBottomDistance(void) const {
            return this->offsets[IDX_BOTTOM];
        }

        /**
         * Answer the distance of the far plane from the origin.
         *
         * @return The far clipping plane distance.
         */
        inline T GetFarDistance(void) const {
            return this->offsets[IDX_FAR];
        }

        /**
         * Answer the distance of the left plane/near plane intersection
         * from the origin.
         *
         * @return The left plane distance.
         */
        inline T GetLeftDistance(void) const {
            return this->offsets[IDX_LEFT];
        }

        /**
         * Answer the distance of the near plane from the origin.
         *
         * @return The near clipping plane distance.
         */
        inline T GetNearDistance(void) const {
            return this->offsets[IDX_NEAR];
        }

        /**
         * Answer the distance of the right plane/near plane intersection
         * from the origin.
         *
         * @return The right plane distance.
         */
        inline T GetRightDistance(void) const {
            return this->offsets[IDX_RIGHT];
        }

        /**
         * Answer the distance of the top plane/near plane intersection
         * from the origin.
         *
         * @return The top plane distance.
         */
        inline T GetTopDistance(void) const {
            return this->offsets[IDX_TOP];
        }

        /**
         * Answer the points that form the top base of the frustum.
         *
         * @param outPoints An array that will receive the points. All existing 
         *                  content will be replaced.
         */
        virtual void GetTopBasePoints(
            vislib::Array<Point<T, 3> >& outPoints) const;

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
        inline void Set(const T left, const T right, 
                const T bottom, const T top, 
                const T zNear, const T zFar) {
            this->offsets[IDX_BOTTOM] = bottom;
            this->offsets[IDX_TOP] = top;
            this->offsets[IDX_LEFT] = left;
            this->offsets[IDX_RIGHT] = right;
            this->offsets[IDX_NEAR] = zNear;
            this->offsets[IDX_FAR] = zFar;
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
         * @param bottom The new plane distance.
         */
        inline void SetBottomDistance(const T bottom) const {
            this->offsets[IDX_BOTTOM] = bottom;
        }

        /**
         * Change the distance of the far plane  from the origin.
         *
         * @param zFar The new plane distance.
         */
        inline void SetFarDistance(const T zFar) const {
            this->offsets[IDX_FAR] = zFar;
        }

        /**
         * Change the distance of the left plane/near plane intersection from
         * the origin.
         *
         * @param left The new plane distance.
         */
        inline void SetLeftDistance(const T left) const {
            this->offsets[IDX_LEFT] = left;
        }

        /**
         * Change the distance of the near plane from the origin.
         *
         * @param zNear The new plane distance.
         */
        inline void SetNearDistance(const T zNear) const {
            this->offsets[IDX_NEAR] = zNear;
        }

        /**
         * Change the distance of the right plane/near plane intersection from
         * the origin.
         *
         * @param right The new plane distance.
         */
        inline void SetRightDistance(const T right) const {
            this->offsets[IDX_RIGHT] = right;
        }

        /**
         * Change the distance of the top plane/near plane intersection from
         * the origin.
         *
         * @param top The new plane distance.
         */
        inline void SetTopDistance(const T top) const {
            this->offsets[IDX_TOP] = top;
        }

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractViewFrustum& operator =(const AbstractViewFrustum& rhs);

        /**
         * Assignment. This operator allows arbitrary frustum to frustum 
         * conversions.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        AbstractViewFrustum& operator =(
            const AbstractViewFrustum<Tp, Sp>& rhs);

        /**
         * Test for equality. The IsEqual function is used for this.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        bool operator ==(const AbstractViewFrustum& rhs) const;

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
        bool operator ==(const AbstractViewFrustum<Tp, Sp>& rhs) const;

        /**
         * Test for inequality. The IsEqual function is used for this.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const AbstractViewFrustum& rhs) const {
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
        inline bool operator !=(
                const AbstractViewFrustum<Tp, Sp>& rhs) const {
            return !(*this == rhs);
        }


    protected:

        /** Superclass typedef. */
        typedef AbstractPyramidalFrustum<T> Super;

        /** The number of elements in 'offset'. */
        static const UINT_PTR CNT_ELEMENTS;

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
         * Disallow instances of this class.
         */
        inline AbstractViewFrustum(void) {}

        /** 
         * The offsets defining the frustum. The semantics of the indices in 
         * the array are given by the IDX_* constants. The storage requirement
         * is at least CNT_ELEMENTS * sizeof(T).
         */
        S offsets;
    };


    /*
     * vislib::math::AbstractViewFrustum<T, S>::~AbstractViewFrustum
     */
    template<class T, class S> 
    AbstractViewFrustum<T, S>::~AbstractViewFrustum(void) {
        VLSTACKTRACE("AbstractViewFrustum::~AbstractViewFrustum", __FILE__, 
            __LINE__);
        // Must not do anything about 'offsets' due to the Crowbar pattern(TM).
    }


    /*
     * vislib::math::AbstractViewFrustum<T, S>::GetBottomBasePoints
     */
    template<class T, class S> 
    void AbstractViewFrustum<T, S>::GetBottomBasePoints(
            vislib::Array<Point<T, 3> >& outPoints) const {
        VLSTACKTRACE("AbstractViewFrustum::GetBottomBasePoints", __FILE__, 
            __LINE__);
        outPoints.Clear();
        outPoints.SetCount(4);
        outPoints[Super::IDX_LEFT_BOTTOM_POINT].Set(
            this->offsets[IDX_LEFT], 
            this->offsets[IDX_BOTTOM], 
            this->offsets[IDX_FAR]);
        outPoints[Super::IDX_RIGHT_BOTTOM_POINT].Set(
            this->offsets[IDX_RIGHT], 
            this->offsets[IDX_BOTTOM], 
            this->offsets[IDX_FAR]);
        outPoints[Super::IDX_RIGHT_TOP_POINT].Set(
            this->offsets[IDX_RIGHT],
            this->offsets[IDX_TOP], 
            this->offsets[IDX_FAR]);
        outPoints[Super::IDX_LEFT_TOP_POINT].Set(
            this->offsets[IDX_LEFT], 
            this->offsets[IDX_TOP], 
            this->offsets[IDX_FAR]);
    }


    /*
     * vislib::math::AbstractViewFrustum<T, S>::GetTopBasePoints
     */
    template<class T, class S> 
    void AbstractViewFrustum<T, S>::GetTopBasePoints(
            vislib::Array<Point<T, 3> >& outPoints) const {
        VLSTACKTRACE("AbstractViewFrustum::GetTopBasePoints", __FILE__, 
            __LINE__);
        outPoints.Clear();
        outPoints.SetCount(4);
        outPoints[Super::IDX_LEFT_BOTTOM_POINT].Set(
            this->offsets[IDX_LEFT], 
            this->offsets[IDX_BOTTOM], 
            this->offsets[IDX_NEAR]);
        outPoints[Super::IDX_RIGHT_BOTTOM_POINT].Set(
            this->offsets[IDX_RIGHT], 
            this->offsets[IDX_BOTTOM], 
            this->offsets[IDX_NEAR]);
        outPoints[Super::IDX_RIGHT_TOP_POINT].Set(
            this->offsets[IDX_RIGHT],
            this->offsets[IDX_TOP], 
            this->offsets[IDX_NEAR]);
        outPoints[Super::IDX_LEFT_TOP_POINT].Set(
            this->offsets[IDX_LEFT], 
            this->offsets[IDX_TOP], 
            this->offsets[IDX_NEAR]);
    }


    /*
     * vislib::math::AbstractViewFrustum<T, S>::Set
     */
    template<class T, class S> 
    void AbstractViewFrustum<T, S>::Set(const T fovy, 
            const double aspectRatio, const T zNear, const T zFar) {
        VLSTACKTRACE("AbstractViewFrustum::Set", __FILE__, __LINE__);
        T height = static_cast<T>(tan(static_cast<double>(fovy) * 0.5));
        T width = static_cast<T>(static_cast<double>(height) * aspectRatio);
        this->Set(-width, width, -height, height, zNear, zFar);
    }


    /*
     * vislib::math::AbstractViewFrustum<T, S>::operator =
     */
    template<class T, class S>
    AbstractViewFrustum<T, S>& AbstractViewFrustum<T, S>::operator =(
            const AbstractViewFrustum& rhs) {
        VLSTACKTRACE("AbstractViewFrustum::operator =", __FILE__, __LINE__);
        if (this != &rhs) {
            ::memcpy(this->offsets, rhs.offsets, CNT_ELEMENTS * sizeof(T));
        }

        return *this;
    }


    /*
     * vislib::math::AbstractViewFrustum<T, S>::operator =
     */
    template<class T, class S>
    template<class Tp, class Sp>
    AbstractViewFrustum<T, S>& AbstractViewFrustum<T, S>::operator =(
            const AbstractViewFrustum<Tp, Sp>& rhs) {
        VLSTACKTRACE("AbstractViewFrustum::operator =", __FILE__, __LINE__);
        if (static_cast<void *>(this) != static_cast<const void *>(&rhs)) {
            this->offsets[IDX_BOTTOM] 
                = static_cast<T>(rhs.GetBottomDistance());
            this->offsets[IDX_TOP] = static_cast<T>(rhs.GetTopDistance());
            this->offsets[IDX_LEFT] = static_cast<T>(rhs.GetLeftDistance());
            this->offsets[IDX_RIGHT] = static_cast<T>(rhs.GetRightDistance());
            this->offsets[IDX_NEAR] = static_cast<T>(rhs.GetNearDistance());
            this->offsets[IDX_FAR] = static_cast<T>(rhs.GetFarDistance());
        }

        return *this;
    }


    /*
     * vislib::math::AbstractViewFrustum<T, S>::operator ==
     */
    template<class T, class S>
    bool AbstractViewFrustum<T, S>::operator ==(
            const AbstractViewFrustum& rhs) const {
        VLSTACKTRACE("AbstractViewFrustum::operator ==", __FILE__, __LINE__);
        return (IsEqual(this->offsets[IDX_BOTTOM], rhs.offsets[IDX_BOTTOM])
            && IsEqual(this->offsets[IDX_TOP], rhs.offsets[IDX_TOP])
            && IsEqual(this->offsets[IDX_LEFT], rhs.offsets[IDX_LEFT])
            && IsEqual(this->offsets[IDX_RIGHT], rhs.offsets[IDX_RIGHT])
            && IsEqual(this->offsets[IDX_NEAR], rhs.offsets[IDX_NEAR])
            && IsEqual(this->offsets[IDX_FAR], rhs.offsets[IDX_FAR]));
    }


    /*
     * vislib::math::AbstractViewFrustum<T, S>::operator ==
     */
    template<class T, class S>
    template<class Tp, class Sp> 
    bool AbstractViewFrustum<T, S>::operator ==(
            const AbstractViewFrustum<Tp, Sp>& rhs) const {
        VLSTACKTRACE("AbstractViewFrustum::operator ==", __FILE__, __LINE__);
        return (IsEqual<T>(this->offsets[IDX_BOTTOM], rhs.offsets[IDX_BOTTOM])
            && IsEqual<T>(this->offsets[IDX_TOP], rhs.offsets[IDX_TOP])
            && IsEqual<T>(this->offsets[IDX_LEFT], rhs.offsets[IDX_LEFT])
            && IsEqual<T>(this->offsets[IDX_RIGHT], rhs.offsets[IDX_RIGHT])
            && IsEqual<T>(this->offsets[IDX_NEAR], rhs.offsets[IDX_NEAR])
            && IsEqual<T>(this->offsets[IDX_FAR], rhs.offsets[IDX_FAR]));
    }


    /*
     * vislib::math::AbstractViewFrustum<T, S>::CNT_ELEMENTS
     */
    template<class T, class S>
    const UINT_PTR AbstractViewFrustum<T, S>::CNT_ELEMENTS = 6;


    /*
     * vislib::math::AbstractViewFrustum<T, S>::IDX_BOTTOM
     */
    template<class T, class S>
    const UINT_PTR AbstractViewFrustum<T, S>::IDX_BOTTOM = 0;


    /*
     * vislib::math::AbstractViewFrustum<T, S>::IDX_FAR
     */
    template<class T, class S>
    const UINT_PTR AbstractViewFrustum<T, S>::IDX_FAR = 4;


    /*
     * vislib::math::AbstractViewFrustum<T, S>::IDX_LEFT
     */
    template<class T, class S>
    const UINT_PTR AbstractViewFrustum<T, S>::IDX_LEFT = 2;


    /*
     * vislib::math::AbstractViewFrustum<T, S>::IDX_NEAR
     */
    template<class T, class S>
    const UINT_PTR AbstractViewFrustum<T, S>::IDX_NEAR = 5;


    /*
     * vislib::math::AbstractViewFrustum<T, S>::IDX_RIGHT
     */    
    template<class T, class S>
    const UINT_PTR AbstractViewFrustum<T, S>::IDX_RIGHT = 3;


    /*
     * vislib::math::AbstractViewFrustum<T, S>::IDX_TOP
     */
    template<class T, class S>
    const UINT_PTR AbstractViewFrustum<T, S>::IDX_TOP = 1;
    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTVIEWFRUSTUM_H_INCLUDED */

