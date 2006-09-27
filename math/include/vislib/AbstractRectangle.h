/*
 * AbstractRectangle.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ASTRACTRECTANGLE_H_INCLUDED
#define VISLIB_ASTRACTRECTANGLE_H_INCLUDED
#if _MSC_VER > 1000
#pragma once
#endif /* _MSC_VER > 1000 */


#include "vislib/assert.h"
#include "vislib/mathfunctions.h"
#include "vislib/memutils.h"
#include "vislib/Point2D.h"
#include "vislib/types.h"
#include "vislib/utils.h"


namespace vislib {
namespace math {


    /**
     * This class represents a rectangle. The origin is located in the 
     * left/bottom corner of the rectangle.
     *
     * The rectangle can be instantiated for different types builtin number
     * types T and storage classes S. For a rectangle that holds its own 
     * corner points, use T[4] as storage class (this is the default). For a
     * rectangle that does not own the storage (we call this "shallow 
     * rectangle"), use T * for S.
     */
    template<class T, class S = T[4]> class AbstractRectangle {

    public:

        /** Dtor. */
        ~AbstractRectangle(void);

        /**
         * Provide direct access to the y-coordinate of the left/bottom point.
         *
         * @return A reference to the y-coordinate of the left/bottom point.
         */
        inline const T& Bottom(void) const {
            return this->bounds[IDX_BOTTOM];
        }

        /**
         * Answer the area covered by the rectangle.
         *
         * @return The area covered by the rectangle.
         */
        inline T CalcArea(void) const {
            return (this->CalcWidth() * this->CalcHeight());
        }

        /** 
         * Answer the center point of the rectangle.
         *
         * @return The center point of the rectangle.
         */
        Point2D<T> CalcCenter(void) const;

        /**
         * Answer the height of the rectangle.
         *
         * @return The height of the rectangle.
         */
        inline T CalcHeight(void) const {
            return (this->bounds[IDX_TOP] > this->bounds[IDX_BOTTOM])
                ? (this->bounds[IDX_TOP] - this->bounds[IDX_BOTTOM])
                : (this->bounds[IDX_BOTTOM] - this->bounds[IDX_TOP]);
        }

        /**
         * Answer the width of the rectangle.
         *
         * @return The width of the rectangle.
         */
        inline T CalcWidth(void) const {
            return (this->bounds[IDX_RIGHT] > this->bounds[IDX_LEFT])
                ? (this->bounds[IDX_RIGHT] - this->bounds[IDX_LEFT])
                : (this->bounds[IDX_LEFT] - this->bounds[IDX_RIGHT]);
        }

        /**
         * Ensures that the size of the rectangle is positive by
         * swapping the left/right and/or top/bottom sides, if they
         * are in the wrong order.
         */
        void EnforcePositiveSize(void);

        /**
         * Answer the y-coordinate of the left/bottom point.
         *
         * @return The y-coordinate of the left/bottom point.
         */
        inline const T& GetBottom(void) const {
            return this->bounds[IDX_BOTTOM];
        }

        /**
         * Answer the x-coordinate of the left/bottom point.
         *
         * @return The x-coordinate of the left/bottom point.
         */
        inline const T& GetLeft(void) const {
            return this->bounds[IDX_LEFT];
        }

        /**
         * Answer the left/bottom point of the rectangle.
         *
         * @return The left/bottom point.
         */
        inline Point2D<T> GetLeftBottom(void) const {
            return Point2D<T>(this->bounds[IDX_LEFT], this->bounds[IDX_BOTTOM]);
        }

        /**
         * Answer the left/top point of the rectangle.
         *
         * @return The left/top point.
         */
        inline Point2D<T>GetLeftTop(void) const {
            return Point2D<T>(this->bounds[IDX_LEFT], this->bounds[IDX_BOTTOM]);
        }

        /** 
         * Answer the rectangle origin (left/bottom).
         *
         * @return The origin point.
         */
        inline Point2D<T> GetOrigin(void) const {
            return Point2D<T>(this->bounds[IDX_LEFT], this->bounds[IDX_BOTTOM]);
        }

        /**
         * Answer the y-coordinate of the right/top point.
         *
         * @return The y-coordinate of the right/top point.
         */
        inline const T& GetRight(void) const {
            return this->bounds[IDX_RIGHT];
        }

        /**
         * Answer the right/bottom point of the rectangle.
         *
         * @return The right/bottom point of the rectangle.
         */
        inline Point2D<T> GetRightBottom(void) const {
            return Point<T>(this->bounds[IDX_RIGHT], this->bounds[IDX_BOTTOM]);
        }

        /**
         * Answer the right/top point of the rectangle.
         *
         * @return The right/top point.
         */
        inline Point2D<T> GetRightTop(void) const {
            return Point2D<T>(this->bounds[IDX_RIGHT], this->bounds[IDX_TOP]);
        }

        // TODO
        ///**
        // * Answer the dimensions of the rectangle.
        // *
        // * @return The dimensions of the rectangle.
        // */
        //inline Dimension2D<T, S> GetSize(void) const {
        //    return Dimension2D<T, S>(this->CalcWidth(), this->CalcHeight());
        //}

        /**
         * Answer the y-coordinate of the right/top point.
         *
         * @return The y-coordinate of the right/top point.
         */
        inline const T& GetTop(void) const {
            return this->bounds[IDX_TOP];
        }

        /**
         * Set this rectangle to the intersection of itself and 'rect'.
         *
         * @param rect The rectangle to build the intersection with.
         *
         * @return true, if there is an intersection, false otherwise, i. e.
         *         if the rectangle is an empty one after that.
         */
        template<class Sp> bool Intersect(const AbstractRectangle<T, Sp>& rect);

        ///**
        // * Answer the intersection of this rectangle and 'rect'.
        // *
        // * @param rect The rectangle to build the intersection with.
        // *
        // * @return The intersection rectangle.
        // */
        //template<class Sp> inline AbstractRectangle<T, T[4]> Intersection(
        //        const AbstractRectangle<T, Sp>& rect) const {
        //    AbstractRectangle<T, T[4]> retval = *this;
        //    retval.Intersect(rect);
        //    return retval;
        //}

        /**
         * Answer whether the rectangle has no area.
         *
         * @return true, if the rectangle has no area, false otherwise.
         */
        inline bool IsEmpty(void) const {
            return (IsEqual<T>(this->bounds[IDX_LEFT], this->bounds[IDX_RIGHT])
                && IsEqual<T>(this->bounds[IDX_BOTTOM], this->bounds[IDX_TOP]));
        }

        /**
         * Provide direct access to the x-coordinate of the left/bottom point.
         *
         * @return A reference to the x-coordinate of the left/bottom point.
         */
        inline const T& Left(void) const {
            return this->bounds[IDX_LEFT];
        }

        /**
         * Move the rectangle.
         *
         * @param dx The offset in x-direction.
         * @param dy The offset in y-direction.
         */
        inline void Move(const T& dx, const T& dy) {
            this->bounds[IDX_BOTTOM] += dy;
            this->bounds[IDX_LEFT] += dx;
            this->bounds[IDX_RIGHT] += dx;
            this->bounds[IDX_TOP] += dy;
        }

        /**
         * Provide direct access to the x-coordinate of the right/top point.
         *
         * @return A reference to the x-coordinate of the right/top point.
         */
        inline const T& Right(void) const {
            return this->bounds[IDX_RIGHT];
        }

        /**
         * Set new rectangle bounds.
         *
         * @param left   The x-coordinate of the left/bottom point.
         * @param bottom The y-coordinate of the left/bottom point.
         * @param right  The x-coordinate of the right/top point.
         * @param top    The y-coordinate of the right/top point.
         */
        void Set(const T& left, const T& bottom, const T& right, const T& top) {
            this->bounds[IDX_BOTTOM] = bottom;
            this->bounds[IDX_LEFT] = left;
            this->bounds[IDX_RIGHT] = right;
            this->bounds[IDX_TOP] = top;
        }

        /**
         * Change the y-coordinate of the left/bottom point.
         *
         * @param bottom The new y-coordinate of the left/bottom point.
         */
        inline void SetBottom(const T& bottom) {
            this->bounds[IDX_BOTTOM] = bottom;
        }

        /**
         * Set a new height.
         *
         * @param height The new height of the rectangle.
         */
        inline void SetHeight(const T& height) {
            this->bounds[IDX_TOP] = this->bounds[IDX_BOTTOM] + height;
        }

        /**
         * Change the x-coordinate of the left/bottom point.
         *
         * @param left The new x-coordinate of the left/bottom point.
         */
        inline void SetLeft(const T& left) {
            this->bounds[IDX_LEFT] = left;
        }

        /**
         * Make the rectangle an empty rectangle a (0, 0).
         */
        inline void SetNull(void) {
            this->Set(static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),
                static_cast<T>(0));
        }

        /**
         * Change the y-coordinate of the right/top point.
         *
         * @param right The new y-coordinate of the right/top point.
         */
        inline void SetRight(const T& right) {
            this->bounds[IDX_RIGHT] = right;
        }

        // TODO
        ///**
        // * Set a new size of the rectangle.
        // *
        // * @param size The new rectangle dimensions.
        // */
        //inline void SetSize(const Dimension2D<T, S>& size) {
        //    this->SetWidth(size.GetWidth());
        //    this->SetHeight(size.GetHeight());
        //}

        /**
         * Change the y-coordinate of the right/top point.
         *
         * @param top The new y-coordinate of the right/top point.
         */
        inline void SetTop(const T& top) {
            this->bounds[IDX_TOP] = top;
        }

        /**
         * Set a new width.
         *
         * @param width The new width of the rectangle.
         */
        inline void SetWidth(const T& width) {
            this->bounds[IDX_RIGHT] = this->bounds[IDX_LEFT] + width;
        }

        /**
         * Swap the left and the right x-coordinate.
         */
        inline void SwapLeftRight(void) {
            Swap(this->bounds[IDX_LEFT], this->bounds[IDX_RIGHT]);
        }

        /**
         * Swap the top and the bottom y-coordinate.
         */
        inline void SwapTopBottom(void) {
            Swap(this->bounds[IDX_TOP], this->bounds[IDX_BOTTOM]);
        }

        /**
         * Provide direct access to the y-coordinate of the right/top point.
         *
         * @return A reference to the y-coordinate of the right/top point.
         */
        inline const T& Top(void) const {
            return this->bounds[IDX_TOP];
        }

        /**
         * Set this rectangle to the bounding rectangle of itself and 'rect'.
         *
         * @param rect The rectangle to compute the union with.
         */
        template<class Sp> void Union(const AbstractRectangle<T, Sp>& rect);

        /**
         * Assigment operator. This operator never creates an alias, even for
         * shallow rectangles!
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractRectangle& operator =(const AbstractRectangle& rhs);

        /**
         * Assigment operator. This operator never creates an alias, even for
         * shallow rectangles!
         *
         * This assignment allows for arbitrary rectangle to rectangle
         * conversions.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        AbstractRectangle& operator =(const AbstractRectangle<Tp, Sp>& rhs);

        /**
         * Test for equality. The operator uses the IsEqual<T, S> function for each
         * member.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this vector are equal, false otherwise.
         */
        bool operator ==(const AbstractRectangle& rhs) const;

        /**
         * Test for inequality. The operator uses the IsEqual<T, S> function for each
         * member.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this vector are not equal, false otherwise.
         */
        inline bool operator !=(const AbstractRectangle& rhs) const {
            return !(*this == rhs);
        }

    protected:

        /** The index of the bottom coordinate in 'bounds'. */
        static const UINT_PTR IDX_BOTTOM;

        /** The index of the right coordinate in 'bounds'. */
        static const UINT_PTR IDX_RIGHT;

        /** The index of the left coordinate in 'bounds'. */
        static const UINT_PTR IDX_LEFT;

        /** The index of the top coordinate in 'bounds'. */
        static const UINT_PTR IDX_TOP;

        /**
         * Forbidden default ctor. This does nothing.
         */
        inline AbstractRectangle(void) {}

        /** 
         * The bounds of the rectangle in following order: left, bottom, 
         * right, top. 
         */
        S bounds;

    };


/*
 * vislib::math::AbstractRectangle<T, S>::~AbstractRectangle
 */
template<class T, class S> AbstractRectangle<T, S>::~AbstractRectangle(void) {
}


/*
 * vislib::math::AbstractRectangle<T, S>::CalcCenter
 */
template<class T, class S> 
Point2D<T> AbstractRectangle<T, S>::CalcCenter(void) const {
    return Point2D<T>(
        this->bounds[IDX_LEFT] + this->CalcWidth() / static_cast<T>(2),
        this->bounds[IDX_BOTTOM] + this->CalcHeight() / static_cast<T>(2));
}


/*
 * vislib::math::AbstractRectangle<T, S>::EnforcePositiveSize
 */
template<class T, class S> 
void AbstractRectangle<T, S>::EnforcePositiveSize(void) {
    if (this->bounds[IDX_BOTTOM] > this->bounds[IDX_TOP]) {
        Swap(this->bounds[IDX_BOTTOM], this->bounds[IDX_TOP]);
    }

    if (this->bounds[IDX_LEFT] > this->bounds[IDX_RIGHT]) {
        Swap(this->bounds[IDX_LEFT], this->bounds[IDX_RIGHT]);
    }
}


/*
 * vislib::math::AbstractRectangle<T, S>::Intersect
 */
template<class T, class S>
template<class Sp>
bool AbstractRectangle<T, S>::Intersect(const AbstractRectangle<T, Sp>& rect) {
    T bottom = MAX(this->bounds[IDX_BOTTOM], rect.Bottom());
    T left = MAX(this->bounds[IDX_LEFT], rect.Left());
    T right = MIN(this->bounds[IDX_RIGHT], rect.Right());
    T top = MIN(this->bounds[IDX_TOP], rect.Top());
    
    if ((top < bottom) || (right < left)) {
        this->SetNull();
        return false;

    } else {
        this->Set(left, bottom, right, top);
        return true;
    }
}


/*
 * vislib::math::AbstractRectangle<T, S>::Union
 */
template<class T, class S>
template<class Sp>
void AbstractRectangle<T, S>::Union(const AbstractRectangle<T, Sp>& rect) {
    T rectBottom, rectLeft, rectRight, rectTop;
    
    if (rect.Bottom() < rect.Top()) {
        rectBottom = rect.Bottom();
        rectTop = rect.Top();
    } else {
        rectBottom = rect.Top();
        rectTop = rect.Bottom();
    }

    if (rect.Left() < rect.Right()) {
        rectLeft = rect.Left();
        rectRight = rect.Right();
    } else {
        rectLeft = rect.Right();
        rectRight = rect.Left();
    }

    this->EnforcePositiveSize();

    ASSERT(this->bounds[IDX_LEFT] <= this->bounds[IDX_RIGHT]);
    ASSERT(this->bounds[IDX_BOTTOM] <= this->bounds[IDX_TOP]);
    ASSERT(rectLeft <= rectRight);
    ASSERT(rectBottom <= rectTop);

    if (rectLeft < this->bounds[IDX_LEFT]) {
        this->bounds[IDX_LEFT] = rectLeft;
    }

    if (rectRight > this->bounds[IDX_RIGHT]) {
        this->bounds[IDX_RIGHT] = rectRight;
    }

    if (rectTop > this->bounds[IDX_TOP]) {
        this->bounds[IDX_TOP] = rectTop;
    }

    if (rectBottom < this->bounds[IDX_BOTTOM]) {
        this->bounds[IDX_BOTTOM] = rectBottom;
    }
}


/*
 * vislib::math::AbstractRectangle<T, S>::operator =
 */
template<class T, class S>
AbstractRectangle<T, S>& AbstractRectangle<T, S>::operator =(
        const AbstractRectangle& rhs) {

    if (this != &rhs) {
        ::memcpy(this->bounds, rhs.bounds, 4 * sizeof(T));
    }

    return *this;
}


/*
 * vislib::math::AbstractRectangle<T, S>::operator =
 */
template<class T, class S>
template<class Tp, class Sp>
AbstractRectangle<T, S>& AbstractRectangle<T, S>::operator =(
        const AbstractRectangle<Tp, Sp>& rhs) {

    if (static_cast<void *>(this) != static_cast<void *>(&rhs)) {
        this->bounds[IDX_BOTTOM] = static_cast<T>(rhs.Bottom());
        this->bounds[IDX_LEFT] = static_cast<T>(rhs.Left());
        this->bounds[IDX_RIGHT] = static_cast<T>(rhs.Right());
        this->bounds[IDX_TOP] = static_cast<T>(rhs.Top());
    }

    return *this;
}


/*
 * vislib::math::AbstractRectangle<T, S>::operator ==
 */
template<class T, class S> 
bool AbstractRectangle<T, S>::operator ==(const AbstractRectangle& rhs) const {
    return (IsEqual<T>(this->bounds[IDX_BOTTOM], rhs.bounds[IDX_BOTTOM])
        && IsEqual<T>(this->bounds[IDX_LEFT] == rhs.bounds[IDX_LEFT]) 
        && IsEqual<T>(this->bounds[IDX_RIGHT] == rhs.bounds[IDX_RIGHT]) 
        && IsEqual<T>(this->bounds[IDX_TOP] == rhs.bounds[IDX_TOP]));
}


/*
 * vislib::math::AbstractRectangle<T, S>::IDX_BOTTOM
 */
template<class T, class S> 
const UINT_PTR AbstractRectangle<T, S>::IDX_BOTTOM = 1;


/*
 * vislib::math::AbstractRectangle<T, S>::IDX_RIGHT
 */
template<class T, class S> 
const UINT_PTR AbstractRectangle<T, S>::IDX_RIGHT = 2;


/*
 * vislib::math::AbstractRectangle<T, S>::IDX_LEFT
 */
template<class T, class S> const UINT_PTR AbstractRectangle<T, S>::IDX_LEFT = 0;


/*
 * vislib::math::AbstractRectangle<T, S>::IDX_TOP
 */
template<class T, class S> const UINT_PTR AbstractRectangle<T, S>::IDX_TOP = 3;

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_ASTRACTRECTANGLE_H_INCLUDED */
