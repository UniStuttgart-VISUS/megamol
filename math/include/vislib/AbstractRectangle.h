/*
 * AbstractRectangle.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ASTRACTRECTANGLE_H_INCLUDED
#define VISLIB_ASTRACTRECTANGLE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/assert.h"
#include "vislib/Dimension.h"
#include "vislib/mathfunctions.h"
#include "vislib/memutils.h"
#include "vislib/Point.h"
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

        /** A bitmask representing all rectangle borders. */
        static const UINT32 BORDER_ALL;

        /** A bitmask representing the bottom border of the rectangle. */
        static const UINT32 BORDER_BOTTOM;

        /** A bitmask representing the left border of the rectangle. */
        static const UINT32 BORDER_LEFT;

        /** A bitmask representing the right border of the rectangle. */
        static const UINT32 BORDER_RIGHT;

        /** A bitmask representing the top border of the rectangle. */
        static const UINT32 BORDER_TOP;

        /** Dtor. */
        ~AbstractRectangle(void);

        /**
         * Answer the area covered by the rectangle.
         *
         * @return The area covered by the rectangle.
         */
        inline T Area(void) const {
            return (this->Width() * this->Height());
        }

        /**
         * Answer the aspect ratio of the rectangle.
         *
         * @return The aspect ratio of the rectangle.
         */
        inline double AspectRatio(void) const {
            if (IsEqual<double>(this->Height(), 0.0)) {
                return 0.0;
            }
            return double(this->Width()) / double(this->Height());
        }

        /**
         * Provide direct access to the y-coordinate of the left/bottom point.
         *
         * @return A reference to the y-coordinate of the left/bottom point.
         */
        inline const T& Bottom(void) const {
            return this->bounds[IDX_BOTTOM];
        }

        /** 
         * Answer the center point of the rectangle.
         *
         * @return The center point of the rectangle.
         */
        Point<T, 2> CalcCenter(void) const;

        /**
         * Answer whether the point 'point' lies within the rectangle.
         *
         * @param point         The point to be tested.
         * @param includeBorder An arbitrary combination of the BORDER_LEFT, 
         *                      BORDER_BOTTOM, BORDER_RIGHT and BORDER_TOP 
         *                      bitmasks. If the border bit is set, points lying 
         *                      on the respective border are regarded as in the
         *                      rectangle, otherwise they are out. Defaults to
         *                      zero, i. e. no border is included.
         *
         * @return True if the point lies within the rectangle, false otherwise.
         */
        template<class Sp>
        bool Contains(const AbstractPoint<T, 2, Sp>& point, 
            const UINT32 includeBorder = 0) const;

        /**
         * Answer the height of the rectangle.
         *
         * @return The height of the rectangle.
         */
        inline T Height(void) const {
            return (this->bounds[IDX_TOP] > this->bounds[IDX_BOTTOM])
                ? (this->bounds[IDX_TOP] - this->bounds[IDX_BOTTOM])
                : (this->bounds[IDX_BOTTOM] - this->bounds[IDX_TOP]);
        }

        /**
         * Answer the width of the rectangle.
         *
         * @return The width of the rectangle.
         */
        inline T Width(void) const {
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
        inline Point<T, 3> GetLeftBottom(void) const {
            return Point<T, 2>(this->bounds[IDX_LEFT], 
                this->bounds[IDX_BOTTOM]);
        }

        /**
         * Answer the left/top point of the rectangle.
         *
         * @return The left/top point.
         */
        inline Point<T, 2>GetLeftTop(void) const {
            return Point<T, 2>(this->bounds[IDX_LEFT], 
                this->bounds[IDX_BOTTOM]);
        }

        /** 
         * Answer the rectangle origin (left/bottom).
         *
         * @return The origin point.
         */
        inline Point<T, 2> GetOrigin(void) const {
            return Point<T, 2>(this->bounds[IDX_LEFT], 
                this->bounds[IDX_BOTTOM]);
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
        inline Point<T, 2> GetRightBottom(void) const {
            return Point<T, 2>(this->bounds[IDX_RIGHT], 
                this->bounds[IDX_BOTTOM]);
        }

        /**
         * Answer the right/top point of the rectangle.
         *
         * @return The right/top point.
         */
        inline Point<T, 2> GetRightTop(void) const {
            return Point<T, 2>(this->bounds[IDX_RIGHT], 
                this->bounds[IDX_TOP]);
        }

        /**
         * Answer the dimensions of the rectangle.
         *
         * @return The dimensions of the rectangle.
         */
        inline Dimension<T, 2> GetSize(void) const {
            return Dimension<T, 2>(this->Width(), this->Height());
        }

        /**
         * Answer the y-coordinate of the right/top point.
         *
         * @return The y-coordinate of the right/top point.
         */
        inline const T& GetTop(void) const {
            return this->bounds[IDX_TOP];
        }

        /**
         * Increases the size of the rectangle to include the given point. 
         * Implicitly calls 'EnforcePositiveSize'.
         *
         * @param p The point to be included.
         */
        template<class Tp, class Sp>
        inline void GrowToPoint(const AbstractPoint<Tp, 2, Sp>& p) {
            this->GrowToPoint(p.X(), p.Y());
        }

        /**
         * Increases the size of the rectangle to include the given point.
         * Implicitly calls 'EnforcePositiveSize'.
         *
         * @param x The x coordinate of the point to be included.
         * @param y The y coordinate of the point to be included.
         */
        template<class Tp>
        void GrowToPoint(const Tp& x, const Tp& y);

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
         * Answer whether this rectangle and 'rect' intersect.
         *
         * @param rect The rectangle to be tested.
         *
         * @return true fi there is an intersection, false otherwise.
         */
        template<class Sp> 
        bool Intersects(const AbstractRectangle<T, Sp>& rect) const;

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
         * Move the bottom side of the rectangle (keeping its size) to the
         * specified location.
         *
         * @param bottom The new bottom border.
         */
        inline void MoveBottomTo(const T& bottom) {
            this->Move(static_cast<T>(0), bottom - this->Bottom());
        }

        /**
         * Move the left side of the rectangle (keeping its size) to the
         * specified location.
         *
         * @param left The new left border.
         */
        inline void MoveLeftTo(const T& left) {
            this->Move(left - this->Left(), static_cast<T>(0));
        }

        /**
         * Move the right side of the rectangle (keeping its size) to the
         * specified location.
         *
         * @param right The new right border.
         */
        inline void MoveRightTo(const T& right) {
            this->Move(right - this->Right(), static_cast<T>(0));
        }

        /**
         * Move the left/bottom side of the rectangle (keeping its size) to the 
         * specified location.
         *
         * Implicitly calls 'EnforcePositiveSize'.
         *
         * @param left   The new left border.
         * @param bottom The new bottom border.
         */
        inline void MoveTo(const T& left, const T& bottom) {
            this->EnforcePositiveSize();
            this->Move(left - this->Left(), bottom - this->Bottom());
        }

        /**
         * Move the top side of the rectangle (keeping its size) to the
         * specified location.
         *
         * @param top The new top border.
         */
        inline void MoveTopTo(const T& top) {
            this->Move(static_cast<T>(0), top - this->Top());
        }

        /**
         * Directly access the internal bounds of the rectangle.
         *
         * @return A pointer to the rectangle bounds.
         */
        inline const T *PeekBounds(void) const {
            return this->bounds;
        }

        /**
         * Directly access the internal bounds of the rectangle.
         *
         * @return A pointer to the rectangle bounds.
         */
        inline T *PeekBounds(void) {
            return this->bounds;
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
        inline void Set(const T& left, const T& bottom, const T& right, 
                const T& top) {
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
         * Set new rectangle bounds.
         *
         * @param left   The x-coordinate of the left/bottom point.
         * @param bottom The y-coordinate of the left/bottom point.
         * @param right  The x-coordinate of the right/top point.
         * @param top    The y-coordinate of the right/top point.
         */
        inline void SetFromBounds(const T& left, const T& bottom, 
                const T& right, const T& top) {
            this->Set(left, bottom, right, top);
        }

        /**
         * Set new rectangle bounds.
         *
         * @param left   The x-coordinate of the left/bottom point.
         * @param bottom The y-coordinate of the left/bottom point.
         * @param width  The width of the rectangle.
         * @param height The height of the rectangle.
         */
        inline void SetFromSize(const T& left, const T& bottom, 
                const T& width, const T& height) {
            this->bounds[IDX_BOTTOM] = bottom;
            this->bounds[IDX_LEFT] = left;
            this->bounds[IDX_RIGHT] = left + width;
            this->bounds[IDX_TOP] = bottom +height;
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

        /**
         * Set a new size of the rectangle.
         *
         * @param size The new rectangle dimensions.
         */
        template<class Tp, class Sp>
        inline void SetSize(const AbstractDimension<Tp, 2, Sp>& size) {
            this->SetWidth(static_cast<T>(size.GetWidth()));
            this->SetHeight(static_cast<T>(size.GetHeight()));
        }

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
     * vislib::math::AbstractRectangle<T, S>::BORDER_ALL
     */
    template<class T, class S>
    const UINT32 AbstractRectangle<T, S>::BORDER_ALL
        = AbstractRectangle<T, S>::BORDER_LEFT
        | AbstractRectangle<T, S>::BORDER_BOTTOM
        | AbstractRectangle<T, S>::BORDER_RIGHT
        | AbstractRectangle<T, S>::BORDER_TOP;


    /*
     * vislib::math::AbstractRectangle<T, S>::BORDER_BOTTOM
     */
    template<class T, class S>
    const UINT32 AbstractRectangle<T, S>::BORDER_BOTTOM 
        = 1 << AbstractRectangle<T, S>::IDX_BOTTOM;

    /*
     * vislib::math::AbstractRectangle<T, S>::BORDER_LEFT
     */
    template<class T, class S>
    const UINT32 AbstractRectangle<T, S>::BORDER_LEFT 
        = 1 << AbstractRectangle<T, S>::IDX_LEFT;


    /*
     * vislib::math::AbstractRectangle<T, S>::BORDER_RIGHT
     */
    template<class T, class S>
    const UINT32 AbstractRectangle<T, S>::BORDER_RIGHT 
        = 1 << AbstractRectangle<T, S>::IDX_RIGHT;


    /*
     * vislib::math::AbstractRectangle<T, S>::BORDER_TOP
     */
    template<class T, class S>
    const UINT32 AbstractRectangle<T, S>::BORDER_TOP
        = 1 << AbstractRectangle<T, S>::IDX_TOP;


    /*
     * vislib::math::AbstractRectangle<T, S>::~AbstractRectangle
     */
    template<class T, class S>
    AbstractRectangle<T, S>::~AbstractRectangle(void) {
    }


    /*
     * vislib::math::AbstractRectangle<T, S>::CalcCenter
     */
    template<class T, class S> 
    Point<T, 2> AbstractRectangle<T, S>::CalcCenter(void) const {
        return Point<T, 2>(
            this->bounds[IDX_LEFT] + this->Width() / static_cast<T>(2),
            this->bounds[IDX_BOTTOM] + this->Height() / static_cast<T>(2));
    }


    /*
     * AbstractRectangle<T, S>::Contains
     */
    template<class T, class S> 
    template<class Sp>
    bool AbstractRectangle<T, S>::Contains(const AbstractPoint<T, 2, Sp>& point,
            const UINT32 includeBorder) const {

        if ((point.X() < this->bounds[IDX_LEFT])
                || ((((includeBorder & BORDER_LEFT) == 0))
                && (point.X() == this->bounds[IDX_LEFT]))) {
            /* Point is left of rectangle. */
            return false;
        }

        if ((point.Y() < this->bounds[IDX_BOTTOM])
                || ((((includeBorder & BORDER_BOTTOM) == 0))
                && (point.Y() == this->bounds[IDX_BOTTOM]))) {
            /* Point is below rectangle. */
            return false;
        }

        if ((point.X() > this->bounds[IDX_RIGHT])
                || ((((includeBorder & BORDER_RIGHT) == 0))
                && (point.X() == this->bounds[IDX_RIGHT]))) {
            /* Point is right of rectangle. */
            return false;
        }

        if ((point.Y() > this->bounds[IDX_TOP])
                || ((((includeBorder & BORDER_TOP) == 0))
                && (point.Y() == this->bounds[IDX_TOP]))) {
            /* Point is above rectangle. */
            return false;
        }

        return true;
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
     * vislib::math::AbstractRectangle<T>::GrowToPoint
     */
    template<class T, class S> template<class Tp>
    void AbstractRectangle<T, S>::GrowToPoint(const Tp& x, const Tp& y) {
        this->EnforcePositiveSize();
        if (this->bounds[IDX_LEFT] > x) {
            this->bounds[IDX_LEFT] = x;
        }
        if (this->bounds[IDX_BOTTOM] > y) {
            this->bounds[IDX_BOTTOM] = y;
        }
        if (this->bounds[IDX_RIGHT] < x) {
            this->bounds[IDX_RIGHT] = x;
        }
        if (this->bounds[IDX_TOP] < y) {
            this->bounds[IDX_TOP] = y;
        }
    }


    /*
     * vislib::math::AbstractRectangle<T, S>::Intersect
     */
    template<class T, class S>
    template<class Sp>
    bool AbstractRectangle<T, S>::Intersect(
            const AbstractRectangle<T, Sp>& rect) {
        T bottom = Max(this->bounds[IDX_BOTTOM], rect.Bottom());
        T left = Max(this->bounds[IDX_LEFT], rect.Left());
        T right = Min(this->bounds[IDX_RIGHT], rect.Right());
        T top = Min(this->bounds[IDX_TOP], rect.Top());
        
        if ((top < bottom) || (right < left)) {
            this->SetNull();
            return false;

        } else {
            this->Set(left, bottom, right, top);
            return true;
        }
    }


    /*
     * vislib::math::AbstractRectangle<T, S>::Intersects
     */
    template<class T, class S>
    template<class Sp>
    bool AbstractRectangle<T, S>::Intersects(
            const AbstractRectangle<T, Sp>& rect) const {
        T bottom = Max(this->bounds[IDX_BOTTOM], rect.Bottom());
        T left = Max(this->bounds[IDX_LEFT], rect.Left());
        T right = Min(this->bounds[IDX_RIGHT], rect.Right());
        T top = Min(this->bounds[IDX_TOP], rect.Top());
        
        return ((top >= bottom) && (right >= left));
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

        if (static_cast<void *>(this) != static_cast<const void *>(&rhs)) {
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
    bool AbstractRectangle<T, S>::operator ==(
            const AbstractRectangle& rhs) const {
        return (IsEqual<T>(this->bounds[IDX_BOTTOM], rhs.bounds[IDX_BOTTOM])
            && IsEqual<T>(this->bounds[IDX_LEFT], rhs.bounds[IDX_LEFT]) 
            && IsEqual<T>(this->bounds[IDX_RIGHT], rhs.bounds[IDX_RIGHT]) 
            && IsEqual<T>(this->bounds[IDX_TOP], rhs.bounds[IDX_TOP]));
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
    template<class T, class S> 
    const UINT_PTR AbstractRectangle<T, S>::IDX_LEFT = 0;


    /*
     * vislib::math::AbstractRectangle<T, S>::IDX_TOP
     */
    template<class T, class S> 
    const UINT_PTR AbstractRectangle<T, S>::IDX_TOP = 3;

} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ASTRACTRECTANGLE_H_INCLUDED */
