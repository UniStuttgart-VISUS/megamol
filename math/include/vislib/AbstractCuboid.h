/*
 * AbstractCuboid.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCUBOID_H_INCLUDED
#define VISLIB_ABSTRACTCUBOID_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/assert.h"
#include "vislib/Dimension.h"
#include "vislib/Point.h"
#include "vislib/types.h"
#include "vislib/mathfunctions.h"


namespace vislib {
namespace math {

    /**
     * This class represents a cubiod. The cuboid has its origin in
     * the left/bottom/back corner, like the OpenGL coordinate system. 
     *
     * @author Christoph Mueller
     */
    template<class T, class S> class AbstractCuboid {

    public:

        /** Dtor. */
        ~AbstractCuboid(void);

        /**
         * Provide direct access to the z-coordinate of the left/bottom/back 
         * point.
         *
         * @return A reference to the z-coordinate of the right/bottom/back
         *         point.
         */
        inline const T& Back(void) const {
            return this->bounds[IDX_BACK];
        }

        /**
         * Provide direct access to the y-coordinate of the left/bottom/back 
         * point.
         *
         * @return A reference to the y-coordinate of the right/bottom/back
         *         point.
         */
        inline const T& Bottom(void) const {
            return this->bounds[IDX_BOTTOM];
        }

        /** 
         * Answer the center point of the cuboid.
         *
         * @return The center point of the cuboid.
         */
        Point<T, 3> CalcCenter(void) const;

        /**
         * Answer the depth of the cuboid.
         *
         * @return The depth of the cuboid.
         */
        inline T Depth(void) const {
            return (this->bounds[IDX_FRONT] > this->bounds[IDX_BACK]) 
                ? (this->bounds[IDX_FRONT] - this->bounds[IDX_BACK])
                : (this->bounds[IDX_BACK] - this->bounds[IDX_FRONT]);
        }

        /**
         * Ensures that the size of the cuboid is positive by
         * swapping the left/right and/or top/bottom and/or front/back sides,
         * if they are in the wrong order.
         */
        void EnforcePositiveSize(void);

        /**
         * Provide direct access to the z-coordinate of the right/top/front
         * point.
         *
         * @return A reference to the z-coordinate of the right/top/front
         *         point.
         */
        inline const T& Front(void) const {
            return this->bounds[IDX_FRONT];
        }

        /**
         * Answer the z-coordinate of the left/bottom/back point.
         *
         * @return The z-coordinate of the left/bottom/back point.
         */
        inline const T& GetBack(void) const {
            return this->bounds[IDX_BACK];
        }
        /**
         * Answer the y-coordinate of the left/bottom/back point.
         *
         * @return The y-coordinate of the left/bottom/back point.
         */
        inline const T& GetBottom(void) const {
            return this->bounds[IDX_BOTTOM];
        }

        /**
         * Answer the z-coordinate of the right/top/front point.
         *
         * @return The z-coordinate of the right/top/front point.
         */
        inline const T& GetFront(void) const {
            return this->bounds[IDX_FRONT];
        }

        /**
         * Answer the x-coordinate of the left/bottom/back point.
         *
         * @return The x-coordinate of the left/bottom/back point.
         */
        inline const T& GetLeft(void) const {
            return this->bounds[IDX_LEFT];
        }

        /**
         * Answer the left/bottom/back point of the cuboid.
         *
         * @return The left/bottom/back cuboid.
         */
        inline Point<T, 3> GetLeftBottomBack(void) const {
            return Point<T, 3>(this->bounds[IDX_LEFT], this->bounds[IDX_BOTTOM],
                this->bounds[IDX_BACK]);
        }

        /**
         * Answer the left/bottom/front point of the cuboid.
         *
         * @return The left/bottom/front cuboid.
         */
        inline Point<T, 3> GetLeftBottomFront(void) const {
            return Point<T, 3>(this->bounds[IDX_LEFT], this->bounds[IDX_BOTTOM],
                this->bounds[IDX_FRONT]);
        }

        /**
         * Answer the left/top/back point of the cuboid.
         *
         * @return The left/top/back point.
         */
        inline Point<T, 3> GetLeftTopBack(void) const {
            return Point<T, 3>(this->bounds[IDX_LEFT], this->bounds[IDX_TOP],
                this->bounds[IDX_BACK]);
        }

        /**
         * Answer the left/top/front point of the cuboid.
         *
         * @return The left/top/front point.
         */
        inline Point<T, 3> GetLeftTopFront(void) const {
            return Point<T, 3>(this->bounds[IDX_LEFT], this->bounds[IDX_TOP],
                this->bounds[IDX_FRONT]);
        }

        /**
         * Answer the height of the cuboid.
         *
         * @return The height of the cuboid.
         */
        inline T Height(void) const {
            return (this->bounds[IDX_TOP] > this->bounds[IDX_BOTTOM]) 
                ? (this->bounds[IDX_TOP] - this->bounds[IDX_BOTTOM])
                : (this->bounds[IDX_BOTTOM]- this->bounds[IDX_TOP]);
        }

        /** 
         * Answer the cuboid origin (left/bottom/back).
         *
         * @return The origin point.
         */
        inline Point<T, 3> GetOrigin(void) const {
            return Point<T, 3>(this->bounds[IDX_LEFT], this->bounds[IDX_BOTTOM],
                this->bounds[IDX_BACK]);
        }

        /**
         * Answer the y-coordinate of the right/top/front point.
         *
         * @return The y-coordinate of the right/top/front point.
         */
        inline const T& GetRight(void) const {
            return this->bounds[IDX_RIGHT];
        }

        /**
         * Answer the right/bottom/back point of the cuboid.
         *
         * @return The right/bottom/back point of the cuboid.
         */
        inline Point<T, 3> GetRightBottomBack(void) const {
            return Point<T, 3>(this->bounds[IDX_RIGHT], 
                this->bounds[IDX_BOTTOM], this->bounds[IDX_BACK]);
        }

        /**
         * Answer the right/bottom/front point of the cuboid.
         *
         * @return The right/bottom/front point of the cuboid.
         */
        inline Point<T, 3> GetRightBottomFront(void) const {
            return Point<T, 3>(this->bounds[IDX_RIGHT], 
                this->bounds[IDX_BOTTOM], this->bounds[IDX_FRONT]);
        }

        /**
         * Answer the right/top/back point of the cuboid.
         *
         * @return The right/top/back point.
         */
        inline Point<T, 3> GetRightTopBack(void) const {
            return Point<T, 3>(this->bounds[IDX_RIGHT], 
                this->bounds[IDX_TOP], this->bounds[IDX_BACK]);
        }

        /**
         * Answer the right/top/front point of the cuboid.
         *
         * @return The right/top/front point.
         */
        inline Point<T, 3> GetRightTopFront(void) const {
           return Point<T, 3>(this->bounds[IDX_RIGHT], 
                this->bounds[IDX_TOP], this->bounds[IDX_FRONT]);
        }

        /**
         * Answer the dimensions of the cuboid.
         *
         * @return The dimensions of the cuboid.
         */
        inline Dimension<T, 3> GetSize(void) const {
            return Dimension<T, 3>(this->Width(), this->Height(),
                this->Depth());
        }

        /**
         * Answer the y-coordinate of the right/top/front point.
         *
         * @return The y-coordinate of the right/top/front point.
         */
        inline const T& GetTop(void) const {
            return this->bounds[IDX_TOP];
        }

        /**
         * Answer whether the cuboid has no area.
         *
         * @return true, if the cuboid has no area, false otherwise.
         */
        inline bool IsEmpty(void) const {
            return (IsEqual<T>(this->bounds[IDX_LEFT], this->bounds[IDX_RIGHT])
                && IsEqual<T>(this->bounds[IDX_BOTTOM], this->bounds[IDX_TOP])
                && IsEqual<T>(this->bounds[IDX_BACK], this->bounds[IDX_FRONT]));
        }

        /**
         * Provide direct access to the x-coordinate of the left/bottom/back
         * point.
         *
         * @return A reference to the x-coordinate of the left/bottom/back
         *         point.
         */
        inline const T& Left(void) const {
            return this->bounds[IDX_LEFT];
        }

        /**
         * Move the cuboid.
         *
         * @param dx The offset in x-direction.
         * @param dy The offset in y-direction.
         * @param dz The offset in z-direction.
         */
        inline void Move(const T& dx, const T& dy, const T& dz) {
            this->bounds[IDX_BACK] += dz;
            this->bounds[IDX_BOTTOM] += dy;
            this->bounds[IDX_FRONT] += dz;
            this->bounds[IDX_LEFT] += dx;
            this->bounds[IDX_RIGHT] += dx;
            this->bounds[IDX_TOP] += dy;
        }

        ///**
        // * Move the cuboid to a new origin.
        // *
        // * @param x The x-coordinate of the new origin.
        // * @param y The y-coordinate of the new origin.
        // * @param z The z-coordinate of the nre origin.
        // */
        //inline void MoveTo(const T& x, const T& y, const T& z) {
        //    this->Move(x - this->left, y - this->bottom, z - this->back);
        //}

        //inline void MoveTo(const Point<T, 3>& origin) {
        //    this->Move(origin[0] - this->left, origin[1] - this->bottom, 
        //        origin[2] - this->back);
        //}

        /**
         * Directly access the internal bounds of the cuboid.
         *
         * @return A pointer to the cuboid bounds.
         */
        inline const T *PeekBounds(void) const {
            return this->bounds;
        }

        /**
         * Provide direct access to the x-coordinate of the right/top/front
         * point.
         *
         * @return A reference to the x-coordinate of the right/top/front.
         */
        inline const T& Right(void) const {
            return this->bounds[IDX_RIGHT];
        }

        /**
         * Set new couboid bounds.
         *
         * @param left   The x-coordinate of the left/bottom/back point.
         * @param bottom The y-coordinate of the left/bottom/back point.
         * @param back   The z-coordinate of the left/bottom/back point.
         * @param right  The x-coordinate of the right/top/front point.
         * @param top    The y-coordinate of the right/top/front point.
         * @param front  The z-coordinate of the right/top/front point.
         */
        inline void Set(const T& left, const T& bottom, const T& back, 
                const T& right, const T& top, const T& front) {
            this->bounds[IDX_LEFT] = left;
            this->bounds[IDX_BOTTOM] = bottom;
            this->bounds[IDX_BACK] = back;
            this->bounds[IDX_RIGHT] = right;
            this->bounds[IDX_TOP] = top;
            this->bounds[IDX_FRONT] = front;
        }

        /**
         * Change the z-coordinate of the left/bottom/back point.
         *
         * @param bottom The new z-coordinate of the left/bottom/back point.
         */
        inline void SetBack(const T& back) {
            this->bounds[IDX_BACK] = back;
        }

        /**
         * Change the y-coordinate of the left/bottom/back point.
         *
         * @param bottom The new y-coordinate of the left/bottom/back point.
         */
        inline void SetBottom(const T& bottom) {
            this->bounds[IDX_BOTTOM] = bottom;
        }

        /**
         * Set a new depth.
         *
         * @param depth The new depth of the cuboid.
         */
        inline void SetDepth(const T& depth) {
            this->bounds[IDX_FRONT] = this->bounds[IDX_BACK] + depth;
        }

        /**
         * Change the z-coordinate of the left/bottom/front point.
         *
         * @param front The new y-coordinate of the left/bottom/front point.
         */
        inline void SetFront(const T& front) {
            this->bounds[IDX_FRONT] = front;
        }

        /**
         * Set a new height.
         *
         * @param height The new height of the cuboid.
         */
        inline void SetHeight(const T& height) {
            this->bounds[IDX_TOP] = this->bounds[IDX_BOTTOM] + height;
        }

        /**
         * Change the x-coordinate of the left/bottom/back point.
         *
         * @param left The new x-coordinate of the left/bottom/back point.
         */
        inline void SetLeft(const T& left) {
            this->bounds[IDX_LEFT] = left;
        }

        /**
         * Make the cuboid an empty cuboid a (0, 0).
         */
        inline void SetNull(void) {
            this->Set(static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),
                static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));
        }

        /**
         * Change the y-coordinate of the right/top/front point.
         *
         * @param right The new y-coordinate of the right/top/front point.
         */
        inline void SetRight(const T& right) {
            this->bounds[IDX_RIGHT] = right;
        }

        /**
         * Set a new size of the cuboid.
         *
         * @param size The new cuboid dimensions.
         */
        template<class Tp, class Sp>
        inline void SetSize(const AbstractDimension<Tp, 3, Sp>& size) {
            this->SetWidth(size.GetWidth());
            this->SetHeight(size.GetHeight());
            this->SetDepth(size.GetDepth());
        }

        /**
         * Change the y-coordinate of the right/top/front point.
         *
         * @param top The new y-coordinate of the right/top/front point.
         */
        inline void SetTop(const T& top) {
            this->bounds[IDX_TOP] = top;
        }

        /**
         * Set a new width.
         *
         * @param width The new width of the cuboid.
         */
        inline void SetWidth(const T& width) {
            this->bounds[IDX_RIGHT] = this->bounds[IDX_LEFT] + width;
        }

        /**
         * Swap the front and the back z-coordinate.
         */
        inline void SwapFrontBack(void) {
            Swap(this->bounds[IDX_FRONT], this->bounds[IDX_BACK]);
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
         * Provide direct access to the y-coordinate of the right/top/front 
         * point.
         *
         * @return A reference to the y-coordinate of the right/top/front point.
         */
        inline const T& Top(void) const {
            return this->bounds[IDX_TOP];
        }

        /**
         * Set this cuboid to the bounding cuboid of itself and 'cuboid'.
         *
         * @param cuboid The cuboid to compute the union with.
         */
        template<class Sp> void Union(const AbstractCuboid<T, Sp>& cuboid);

        /**
         * Answer the volume of the cuboid.
         *
         * @return The volume of the cuboid.
         */
        inline T Volume(void) const {
            return (this->Width() * this->Height() * this->Depth());
        }

        /**
         * Answer the width of the cuboid.
         *
         * @return The width of the cuboid.
         */
        inline T Width(void) const {
            return (this->bounds[IDX_RIGHT] > this->bounds[IDX_LEFT])
                ? (this->bounds[IDX_RIGHT] - this->bounds[IDX_LEFT])
                : (this->bounds[IDX_LEFT] - this->bounds[IDX_RIGHT]);
        }

        /**
         * Assigment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractCuboid& operator =(const AbstractCuboid& rhs);

        /**
         * Assigment operator. This operator never creates an alias, even for
         * shallow cuboids!
         *
         * This assignment allows for arbitrary cuboid to cuboid conversions.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        AbstractCuboid& operator =(const AbstractCuboid<Tp, Sp>& rhs);

        /**
         * Test for equality. The operator uses the == operator for T for each
         * member.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this vector are equal, false otherwise.
         */
        bool operator ==(const AbstractCuboid& rhs) const;

        /**
         * Test for inequality. The operator uses the == operator for T for each
         * member.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this vector are not equal, false otherwise.
         */
        inline bool operator !=(const AbstractCuboid& rhs) const {
            return !(*this == rhs);
        }

        //vislib::math::Point<T, 3> operator[](const int i) const;

    protected:

        /** The index of the back coordinate in 'bounds'. */
        static const UINT_PTR IDX_BACK;

        /** The index of the bottom coordinate in 'bounds'. */
        static const UINT_PTR IDX_BOTTOM;

        /** The index of the front coordinate in 'bounds'. */
        static const UINT_PTR IDX_FRONT;

        /** The index of the right coordinate in 'bounds'. */
        static const UINT_PTR IDX_RIGHT;

        /** The index of the left coordinate in 'bounds'. */
        static const UINT_PTR IDX_LEFT;

        /** The index of the top coordinate in 'bounds'. */
        static const UINT_PTR IDX_TOP;

        /**
         * Forbidden default ctor. This does nothing.
         */
        inline AbstractCuboid(void) {}

        /** 
         * The bounds of the cuboid, in the following order: left, bottom, back,
         * right, top, front.
         */
        S bounds;

    };


    /*
     * vislib::math::AbstractCuboid<T>::~AbstractCuboid
     */
    template<class T, class S> 
    vislib::math::AbstractCuboid<T, S>::~AbstractCuboid(void) {
    }


    /*
     * vislib::math::AbstractCuboid<T>::CalcCenter
     */
    template<class T, class S> 
    vislib::math::Point<T, 3> 
    vislib::math::AbstractCuboid<T, S>::CalcCenter(void) const {
        return Point<T, 3>(
            this->bounds[IDX_LEFT] + this->Width() / static_cast<T>(2),
            this->bounds[IDX_BOTTOM] + this->Height() / static_cast<T>(2),
            this->bounds[IDX_BACK] + this->Depth() / static_cast<T>(2));
    }


    /*
     * vislib::math::AbstractCuboid<T>::EnforcePositiveSize
     */
    template<class T, class S> 
    void vislib::math::AbstractCuboid<T, S>::EnforcePositiveSize(void) {
        if (this->bounds[IDX_BOTTOM] > this->bounds[IDX_TOP]) {
            Swap(this->bounds[IDX_TOP], this->bounds[IDX_BOTTOM]);
        }

        if (this->bounds[IDX_LEFT] > this->bounds[IDX_RIGHT]) {
            Swap(this->bounds[IDX_LEFT], this->bounds[IDX_RIGHT]);
        }

        if (this->bounds[IDX_BACK] > this->bounds[IDX_FRONT]) {
            Swap(this->bounds[IDX_FRONT], this->bounds[IDX_BACK]);
        }
    }


    /*
     * vislib::math::AbstractCuboid<T>::Union
     */
    template<class T, class S>
    template<class Sp> void vislib::math::AbstractCuboid<T, S>::Union(
            const AbstractCuboid<T, Sp>& cuboid) {
        T cubBack, cubBottom, cubFront, cubLeft, cubRight, cubTop;
        
        if (cuboid.Bottom() < cuboid.Top()) {
            cubBottom = cuboid.Bottom();
            cubTop = cuboid.Top();
        } else {
            cubBottom = cuboid.Top();
            cubTop = cuboid.Bottom();
        }

        if (cuboid.Left() < cuboid.Right()) {
            cubLeft = cuboid.Left();
            cubRight = cuboid.Right();
        } else {
            cubLeft = cuboid.Right();
            cubRight = cuboid.Left();
        }

        if (cuboid.Back() < cuboid.Front()) {
            cubBack = cuboid.Back();
            cubFront = cuboid.Front();
        } else {
            cubBack = cuboid.Back();
            cubFront = cuboid.Front();
        }

        this->EnforcePositiveSize();

        ASSERT(this->bounds[IDX_LEFT] <= this->bounds[IDX_RIGHT]);
        ASSERT(this->bounds[IDX_BOTTOM] <= this->bounds[IDX_TOP]);
        ASSERT(this->bounds[IDX_BACK] <= this->bounds[IDX_FRONT]);
        ASSERT(cubLeft <= cubRight);
        ASSERT(cubBottom <= cubTop);
        ASSERT(cubBack <= cubFront);

        if (cubLeft < this->bounds[IDX_LEFT]) {
            this->bounds[IDX_LEFT] = cubLeft;
        }

        if (cubRight > this->bounds[IDX_RIGHT]) {
            this->bounds[IDX_RIGHT] = cubRight;
        }

        if (cubTop > this->bounds[IDX_TOP]) {
            this->bounds[IDX_TOP] = cubTop;
        }

        if (cubBottom < this->bounds[IDX_BOTTOM]) {
            this->bounds[IDX_BOTTOM] = cubBottom;
        }

        if (cubBack < this->bounds[IDX_BACK]) {
            this->bounds[IDX_BACK] = cubBack;
        }

        if (cubFront > this->bounds[IDX_FRONT]) {
            this->bounds[IDX_FRONT] = cubFront;
        }
    }


    /*
     * vislib::math::AbstractCuboid<T>::operator =
     */
    template<class T, class S> 
    vislib::math::AbstractCuboid<T, S>& 
    vislib::math::AbstractCuboid<T, S>::operator =(const AbstractCuboid& rhs) {

        if (this != &rhs) {
            ::memcpy(this->bounds, rhs.bounds, 6 * sizeof(T));
        }

        return *this;
    }


    /*
     * vislib::math::AbstractCuboid<T, S>::operator =
     */
    template<class T, class S>
    template<class Tp, class Sp>
    AbstractCuboid<T, S>& AbstractCuboid<T, S>::operator =(
            const AbstractCuboid<Tp, Sp>& rhs) {

        if (static_cast<void *>(this) != static_cast<const void *>(&rhs)) {
            this->bounds[IDX_BACK] = static_cast<T>(rhs.Back());
            this->bounds[IDX_BOTTOM] = static_cast<T>(rhs.Bottom());
            this->bounds[IDX_FRONT] = static_cast<T>(rhs.Front());
            this->bounds[IDX_LEFT] = static_cast<T>(rhs.Left());
            this->bounds[IDX_RIGHT] = static_cast<T>(rhs.Right());
            this->bounds[IDX_TOP] = static_cast<T>(rhs.Top());
        }

        return *this;
    }


    /*
     * vislib::math::AbstractCuboid<T>::operator ==
     */
    template<class T, class S> 
    bool vislib::math::AbstractCuboid<T, S>::operator ==(
            const AbstractCuboid& rhs) const {
        return (IsEqual<T>(this->bounds[IDX_BACK], rhs.bounds[IDX_BACK])
            && IsEqual<T>(this->bounds[IDX_BOTTOM], rhs.bounds[IDX_BOTTOM])
            && IsEqual<T>(this->bounds[IDX_FRONT], rhs.bounds[IDX_FRONT])
            && IsEqual<T>(this->bounds[IDX_LEFT], rhs.bounds[IDX_LEFT]) 
            && IsEqual<T>(this->bounds[IDX_RIGHT], rhs.bounds[IDX_RIGHT]) 
            && IsEqual<T>(this->bounds[IDX_TOP], rhs.bounds[IDX_TOP]));
    }


    /*
     * vislib::math::AbstractCuboid<T, S>::IDX_BACK
     */
    template<class T, class S> 
    const UINT_PTR vislib::math::AbstractCuboid<T, S>::IDX_BACK = 2;


    /*
     * vislib::math::AbstractCuboid<T, S>::IDX_BOTTOM
     */
    template<class T, class S> 
    const UINT_PTR vislib::math::AbstractCuboid<T, S>::IDX_BOTTOM = 1;
    

    /*
     * vislib::math::AbstractCuboid<T, S>::IDX_FRONT
     */
    template<class T, class S> 
    const UINT_PTR vislib::math::AbstractCuboid<T, S>::IDX_FRONT = 5;
    

    /*
     * vislib::math::AbstractCuboid<T, S>::IDX_RIGHT
     */
    template<class T, class S> 
    const UINT_PTR vislib::math::AbstractCuboid<T, S>::IDX_RIGHT = 3;


    /*
     * vislib::math::AbstractCuboid<T, S>::IDX_LEFT
     */
    template<class T, class S> 
    const UINT_PTR vislib::math::AbstractCuboid<T, S>::IDX_LEFT = 0;


    /*
     * vislib::math::AbstractCuboid<T, S>::IDX_TOP
     */
    template<class T, class S> 
    const UINT_PTR vislib::math::AbstractCuboid<T, S>::IDX_TOP = 4;


    ///*
    // * vislib::math::AbstractCuboid<T>::operator []
    // */
    //template<class T, class S> 
    //vislib::math::Point<T, 3> vislib::math::Cuboid<T>::operator [](const int i) const {
    //    T tmp[3];

    //    switch (i) {

    //        case 0:
    //            tmp[0] = this->left;
    //            tmp[1] = this->bottom;
    //            tmp[2] = this->back;
    //            break;

    //        case 1:
    //            tmp[0] = this->right;
    //            tmp[1] = this->bottom;
    //            tmp[2] = this->back;
    //            break;

    //        case 2:
    //            tmp[0] = this->right;
    //            tmp[1] = this->bottom;
    //            tmp[2] = this->front;
    //            break;

    //        case 3:
    //            tmp[0] = this->left;
    //            tmp[1] = this->bottom;
    //            tmp[2] = this->front;
    //            break;

    //        case 4:
    //            tmp[0] = this->left;
    //            tmp[1] = this->top;
    //            tmp[2] = this->back;
    //            break;

    //        case 5:
    //            tmp[0] = this->right;
    //            tmp[1] = this->top;
    //            tmp[2] = this->back;
    //            break;

    //        case 6:
    //            tmp[0] = this->right;
    //            tmp[1] = this->top;
    //            tmp[2] = this->front;
    //            break;

    //        case 7:
    //            tmp[0] = this->left;
    //            tmp[1] = this->top;
    //            tmp[2] = this->front;
    //            break;

    //        default:
    //            throw util::RangeException(i, 0, 7, __FILE__, __LINE__);
    //    }

    //    return vislib::math::Point<T, 3>(tmp);
    //}

} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCUBOID_H_INCLUDED */
