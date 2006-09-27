/*
 * Rectangle.h  27.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_RECTANGLE_H_INCLUDED
#define VISLIB_RECTANGLE_H_INCLUDED
#if _MSC_VER > 1000
#pragma once
#endif /* _MSC_VER > 1000 */


#include "vislib/AbstractRectangle.h"


namespace vislib {
namespace math {


    /**
     * This class represents a rectangle.
     */
    template<class T> class Rectangle : public AbstractRectangle<T, T[4]> {

    public:

        /**
         * Create an empty rectangle in the origin.
         */ 
        inline Rectangle(void) {
            this->bounds[0] = this->bounds[1] = this->bounds[2] 
                = this->bounds[3] = static_cast<T>(0);
        }

        /**
         * Create a new rectangle.
         *
         * @param left   The left border of the rectangle.
         * @param bottom The bottom border of the rectangle.
         * @param right  The right border of the rectangle.
         * @param top    The top border of the rectangle.
         */
        inline Rectangle(const T& left, const T& bottom, const T& right, 
                const T& top) {
            this->bounds[IDX_LEFT] = left;
            this->bounds[IDX_BOTTOM] = bottom;
            this->bounds[IDX_RIGHT] = right;
            this->bounds[IDX_TOP] = top;
        }

        /**
         * Construct a rectangle from an array holding its bounds. The array
         * 'bounds' holds in this order to following borders of the rectangle:
         * left, bottom, right, top.
         *
         * @return The left, bottom, right and top border of the rectangle in
         *         a consecutive order.
         */
        explicit inline Rectangle(const T *bounds) {
            ::memcpy(this->bounds, bounds, 4 * sizeof(T));
        }

        //template<class Sp>
        //Rectangle(const AbstractPoint2D<T, Sp>& origin, 
        //    const Dimension2D<T, Sp> size);

        /**
         * Copy ctor.
         *
         * @param rhs The object to clone.
         */
        inline Rectangle(const Rectangle& rhs) {
            ::memcpy(this->bounds, rhs.bounds, 4 * sizeof(T));
        }

        /**
         * Allow arbitrary rectangle to rectangle conversions.
         *
         * @param rhs The object to clone.
         */
        template<class Tp, class Sp>
        inline Rectangle(const AbstractRectangle<Tp, Sp>& rhs) {
            this->bounds[IDX_BOTTOM] = static_cast<T>(rhs.Bottom());
            this->bounds[IDX_LEFT] = static_cast<T>(rhs.Left());
            this->bounds[IDX_RIGHT] = static_cast<T>(rhs.Right());
            this->bounds[IDX_TOP] = static_cast<T>(rhs.Top());
        }


        /** Dtor. */
        ~Rectangle(void);

        /**
         * Assigment operator. This operator never creates an alias, even for
         * shallow rectangles!
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline Rectangle& operator =(const Rectangle& rhs) {
            AbstractRectangle<T, T[4]>::operator =(rhs);
            return *this;
        }

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
        inline Rectangle& operator =(const AbstractRectangle<Tp, Sp>& rhs) {
            AbstractRectangle<T, T[4]>::operator =(rhs);
            return *this;
        }
    };


/*
 * vislib::math::Rectangle<T>::~Rectangle
 */
template<class T> Rectangle<T>::~Rectangle(void) {
}

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_RECTANGLE_H_INCLUDED */
