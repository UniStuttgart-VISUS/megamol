/*
 * ShallowRectangle.h  27.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHALLOWRECTANGLE_H_INCLUDED
#define VISLIB_SHALLOWRECTANGLE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractRectangle.h"


namespace vislib {
namespace math {


    /**
     * This class represents a shallow rectangle, that uses memory provided
     * by the caller.
     */
    template<class T> class ShallowRectangle 
            : public AbstractRectangle<T, T *> {

    public:

        /**
         * Construct a rectangle from an array holding its bounds. The array
         * 'bounds' holds in this order to following borders of the rectangle:
         * left, bottom, right, top.
         *
         * @param bounds The left, bottom, right and top border of the 
         *               rectangle in a consecutive order.
         */
        explicit inline ShallowRectangle(T *bounds) {
            this->bounds = bounds;
        }

        /**
         * Copy ctor. This ctor creates an alias!
         *
         * @param rhs The object to clone.
         */
        inline ShallowRectangle(const ShallowRectangle& rhs) {
            this->bounds = rhs.bounds;
        }

        /** Dtor. */
        ~ShallowRectangle(void);

        /**
         * Assigment operator. This operator never creates an alias, even for
         * shallow rectangles!
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline ShallowRectangle& operator =(const ShallowRectangle& rhs) {
            Super::operator =(rhs);
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
        inline ShallowRectangle& operator =(
                const AbstractRectangle<Tp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    public:

        /** Typedef for the super class. */
        typedef AbstractRectangle<T, T *> Super;
    };


    /*
     * vislib::math::ShallowRectangle<T>::~ShallowRectangle
     */
    template<class T> ShallowRectangle<T>::~ShallowRectangle(void) {
    }


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHALLOWRECTANGLE_H_INCLUDED */
