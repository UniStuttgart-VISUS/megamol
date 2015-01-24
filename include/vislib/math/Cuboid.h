/*
 * Cuboid.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CUBOID_H_INCLUDED
#define VISLIB_CUBOID_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractCuboid.h"


namespace vislib {
namespace math {

    /**
     * This class represents a cubiod. The cuboid has its origin in
     * the left/bottom/back corner, like the OpenGL coordinate system. 
     *
     * @author Christoph Mueller
     */
    template<class T> class Cuboid : public AbstractCuboid<T, T[6]> {

    public:

        /**
         * Create a zero sized cuboid beginning at (0, 0, 0).
         */
        inline Cuboid(void) : Super() {
            this->bounds[0] = this->bounds[1] = this->bounds[2] 
                = this->bounds[3] = this->bounds[4] = this->bounds[5]
                = static_cast<T>(0);
        }

        /**
         * Create a cuboid with the specified borders.
         *
         * @param left   The x-coordinate of the left/bottom/back point.
         * @param bottom The y-coordinate of the left/bottom/back point.
         * @param back   The z-coordinate of the left/bottom/back point.
         * @param right  The x-coordinate of the right/top/front point.
         * @param top    The y-coordinate of the right/top/front point.
         * @param front  The z-coordinate of the right/top/front point.
         */
        inline Cuboid(const T& left, const T& bottom, const T& back, 
                const T& right, const T& top, const T& front) : Super() {
            this->bounds[Super::IDX_LEFT] = left;
            this->bounds[Super::IDX_BOTTOM] = bottom;
            this->bounds[Super::IDX_BACK] = back;
            this->bounds[Super::IDX_RIGHT] = right;
            this->bounds[Super::IDX_TOP] = top;
            this->bounds[Super::IDX_FRONT] = front;
        }

        /**
         * Create a new cuboid having the specified origin (left, bottom, back)
         * and the specified dimension.
         *
         * @param origin The origin of the cuboid.
         * @param size   The dimension of the cuboid.
         */
        template<class Tp1, class Sp1, class Tp2, class Sp2>
        inline Cuboid(const AbstractPoint<Tp1, 3, Sp1>& origin, 
                const AbstractDimension<Tp2, 3, Sp2> size) {
            this->bounds[Super::IDX_LEFT] = this->bounds[Super::IDX_RIGHT] 
                = origin.X();
            this->bounds[Super::IDX_BOTTOM] = this->bounds[Super::IDX_TOP] 
                = origin.Y();
            this->bounds[Super::IDX_BACK] = this->bounds[Super::IDX_FRONT]
                = origin.Z();
            this->bounds[Super::IDX_RIGHT] += size.Width();
            this->bounds[Super::IDX_TOP] += size.Height();
            this->bounds[Super::IDX_FRONT] += size.Depth();
        }

        /**
         * Clone rhs.
         *
         * @param rhs The object to clone.
         */
        inline Cuboid(const Cuboid& rhs) {
            ::memcpy(this->bounds, rhs.bounds, 6 * sizeof(T));
        }

        /**
         * Allow arbitrary cuboid to cuboid conversions.
         *
         * @param rhs The object to clone.
         */
        template<class Tp, class Sp>
        inline Cuboid(const AbstractCuboid<Tp, Sp>& rhs) {
            this->bounds[Super::IDX_BOTTOM] = static_cast<T>(rhs.Bottom());
            this->bounds[Super::IDX_LEFT] = static_cast<T>(rhs.Left());
            this->bounds[Super::IDX_BACK] = static_cast<T>(rhs.Back());
            this->bounds[Super::IDX_RIGHT] = static_cast<T>(rhs.Right());
            this->bounds[Super::IDX_TOP] = static_cast<T>(rhs.Top());
            this->bounds[Super::IDX_FRONT] = static_cast<T>(rhs.Front());
        }
    
        /** Dtor. */
        ~Cuboid(void);

    protected:

        /** Super class typedef. */
        typedef AbstractCuboid<T, T[6]> Super;
    };


    /*
     * vislib::math::Cuboid<T>::~Cuboid
     */
    template<class T> Cuboid<T>::~Cuboid(void) {
    }


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CUBOID_H_INCLUDED */
