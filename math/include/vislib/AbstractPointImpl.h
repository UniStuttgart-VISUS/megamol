/*
 * AbstractPointImpl.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTPOINTIMPL_H_INCLUDED
#define VISLIB_ABSTRACTPOINTIMPL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractVector.h"
#include "vislib/assert.h"
#include "vislib/mathfunctions.h"
#include "vislib/mathtypes.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Vector.h"


namespace vislib {
namespace math {


    /**
     * TODO: comment class
     */
    template<class T, unsigned int D, class S, 
            template<class T, unsigned int D, class S> class C> 
            class AbstractPointImpl {

    public:

        /** Dtor. */
        ~AbstractPointImpl(void);

        /**
         * Answer the distance from this point to 'toPoint'.
         *
         * @param toPoint The point to calculate the distance to.
         *
         * @return The distance between the two points.
         */
        template<class Tp, class Sp>
        T Distance(const C<Tp, D, Sp>& toPoint) const;

        /**
         * Answer in which halfspace 'point' lies in respect to the plane
         * defined by 'this' point and the 'normal' vector.
         *
         * @param normal The normal vector defining the plane.
         * @param point The point to be tested.
         *
         * @return The halfspace the point lies in.
         */
        template<class Tpp, class Spp, class Tpv, class Spv>
        HalfSpace Halfspace(const AbstractVector<Tpv, D, Spv>& normal,
            const C<Tpp, D, Spp>& point) const;

        /**
         * Interpolates a position between 'this' and 'rhs' based on the value
         * of 't'.
         *
         * @param rhs The right hand side operand.
         * @param t   The interpolation value. (Should be [0, 1])
         *
         * @return The interpolated position.
         */
        template<class Tp, class Sp, class Tp2>
        C<T, D, T[D]> Interpolate(const C<Tp, D, Sp>& rhs, Tp2 t) const;

        /**
         * Answer whether the point is the coordinate system origin (0, ..., 0).
         *
         * @return true, if the point is the origin, false otherwise.
         */
        bool IsOrigin(void) const;

        /**
         * Directly access the internal pointer holding the coordinates.
         * The object remains owner of the memory returned.
         *
         * @return The coordinates in an array.
         */
        inline T *PeekCoordinates(void) {
            return this->coordinates;
        }

        /**
         * Directly access the internal pointer holding the coordinates.
         * The object remains owner of the memory returned.
         *
         * @return The coordinates in an array.
         */
        inline const T *PeekCoordinates(void) const {
            return this->coordinates;
        }

        /**
         * Answer the square of the distance from this point to 'toPoint'.
         *
         * @param toPoint The point to calculate the distance to.
         *
         * @return The distance between the two points.
         */
        template<class Tp, class Sp>
        T SquareDistance(const C<Tp, D, Sp>& toPoint) const;

        /**
         * Assignment operator.
         *
         * This operation does <b>not</b> create aliases.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractPointImpl<T, D, S, C>& operator =(const C<T, D, S>& rhs);

        /**
         * Assigment for arbitrary points. A valid static_cast between T and Tp
         * is a precondition for instantiating this template.
         *
         * This operation does <b>not</b> create aliases. 
         *
         * If the two operands have different dimensions, the behaviour is as 
         * follows: If the left hand side operand has lower dimension, the 
         * highest (Dp - D) dimensions are discarded. If the left hand side
         * operand has higher dimension, the missing dimensions are filled with 
         * zero components.
         *
         * Subclasses must ensure that sufficient memory for the 'coordinates'
         * member has been allocated before calling this operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        template<class Tp, unsigned int Dp, class Sp>
        AbstractPointImpl<T, D, S, C>& operator =(const C<Tp, Dp, Sp>& rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        bool operator ==(const C<T, D, S>& rhs) const;

        /**
         * Test for equality of arbitrary points. This operation uses the
         * IsEqual function of the left hand side operand. Note that points 
         * with different dimensions are never equal.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this point are equal, false otherwise.
         */
        template<class Tp, unsigned int Dp, class Sp>
        bool operator ==(const C<Tp, Dp, Sp>& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const C<T, D, S>& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Test for inequality of arbitrary points. See operator == for further
         * details.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this vector are not equal, false otherwise.
         */
        template<class Tp, unsigned int Dp, class Sp>
        inline bool operator !=(const C<Tp, Dp, Sp>& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Move the point along the vector 'rhs'.
         *
         * @param rhs The direction vector to move the point to.
         *
         * @return A copy of this point moved by rhs.
         */
        template<class Tp, class Sp>
        C<T, D, T[D]> operator +(const AbstractVector<Tp, D, Sp>& rhs) const;

        /**
         * Move the point along the vector 'rhs'.
         *
         * @param rhs The direction vector to move the point to.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        AbstractPointImpl<T, D, S, C>& operator +=(
            const AbstractVector<Tp, D, Sp>& rhs);

        /**
         * Move the point along the negative vector 'rhs'.
         *
         * @param rhs The direction vector to move the point against.
         *
         * @return A copy of this point moved by rhs.
         */
        template<class Tp, class Sp>
        C<T, D, T[D]> operator -(const AbstractVector<Tp, D, Sp>& rhs) const;

        /**
         * Move the point along the negative vector 'rhs'.
         *
         * @param rhs The direction vector to move the point against.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        AbstractPointImpl<T, D, S, C>& operator -=(
            const AbstractVector<Tp, D, Sp>& rhs);

        /**
         * Subtract 'rhs' from this point.
         *
         * @param rhs The right hand side operand.
         *
         * @return A vector pointing from this point to 'rhs'.
         */
        template<class Tp, class Sp>
        Vector<T, D> operator -(const C<Tp, D, Sp>& rhs) const;

        /**
         * Directly access the 'i'th coordinate of the point.
         *
         * @param i The index of the coordinate within [0, D[.
         *
         * @return A reference to the x-coordinate for 0, 
         *         the y-coordinate for 1, etc.
         *
         * @throws OutOfRangeException, if 'i' is not within [0, D[.
         */
        T& operator [](const int i);

        /**
         * Answer the coordinates of the point.
         *
         * @param i The index of the coordinate within [0, D[.
         *
         * @return The x-coordinate for 0, the y-coordinate for 1, etc.
         *
         * @throws OutOfRangeException, if 'i' is not within [0, D[.
         */
        T operator [](const int i) const;

        /**
         * Cast to Vector. The resulting vector is the position vector of
         * this point.
         *
         * @return The position vector of the point.
         */
        inline operator Vector<T, D>(void) const {
            return Vector<T, D>(this->coordinates);
        }

    protected:

        /**
         * Disallow instances of this class. This ctor does nothing!
         */
        inline AbstractPointImpl(void) {};

        /** 
         * The coordinates of the point. This can be a T * pointer or a T[D]
         * static array.
         */
        S coordinates;
    };


    /*
     * vislib::math::AbstractPointImpl<T, D, S>::~AbstractPointImpl
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    AbstractPointImpl<T, D, S, C>::~AbstractPointImpl(void) {
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S>::Distance
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    template<class Tp, class Sp>
    T AbstractPointImpl<T, D, S, C>::Distance(
            const C<Tp, D, Sp>& toPoint) const {
        return static_cast<T>(::sqrt(static_cast<double>(
            this->SquareDistance(toPoint))));
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S, C>::Halfspace
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    template<class Tpp, class Spp, class Tpv, class Spv>
    HalfSpace AbstractPointImpl<T, D, S, C>::Halfspace(
            const AbstractVector<Tpv, D, Spv>& normal,
            const C<Tpp, D, Spp>& point) const {

        T val = static_cast<T>(0);
        for (unsigned int d = 0; d < D; d++) {
            val += static_cast<T>(normal[d]) * (static_cast<T>(point[d])
                - this->coordinates[d]);
        }

        if (IsEqual(val, static_cast<T>(0))) {
            return HALFSPACE_IN_PLANE;

        } else if (val > static_cast<T>(0)) {
            return HALFSPACE_POSITIVE;//HALFSPACE_NEGATIVE;

        } else if (val < static_cast<T>(0)) {
            return HALFSPACE_NEGATIVE;//HALFSPACE_POSITIVE;
    
        } else {
            ASSERT(false);      // Should never happen.
            return HALFSPACE_IN_PLANE;
        }        
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S>::Interpolate
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    template<class Tp, class Sp, class Tp2>
    C<T, D, T[D]> AbstractPointImpl<T, D, S, C>::Interpolate(
            const C<Tp, D, Sp>& rhs, Tp2 t) const {
        C<T, D, T[D]> retval;
        Tp2 at = static_cast<Tp2>(1) - t;

        for (unsigned int d = 0; d < D; d++) {
            retval.coordinates[d] = static_cast<Tp>(
                static_cast<Tp2>(this->coordinates[d]) * at
                + static_cast<Tp2>(rhs[d]) * t);
        }

        return retval;
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S, C>::IsOrigin
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    bool AbstractPointImpl<T, D, S, C>::IsOrigin(void) const {
        for (unsigned int i = 0; i < D; i++) {
            if (!IsEqual<T>(this->coordinates[i], 0)) {
                return false;
            }
        }

        return true;
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S>::SquareDistance
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    template<class Tp, class Sp>
    T AbstractPointImpl<T, D, S, C>::SquareDistance(
            const C<Tp, D, Sp>& toPoint) const {
        T retval = static_cast<T>(0);

        for (unsigned int i = 0; i < D; i++) {
            retval += static_cast<T>(Sqr(toPoint.coordinates[i] - this->coordinates[i]));
        }

        return retval;
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S>::operator =
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    AbstractPointImpl<T, D, S, C>& AbstractPointImpl<T, D, S, C>::operator =(
            const C<T, D, S>& rhs) {
        if (this != &rhs) {
            ::memcpy(this->coordinates, rhs.coordinates, D * sizeof(T));
        }

        return *this;
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S>::operator =
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    template<class Tp, unsigned int Dp, class Sp>
    AbstractPointImpl<T, D, S, C>& AbstractPointImpl<T, D, S, C>::operator =(
           const C<Tp, Dp, Sp>& rhs) {

        if (static_cast<void *>(this) != static_cast<const void *>(&rhs)) {
            for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
                this->coordinates[d] = static_cast<T>(rhs[d]);
            }
            for (unsigned int d = Dp; d < D; d++) {
                this->coordinates[d] = static_cast<T>(0);
            }            
        }

        return *this;
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S>::operator ==
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    bool AbstractPointImpl<T, D, S, C>::operator ==(
            const C<T, D, S>& rhs) const {

        for (unsigned int d = 0; d < D; d++) {
            if (!IsEqual<T>(this->coordinates[d], rhs.coordinates[d])) {
                return false;
            }
        }

        return true;
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S>::operator ==
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    template<class Tp, unsigned int Dp, class Sp>
    bool AbstractPointImpl<T, D, S, C>::operator ==(
            const C<Tp, Dp, Sp>& rhs) const {

        if (D != Dp) {
            return false;
        }

        for (unsigned int d = 0; d < D; d++) {
            if (!IsEqual<T>(this->coordinates[d], rhs.coordinates[d])) {
                return false;
            }
        }

        return true;
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S>::operator +
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    template<class Tp, class Sp>
    C<T, D, T[D]> AbstractPointImpl<T, D, S, C>::operator +(
            const AbstractVector<Tp, D, Sp>& rhs) const {
        C<T, D, T[D]> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval.coordinates[d] = this->coordinates[d] 
                + static_cast<T>(rhs[d]);
        }

        return retval;
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S>::operator +=
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    template<class Tp, class Sp>
    AbstractPointImpl<T, D, S, C>& AbstractPointImpl<T, D, S, C>::operator +=(
           const AbstractVector<Tp, D, Sp>& rhs) {
        for (unsigned int d = 0; d < D; d++) {
            this->coordinates[d] += static_cast<T>(rhs[d]);
        }

        return *this;
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S>::operator -
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    template<class Tp, class Sp>
    C<T, D, T[D]> AbstractPointImpl<T, D, S, C>::operator -(
           const AbstractVector<Tp, D, Sp>& rhs) const {
        C<T, D, T[D]> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval.coordinates[d] = this->coordinates[d] 
                - static_cast<T>(rhs[d]);
        }

        return retval;
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S>::operator -=
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    template<class Tp, class Sp>
    AbstractPointImpl<T, D, S, C>& AbstractPointImpl<T, D, S, C>::operator -=(
           const AbstractVector<Tp, D, Sp>& rhs) {
        for (unsigned int d = 0; d < D; d++) {
            this->coordinates[d] -= static_cast<T>(rhs[d]);
        }

        return *this;
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S>::operator -
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    template<class Tp, class Sp>
    Vector<T, D> AbstractPointImpl<T, D, S, C>::operator -(
           const C<Tp, D, Sp>& rhs) const {
        Vector<T, D> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval[d] = this->coordinates[d] 
                - static_cast<T>(rhs.coordinates[d]);
        }

        return retval;
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S>::operator []
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    T& AbstractPointImpl<T, D, S, C>::operator [](const int i) {
        if ((i >= 0) && (i < static_cast<int>(D))) {
            return this->coordinates[i];
        } else {
            throw OutOfRangeException(i, 0, D - 1, __FILE__, __LINE__);
        }
    }


    /*
     * vislib::math::AbstractPointImpl<T, D, S>::operator []
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    T AbstractPointImpl<T, D, S, C>::operator [](const int i) const {
        if ((i >= 0) && (i < static_cast<int>(D))) {
            return this->coordinates[i];
        } else {
            throw OutOfRangeException(i, 0, D - 1, __FILE__, __LINE__);
        }
    }
    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTPOINTIMPL_H_INCLUDED */

