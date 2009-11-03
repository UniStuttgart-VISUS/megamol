/*
 * Point.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_POINT_H_INCLUDED
#define VISLIB_POINT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractPoint.h"


namespace vislib {
namespace math {

    /**
     * This is the implementation of an AbstractPoint that uses its own memory 
     * in a statically allocated array of dimension D. Usually, you want to use
     * this point class or derived classes.
     *
     * See documentation of AbstractPoint for further information about the 
     * vector classes.
     */
    template<class T, unsigned int D> 
    class Point : public AbstractPoint<T, D, T[D]> {

    public:

        /**
         * Create a point in the coordinate origin.
         */
        Point(void);

        /**
         * Create a new point initialised with 'coordinates'. 'coordinates' must
         * not be a NULL pointer. 
         *
         * @param coordinates The initial coordinates of the point.
         */
        explicit inline Point(const T *coordinates) : Super() {
            ASSERT(coordinates != NULL);
            ::memcpy(this->coordinates, coordinates, D * sizeof(T));
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Point(const Point& rhs) : Super() {
            ::memcpy(this->coordinates, rhs.coordinates, D * sizeof(T));
        }

        /**
         * Create a copy of 'rhs'. This ctor allows for arbitrary point to
         * point conversions.
         *
         * @param rhs The vector to be cloned.
         */
        template<class Tp, unsigned int Dp, class Sp>
        Point(const AbstractPoint<Tp, Dp, Sp>& rhs);

        /** Dtor. */
        ~Point(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline Point& operator =(const Point& rhs) {
            Super::operator =(rhs);
            return *this;
        }

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
        inline Point& operator =(const AbstractPoint<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    private:

        /** Super class typedef. */
        typedef AbstractPoint<T, D, T[D]> Super;
    };


    /*
     * vislib::math::Point<T, D>::Point
     */
    template<class T, unsigned int D>
    Point<T, D>::Point(void) : Super() {
        for (unsigned int d = 0; d < D; d++) {
            this->coordinates[d] = static_cast<T>(0);
        }
    }


    /*
     * vislib::math::Point<T, D>::Point
     */
    template<class T, unsigned int D>
    template<class Tp, unsigned int Dp, class Sp>
    Point<T, D>::Point(const AbstractPoint<Tp, Dp, Sp>& rhs) : Super() {
        for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
            this->coordinates[d] = static_cast<T>(rhs[d]);
        }
        for (unsigned int d = Dp; d < D; d++) {
            this->coordinates[d] = static_cast<T>(0);
        }
    }


    /*
     * vislib::math::Point<T, D>::~Point
     */
    template<class T, unsigned int D>
    Point<T, D>::~Point(void) {
    }


    /**
     * Partial template specialisation for two-dimensional point. This class
     * provides a constructor with separate components.
     */
    template<class T> 
    class Point<T, 2> : public AbstractPoint<T, 2, T[2]> {

    public:

        /** Behaves like primary class template. */
        Point(void);

        /** Behaves like primary class template. */
        explicit inline Point(const T *coordinates) : Super() {
            ASSERT(coordinates != NULL);
            ::memcpy(this->coordinates, coordinates, D * sizeof(T));
        }

        /**
         * Create a new point.
         *
         * @param x The x-coordinate. 
         * @param y The y-coordinate.
         */
        inline Point(const T& x, const T& y) : Super() {
            this->coordinates[0] = x;
            this->coordinates[1] = y;
        }

        /** Behaves like primary class template. */
        inline Point(const Point& rhs) : Super() {
            ::memcpy(this->coordinates, rhs.coordinates, D * sizeof(T));
        }

        /** Behaves like primary class template. */
        template<class Tp, unsigned int Dp, class Sp>
        Point(const AbstractPoint<Tp, Dp, Sp>& rhs);

        /** Behaves like primary class template. */
        ~Point(void);

        /** Behaves like primary class template. */
        inline Point& operator =(const Point& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /** Behaves like primary class template. */
        template<class Tp, unsigned int Dp, class Sp>
        inline Point& operator =(const AbstractPoint<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    private:

        /** The dimensionality of the point. */
        static const unsigned int D;

        /** Super class typedef. */
        typedef AbstractPoint<T, 2, T[2]> Super;

        
    };


    /*
     * vislib::math::Point<T, 2>::Point
     */
    template<class T> Point<T, 2>::Point(void) : Super() {
        for (unsigned int d = 0; d < D; d++) {
            this->coordinates[d] = static_cast<T>(0);
        }
    }


    /*
     * vislib::math::Point<T, 2>::Point
     */
    template<class T>
    template<class Tp, unsigned int Dp, class Sp>
    Point<T, 2>::Point(const AbstractPoint<Tp, Dp, Sp>& rhs) : Super() {
        for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
            this->coordinates[d] = static_cast<T>(rhs[d]);
        }
        for (unsigned int d = Dp; d < D; d++) {
            this->coordinates[d] = static_cast<T>(0);
        }
    }


    /*
     * vislib::math::Point<T, 2>::~Point
     */
    template<class T> Point<T, 2>::~Point(void) {
    }


    /*
     * vislib::math::Point<T, 2>::D
     */
    template<class T> const unsigned int Point<T, 2>::D = 2;


    /**
     * Partial template specialisation for three-dimensional point. This class
     * provides a constructor with separate components.
     */
    template<class T> 
    class Point<T, 3> : public AbstractPoint<T, 3, T[3]> {

    public:

        /** Behaves like primary class template. */
        Point(void);

        /** Behaves like primary class template. */
        explicit inline Point(const T *coordinates) : Super() {
            ASSERT(coordinates != NULL);
            ::memcpy(this->coordinates, coordinates, D * sizeof(T));
        }

        /**
         * Create a new point.
         *
         * @param x The x-coordinate. 
         * @param y The y-coordinate.
         * @param z The z-coordinate.
         */
        inline Point(const T& x, const T& y, const T& z) : Super() {
            this->coordinates[0] = x;
            this->coordinates[1] = y;
            this->coordinates[2] = z;
        }

        /** Behaves like primary class template. */
        inline Point(const Point& rhs) : Super() {
            ::memcpy(this->coordinates, rhs.coordinates, D * sizeof(T));
        }

        /** Behaves like primary class template. */
        template<class Tp, unsigned int Dp, class Sp>
        Point(const AbstractPoint<Tp, Dp, Sp>& rhs);

        /** Behaves like primary class template. */
        ~Point(void);

        /** Behaves like primary class template. */
        inline Point& operator =(const Point& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /** Behaves like primary class template. */
        template<class Tp, unsigned int Dp, class Sp>
        inline Point& operator =(const AbstractPoint<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    private:

        /** The dimensionality of the point. */
        static const unsigned int D;

        /** Super class typedef. */
        typedef AbstractPoint<T, 3, T[3]> Super;

        
    };


    /*
     * vislib::math::Point<T, 3>::Point
     */
    template<class T> Point<T, 3>::Point(void) : Super() {
        for (unsigned int d = 0; d < D; d++) {
            this->coordinates[d] = static_cast<T>(0);
        }
    }


    /*
     * vislib::math::Point<T, 3>::Point
     */
    template<class T>
    template<class Tp, unsigned int Dp, class Sp>
    Point<T, 3>::Point(const AbstractPoint<Tp, Dp, Sp>& rhs) : Super() {
        for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
            this->coordinates[d] = static_cast<T>(rhs[d]);
        }
        for (unsigned int d = Dp; d < D; d++) {
            this->coordinates[d] = static_cast<T>(0);
        }
    }


    /*
     * vislib::math::Point<T, 3>::~Point
     */
    template<class T> Point<T, 3>::~Point(void) {
    }


    /*
     * vislib::math::Point<T, 3>::D
     */
    template<class T> const unsigned int Point<T, 3>::D = 3;


    /**
     * Partial template specialisation for four-dimensional point. This class
     * provides a constructor with separate components.
     */
    template<class T> 
    class Point<T, 4> : public AbstractPoint<T, 4, T[4]> {

    public:

        /** Behaves like primary class template. */
        Point(void);

        /** Behaves like primary class template. */
        explicit inline Point(const T *coordinates) : Super() {
            ASSERT(coordinates != NULL);
            ::memcpy(this->coordinates, coordinates, D * sizeof(T));
        }

        /**
         * Create a new point.
         *
         * @param x The x-coordinate. 
         * @param y The y-coordinate.
         * @param z The z-coordinate.
         * @param w The w-coordinate.
         */
        inline Point(const T& x, const T& y, const T& z, const T& w)
                : Super() {
            this->coordinates[0] = x;
            this->coordinates[1] = y;
            this->coordinates[2] = z;
            this->coordinates[3] = w;
        }

        /** Behaves like primary class template. */
        inline Point(const Point& rhs) : Super() {
            ::memcpy(this->coordinates, rhs.coordinates, D * sizeof(T));
        }

        /** Behaves like primary class template. */
        template<class Tp, unsigned int Dp, class Sp>
        Point(const AbstractPoint<Tp, Dp, Sp>& rhs);

        /** Behaves like primary class template. */
        ~Point(void);

        /** Behaves like primary class template. */
        inline Point& operator =(const Point& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /** Behaves like primary class template. */
        template<class Tp, unsigned int Dp, class Sp>
        inline Point& operator =(const AbstractPoint<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    private:

        /** The dimensionality of the point. */
        static const unsigned int D;

        /** Super class typedef. */
        typedef AbstractPoint<T, 4, T[4]> Super;

        
    };


    /*
     * vislib::math::Point<T, 4>::Point
     */
    template<class T> Point<T, 4>::Point(void) : Super() {
        for (unsigned int d = 0; d < D; d++) {
            this->coordinates[d] = static_cast<T>(0);
        }
    }


    /*
     * vislib::math::Point<T, 4>::Point
     */
    template<class T>
    template<class Tp, unsigned int Dp, class Sp>
    Point<T, 4>::Point(const AbstractPoint<Tp, Dp, Sp>& rhs) : Super() {
        for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
            this->coordinates[d] = static_cast<T>(rhs[d]);
        }
        for (unsigned int d = Dp; d < D; d++) {
            this->coordinates[d] = static_cast<T>(0);
        }
    }


    /*
     * vislib::math::Point<T, 4>::~Point
     */
    template<class T> Point<T, 4>::~Point(void) {
        // intentionally empty
    }


    /*
     * vislib::math::Point<T, 4>::D
     */
    template<class T> const unsigned int Point<T, 4>::D = 4;



} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_POINT_H_INCLUDED */
