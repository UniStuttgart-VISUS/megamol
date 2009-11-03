/*
 * AbstractPoint.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTPOINT_H_INCLUDED
#define VISLIB_ABSTRACTPOINT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractPointImpl.h"


namespace vislib {
namespace math {

    /**
     * TODO: documentation
     */
    template<class T, unsigned int D, class S> class AbstractPoint 
            : public AbstractPointImpl<T, D, S, AbstractPoint> {

    public:

        /** Dtor. */
        ~AbstractPoint(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline AbstractPoint& operator =(const AbstractPoint& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Assigment for arbitrary vectors. A valid static_cast between T and Tp
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
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        template<class Tp, unsigned int Dp, class Sp>
        inline AbstractPoint& operator =(
                const AbstractPoint<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** Typedef for our super class. */
        typedef AbstractPointImpl<T, D, S, vislib::math::AbstractPoint> Super;

        /**
         * Disallow instances of this class. 
         */
        inline AbstractPoint(void) : Super() {}

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1, 
            template<class Tf2, unsigned int Df2, class Sf2> class Cf> 
            friend class AbstractPointImpl;
    };


    /*
     * vislib::math::AbstractPoint<T, D, S>::~AbstractPoint
     */
    template<class T, unsigned int D, class S>
    AbstractPoint<T, D, S>::~AbstractPoint(void) {
    }


    /**
     * Partial template specialisation for two-dimensional points. This 
     * implementation provides convenience access methods to the two 
     * coordinates.
     */
    template<class T, class S> class AbstractPoint<T, 2, S> 
            : public AbstractPointImpl<T, 2, S, AbstractPoint> {

    public:

        /** Behaves like primary class template. */
        ~AbstractPoint(void);

        /**
         * Answer the x-coordinate of the point.
         *
         * @return The x-coordinate of the point.
         */
        inline const T& GetX(void) const {
            return this->coordinates[0];
        }

        /**
         * Answer the y-coordinate of the point.
         *
         * @return The y-coordinate of the point.
         */
        inline const T& GetY(void) const {
            return this->coordinates[1];
        }

        /**
         * Set the coordinates ot the point.
         *
         * @param x The x-coordinate of the point.
         * @param y The y-coordinate of the point.
         */
        inline void Set(const T& x, const T& y) {
            this->coordinates[0] = x;
            this->coordinates[1] = y;
        }

        /**
         * Set the x-coordinate of the point.
         *
         * @param x The new x-coordinate.
         */
        inline void SetX(const T& x) {
            this->coordinates[0] = x;
        }

        /**
         * Set the y-coordinate of the point.
         *
         * @param y The new y-coordinate.
         */
        inline void SetY(const T& y) {
            this->coordinates[1] = y;
        }

        /**
         * Answer the x-coordinate of the point.
         *
         * @return The x-coordinate of the point.
         */
        inline const T& X(void) const {
            return this->coordinates[0];
        }

        /**
         * Answer the y-component of the point.
         *
         * @return The y-component of the point.
         */
        inline const T& Y(void) const {
            return this->coordinates[1];
        }

        /** Behaves like primary class template. */
        inline AbstractPoint& operator =(const AbstractPoint& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /** Behaves like primary class template. */
        template<class Tp, unsigned int Dp, class Sp>
        inline AbstractPoint& operator =(
                const AbstractPoint<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }


    protected:

        /** Typedef for our super class. */
        typedef AbstractPointImpl<T, 2, S, vislib::math::AbstractPoint> Super;

        /**
         * Disallow instances of this class. 
         */
        inline AbstractPoint(void) : Super() {}

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1, 
            template<class Tf2, unsigned int Df2, class Sf2> class Cf> 
            friend class AbstractPointImpl;
    };


    /*
     * vislib::math::AbstractPoint<T, 2, S>::~AbstractPoint
     */
    template<class T, class S> AbstractPoint<T, 2, S>::~AbstractPoint(void) {
    }


    /**
     * Partial template specialisation for three-dimensional points. This 
     * implementation provides convenience access methods to the three 
     * coordinates.
     */
    template<class T, class S> class AbstractPoint<T, 3, S> 
            : public AbstractPointImpl<T, 3, S, AbstractPoint> {

    public:

        /** Behaves like primary class template. */
        ~AbstractPoint(void);

        /**
         * Answer the x-coordinate of the point.
         *
         * @return The x-coordinate of the point.
         */
        inline const T& GetX(void) const {
            return this->coordinates[0];
        }

        /**
         * Answer the y-coordinate of the point.
         *
         * @return The y-coordinate of the point.
         */
        inline const T& GetY(void) const {
            return this->coordinates[1];
        }

        /**
         * Answer the z-coordinate of the point.
         *
         * @return The z-coordinate of the point.
         */
        inline const T& GetZ(void) const {
            return this->coordinates[2];
        }

        /**
         * Set the coordinates ot the point.
         *
         * @param x The x-coordinate of the point.
         * @param y The y-coordinate of the point.
         * @param z The z-coordinate of the point.
         */
        inline void Set(const T& x, const T& y, const T& z) {
            this->coordinates[0] = x;
            this->coordinates[1] = y;
            this->coordinates[2] = z;
        }

        /**
         * Set the x-coordinate of the point.
         *
         * @param x The new x-coordinate.
         */
        inline void SetX(const T& x) {
            this->coordinates[0] = x;
        }

        /**
         * Set the y-coordinate of the point.
         *
         * @param y The new y-coordinate.
         */
        inline void SetY(const T& y) {
            this->coordinates[1] = y;
        }

        /**
         * Set the z-coordinate of the point.
         *
         * @param z The new z-coordinate.
         */
        inline void SetZ(const T& z) {
            this->coordinates[2] = z;
        }

        /**
         * Answer the x-coordinate of the point.
         *
         * @return The x-coordinate of the point.
         */
        inline const T& X(void) const {
            return this->coordinates[0];
        }

        /**
         * Answer the y-coordinate of the point.
         *
         * @return The y-coordinate of the point.
         */
        inline const T& Y(void) const {
            return this->coordinates[1];
        }

        /**
         * Answer the z-coordinate of the point.
         *
         * @return The z-coordinate of the point.
         */
        inline const T& Z(void) const {
            return this->coordinates[2];
        }

        /** Behaves like primary class template. */
        inline AbstractPoint& operator =(const AbstractPoint& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /** Behaves like primary class template. */
        template<class Tp, unsigned int Dp, class Sp>
        inline AbstractPoint& operator =(
                const AbstractPoint<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** Typedef for our super class. */
        typedef AbstractPointImpl<T, 3, S, vislib::math::AbstractPoint> Super;

        /**
         * Disallow instances of this class. 
         */
        inline AbstractPoint(void) : Super() {}

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1, 
            template<class Tf2, unsigned int Df2, class Sf2> class Cf> 
            friend class AbstractPointImpl;
    };


    /*
     * vislib::math::AbstractPoint<T, 3, S>::~AbstractPoint
     */
    template<class T, class S> AbstractPoint<T, 3, S>::~AbstractPoint(void) {
    }


    /**
     * Partial template specialisation for four-dimensional points. This 
     * implementation provides convenience access methods to the four 
     * coordinates.
     */
    template<class T, class S> class AbstractPoint<T, 4, S> 
            : public AbstractPointImpl<T, 4, S, AbstractPoint> {

    public:

        /** Behaves like primary class template. */
        ~AbstractPoint(void);

        /**
         * Answer the x-coordinate of the point.
         *
         * @return The x-coordinate of the point.
         */
        inline const T& GetX(void) const {
            return this->coordinates[0];
        }

        /**
         * Answer the y-coordinate of the point.
         *
         * @return The y-coordinate of the point.
         */
        inline const T& GetY(void) const {
            return this->coordinates[1];
        }

        /**
         * Answer the z-coordinate of the point.
         *
         * @return The z-coordinate of the point.
         */
        inline const T& GetZ(void) const {
            return this->coordinates[2];
        }

        /**
         * Answer the w-coordinate of the point.
         *
         * @return The w-coordinate of the point.
         */
        inline const T& GetW(void) const {
            return this->coordinates[3];
        }

        /**
         * Set the coordinates ot the point.
         *
         * @param x The x-coordinate of the point.
         * @param y The y-coordinate of the point.
         * @param z The z-coordinate of the point.
         * @param w The w-coordinate of the point.
         */
        inline void Set(const T& x, const T& y, const T& z, const T& w) {
            this->coordinates[0] = x;
            this->coordinates[1] = y;
            this->coordinates[2] = z;
            this->coordinates[3] = w;
        }

        /**
         * Set the x-coordinate of the point.
         *
         * @param x The new x-coordinate.
         */
        inline void SetX(const T& x) {
            this->coordinates[0] = x;
        }

        /**
         * Set the y-coordinate of the point.
         *
         * @param y The new y-coordinate.
         */
        inline void SetY(const T& y) {
            this->coordinates[1] = y;
        }

        /**
         * Set the z-coordinate of the point.
         *
         * @param z The new z-coordinate.
         */
        inline void SetZ(const T& z) {
            this->coordinates[2] = z;
        }

        /**
         * Set the w-coordinate of the point.
         *
         * @param w The new w-coordinate.
         */
        inline void SetW(const T& w) {
            this->coordinates[3] = w;
        }

        /**
         * Answer the x-coordinate of the point.
         *
         * @return The x-coordinate of the point.
         */
        inline const T& X(void) const {
            return this->coordinates[0];
        }

        /**
         * Answer the y-coordinate of the point.
         *
         * @return The y-coordinate of the point.
         */
        inline const T& Y(void) const {
            return this->coordinates[1];
        }

        /**
         * Answer the z-coordinate of the point.
         *
         * @return The z-coordinate of the point.
         */
        inline const T& Z(void) const {
            return this->coordinates[2];
        }

        /**
         * Answer the w-coordinate of the point.
         *
         * @return The w-coordinate of the point.
         */
        inline const T& W(void) const {
            return this->coordinates[3];
        }

        /** Behaves like primary class template. */
        inline AbstractPoint& operator =(const AbstractPoint& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /** Behaves like primary class template. */
        template<class Tp, unsigned int Dp, class Sp>
        inline AbstractPoint& operator =(
                const AbstractPoint<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** Typedef for our super class. */
        typedef AbstractPointImpl<T, 4, S, vislib::math::AbstractPoint> Super;

        /**
         * Disallow instances of this class. 
         */
        inline AbstractPoint(void) : Super() {}

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1, 
            template<class Tf2, unsigned int Df2, class Sf2> class Cf> 
            friend class AbstractPointImpl;
    };


    /*
     * vislib::math::AbstractPoint<T, 4, S>::~AbstractPoint
     */
    template<class T, class S> AbstractPoint<T, 4, S>::~AbstractPoint(void) {
        // intentionally empty
    }

} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTPOINT_H_INCLUDED */
