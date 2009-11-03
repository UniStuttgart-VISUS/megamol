/*
 * AbstractVector.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTVECTOR_H_INCLUDED
#define VISLIB_ABSTRACTVECTOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractVectorImpl.h"


namespace vislib {
namespace math {


    /**
     *
     */
    template<class T, unsigned int D, class S> class AbstractVector 
            : public AbstractVectorImpl<T, D, S, AbstractVector> {

    public:

        /** Dtor. */
        ~AbstractVector(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline AbstractVector& operator =(const AbstractVector& rhs) {
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
        inline AbstractVector& operator =(
                const AbstractVector<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** Typedef for our super class. */
        typedef AbstractVectorImpl<T, D, S, vislib::math::AbstractVector> Super;

        /**
         * Disallow instances of this class. 
         */
        inline AbstractVector(void) : Super() {}

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1, 
            template<class Tf2, unsigned int Df2, class Sf2> class Cf> 
            friend class AbstractVectorImpl;
    };


    /*
     * vislib::math::AbstractVector<T, D, S>::~AbstractVector
     */
    template<class T, unsigned int D, class S>
    AbstractVector<T, D, S>::~AbstractVector(void) {
    }


    /**
     * Partial template specialisation for two-dimensional vectors. This
     * specialisation provides named accessors to the vector's components.
     */
    template<class T, class S> class AbstractVector<T, 2, S> 
            : public AbstractVectorImpl<T, 2, S, AbstractVector> {

    public:

        /** Behaves like primary class template. */
        ~AbstractVector(void);

        /**
         * Answer the x-component of the vector.
         *
         * @return The x-component of the vector.
         */
        inline const T& GetX(void) const {
            return this->components[0];
        }

        /**
         * Answer the y-component of the vector.
         *
         * @return The y-component of the vector.
         */
        inline const T& GetY(void) const {
            return this->components[1];
        }

        /**
         * Set the three components of the vector.
         *
         * @param x The new x-component.
         * @param y The new y-component.
         */
        inline void Set(const T& x, const T& y) {
            this->components[0] = x;
            this->components[1] = y;
        }

        /**
         * Set the x-component of the vector.
         *
         * @param x The new x-component.
         */
        inline void SetX(const T& x) {
            this->components[0] = x;
        }

        /**
         * Set the y-component of the vector.
         *
         * @param y The new y-component.
         */
        inline void SetY(const T& y) {
            this->components[1] = y;
        }

        /**
         * Answer the x-component of the vector.
         *
         * @return The x-component of the vector.
         */
        inline const T& X(void) const {
            return this->components[0];
        }

        /**
         * Answer the y-component of the vector.
         *
         * @return The y-component of the vector.
         */
        inline const T& Y(void) const {
            return this->components[1];
        }

        /** Behaves like primary class template. */
        inline AbstractVector& operator =(const AbstractVector& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /** Behaves like primary class template. */
        template<class Tp, unsigned int Dp, class Sp>
        inline AbstractVector& operator =(
                const AbstractVector<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** Typedef for our super class. */
        typedef AbstractVectorImpl<T, 2, S, vislib::math::AbstractVector> Super;

        /**
         * Disallow instances of this class. 
         */
        inline AbstractVector(void) : Super() {}

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1, 
            template<class Tf2, unsigned int Df2, class Sf2> class Cf> 
            friend class AbstractVectorImpl;
    };


    /*
     * vislib::math::AbstractVector<T, 2, S>::~AbstractVector
     */
    template<class T, class S>
    AbstractVector<T, 2, S>::~AbstractVector(void) {
    }


    /**
     * Partial template specialisation for three-dimensional vectors. This
     * specialisation provides named accessors to the vector's components
     * and some special operations.
     */
    template<class T, class S> class AbstractVector<T, 3, S> 
            : public AbstractVectorImpl<T, 3, S, AbstractVector> {

    public:

        /** Dtor. */
        ~AbstractVector(void);

       /**
         * Answer the cross product of this vector and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The cross product of this vector and 'rhs'.
         */
        template<class Tp, class Sp> AbstractVector<T, 3, T[3]> Cross(
            const AbstractVector<Tp, 3, Sp>& rhs) const;

        ///**
        // * Calculate the cross product of this vector and 'rhs' and assign it
        // * to this vector.
        // *
        // * @param rhs The right hand side operand.
        // *
        // * @return *this.
        // */
        //template<class Tp, class Sp>
        //inline AbstractVector3D& CrossAssign(
        //        const AbstractVector3D<Tp, Sp>& rhs) {
        //    return (*this = this->Cross(rhs));
        //}

        /**
         * Answer the x-component of the vector.
         *
         * @return The x-component of the vector.
         */
        inline const T& GetX(void) const {
            return this->components[0];
        }

        /**
         * Answer the y-component of the vector.
         *
         * @return The y-component of the vector.
         */
        inline const T& GetY(void) const {
            return this->components[1];
        }

        /**
         * Answer the z-component of the vector.
         *
         * @return The z-component of the vector.
         */
        inline const T& GetZ(void) const {
            return this->components[2];
        }

        /**
         * Set the three components of the vector.
         *
         * @param x The new x-component.
         * @param y The new y-component.
         * @param z The new z-component.
         */
        inline void Set(const T& x, const T& y, const T& z) {
            this->components[0] = x;
            this->components[1] = y;
            this->components[2] = z;
        }

        /**
         * Set the x-component of the vector.
         *
         * @param x The new x-component.
         */
        inline void SetX(const T& x) {
            this->components[0] = x;
        }

        /**
         * Set the y-component of the vector.
         *
         * @param y The new y-component.
         */
        inline void SetY(const T& y) {
            this->components[1] = y;
        }

        /**
         * Set the z-component of the vector.
         *
         * @param z The new z-component.
         */
        inline void SetZ(const T& z) {
            this->components[2] = z;
        }

        /**
         * Answer the x-component of the vector.
         *
         * @return The x-component of the vector.
         */
        inline const T& X(void) const {
            return this->components[0];
        }

        /**
         * Answer the y-component of the vector.
         *
         * @return The y-component of the vector.
         */
        inline const T& Y(void) const {
            return this->components[1];
        }

        /**
         * Answer the z-component of the vector.
         *
         * @return The z-component of the vector.
         */
        inline const T& Z(void) const {
            return this->components[2];
        }

        /** Behaves like primary class template. */
        inline AbstractVector& operator =(const AbstractVector& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /** Behaves like primary class template. */
        template<class Tp, unsigned int Dp, class Sp>
        inline AbstractVector& operator =(
                const AbstractVector<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }


    protected:

        /** Typedef for our super class. */
        typedef AbstractVectorImpl<T, 3, S, vislib::math::AbstractVector> Super;

        /**
         * Disallow instances of this class. 
         */
        inline AbstractVector(void) : Super() {}

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1, 
            template<class Tf2, unsigned int Df2, class Sf2> class Cf> 
            friend class AbstractVectorImpl;
        template<class Tf1, unsigned int Df1, class Sf1> 
            friend class AbstractVector;
    };


    /*
     * vislib::math::AbstractVector<T, 3, S>::~AbstractVector
     */
    template<class T, class S>
    AbstractVector<T, 3, S>::~AbstractVector(void) {
    }


    /*
     * vislib::math::AbstractVector<T, S>::Cross
     */
    template<class T, class S>
    template<class Tp, class Sp> 
    AbstractVector<T, 3, T[3]> AbstractVector<T, 3, S>::Cross(
            const AbstractVector<Tp, 3, Sp>& rhs) const {
        AbstractVector<T, 3, T[3]> retval;
        retval.Set(
            this->components[1] * static_cast<T>(rhs[2])
            - this->components[2] * static_cast<T>(rhs[1]),
            this->components[2] * static_cast<T>(rhs[0])
            - this->components[0] * static_cast<T>(rhs[2]),
            this->components[0] * static_cast<T>(rhs[1])
            - this->components[1] * static_cast<T>(rhs[0]));
        return retval;
    }


    /**
     * Partial template specialisation for four-dimensional vectors. This
     * specialisation provides named accessors to the vector's components.
     */
    template<class T, class S> class AbstractVector<T, 4, S> 
            : public AbstractVectorImpl<T, 4, S, AbstractVector> {

    public:

        /** Dtor. */
        ~AbstractVector(void);

        /**
         * Answer the x-component of the vector.
         *
         * @return The x-component of the vector.
         */
        inline const T& GetX(void) const {
            return this->components[0];
        }

        /**
         * Answer the y-component of the vector.
         *
         * @return The y-component of the vector.
         */
        inline const T& GetY(void) const {
            return this->components[1];
        }

        /**
         * Answer the z-component of the vector.
         *
         * @return The z-component of the vector.
         */
        inline const T& GetZ(void) const {
            return this->components[2];
        }

        /**
         * Answer the w-component of the vector.
         *
         * @return The w-component of the vector.
         */
        inline const T& GetW(void) const {
            return this->components[3];
        }

        /**
         * Set the three components of the vector.
         *
         * @param x The new x-component.
         * @param y The new y-component.
         * @param z The new z-component.
         * @param w The new w-component.
         */
        inline void Set(const T& x, const T& y, const T& z, const T& w) {
            this->components[0] = x;
            this->components[1] = y;
            this->components[2] = z;
            this->components[3] = w;
        }

        /**
         * Set the x-component of the vector.
         *
         * @param x The new x-component.
         */
        inline void SetX(const T& x) {
            this->components[0] = x;
        }

        /**
         * Set the y-component of the vector.
         *
         * @param y The new y-component.
         */
        inline void SetY(const T& y) {
            this->components[1] = y;
        }

        /**
         * Set the z-component of the vector.
         *
         * @param z The new z-component.
         */
        inline void SetZ(const T& z) {
            this->components[2] = z;
        }

        /**
         * Set the w-component of the vector.
         *
         * @param w The new w-component.
         */
        inline void SetW(const T& w) {
            this->components[3] = w;
        }

        /**
         * Answer the x-component of the vector.
         *
         * @return The x-component of the vector.
         */
        inline const T& X(void) const {
            return this->components[0];
        }

        /**
         * Answer the y-component of the vector.
         *
         * @return The y-component of the vector.
         */
        inline const T& Y(void) const {
            return this->components[1];
        }

        /**
         * Answer the z-component of the vector.
         *
         * @return The z-component of the vector.
         */
        inline const T& Z(void) const {
            return this->components[2];
        }

        /**
         * Answer the w-component of the vector.
         *
         * @return The w-component of the vector.
         */
        inline const T& W(void) const {
            return this->components[3];
        }

        /** Behaves like primary class template. */
        inline AbstractVector& operator =(const AbstractVector& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /** Behaves like primary class template. */
        template<class Tp, unsigned int Dp, class Sp>
        inline AbstractVector& operator =(
                const AbstractVector<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }


    protected:

        /** Typedef for our super class. */
        typedef AbstractVectorImpl<T, 4, S, vislib::math::AbstractVector> Super;

        /**
         * Disallow instances of this class. 
         */
        inline AbstractVector(void) : Super() {}

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1, 
            template<class Tf2, unsigned int Df2, class Sf2> class Cf> 
            friend class AbstractVectorImpl;
        template<class Tf1, unsigned int Df1, class Sf1> 
            friend class AbstractVector;
    };


    /*
     * vislib::math::AbstractVector<T, 4, S>::~AbstractVector
     */
    template<class T, class S>
    AbstractVector<T, 4, S>::~AbstractVector(void) {
        // intentionally empty
    }


    /**
     * Scalar multiplication from left.
     *
     * @param lhs The left hand side operand, the scalar.
     * @param rhs The right hand side operand, the vector.
     *
     * @return The result of the scalar multiplication.
     */
    template<class T, unsigned int D, class S> 
    AbstractVector<T, D, T[D]> operator *(const T lhs, 
            const AbstractVector<T, D, S>& rhs) {
        return rhs * lhs;
    }


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTVECTOR_H_INCLUDED */
