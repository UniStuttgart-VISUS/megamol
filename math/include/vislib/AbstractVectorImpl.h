/*
 * AbstractVectorImplImpl.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTVECTORIMPL_H_INCLUDED
#define VISLIB_ABSTRACTVECTORIMPL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <limits>

#include "vislib/assert.h"
#include "vislib/mathfunctions.h"
#include "vislib/memutils.h"
#include "vislib/OutOfRangeException.h"


namespace vislib {
namespace math {

    /**
     * This template implements the major behaviour of vectors. Its only 
     * intended use is being the super class of the AbstractVector template
     * and its partial template specialisation. It should never be used in
     * any other context. It is especially not intended to be an interface!
     *
     * This template has several template parameters that allow instantiations
     * for different scalar types, different dimensions and different storage
     * classes. If the storage class S is T[D], the implementation uses its
     * own memory for storing its data. If S is T *, an instance that uses
     * foreign memory is created - we call this "shallow vectors". This 
     * instantiation is used to make a user defined memory block behave like a
     * vector. The template parameter C must be the direct subclass. It allows
     * this class to create appropriate return values for the arithmetic
     * operations. We use this approach to share this implementation between
     * the primary class template AbstractVector and its partial template
     * specialisations.
     *
     * T: The type used for scalars and vector components.
     * D: The dimensions of the vector, which must be an integer greater or
     *    equal 1.
     * S: The "storage class". This can be either T[D] for a "deep vector" or
     *    T * for a "shallow vector". Other instantiations are inherently 
     *    dangerous and should never be used.
     * C: The direct subclass, i. e. AbstractVector. This allows the 
     *    implementation to create the required return values.
     */
    template<class T, unsigned int D, class S, 
            template<class T, unsigned int D, class S> class C> 
            class AbstractVectorImpl {

    public:

        /** Dtor. */
        ~AbstractVectorImpl(void);

        /**
         * Answer the angle between this vector and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The angle between this vector and 'rhs'.
         */
        AngleRad Angle(const C<T, D, S>& rhs) const;

        /**
         * Answer the dot product of this vector and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The dot product of this vector and 'rhs'.
         */
        T Dot(const C<T, D, S>& rhs) const;

        /**
         * Answer whether the vector is normalised.
         *
         * @return true, if the vector is normalised, false otherwise.
         */
        inline bool IsNormalised(void) const {
            return IsEqual<T>(this->Length(), static_cast<T>(1));
        }

        /**
         * Answer whether the vector is a null vector.
         *
         * @return true, if the vector is a null vector, false otherwise.
         */
        bool IsNull(void) const;

        /**
         * Answer whether the rhs vector and this are parallel.
         *
         * If both vectors are null vectors, they are not considered to be 
         * parallel and the the return value will be false.
         *
         * @return true, if both vectors are parallel, false otherwise.
         */
        template<class Tp, class Sp>
        bool IsParallel(const C<Tp, D, Sp>& rhs) const;

        /**
         * Answer the length of the vector.
         *
         * @return The length of the vector.
         */
        T Length(void) const;

        /**
         * Answer the maximum norm of the vector.
         *
         * @return The maximum norm of the vector.
         */
        T MaxNorm(void) const;

        /**
         * Answer the euclidean norm (length) of the vector.
         *
         * @return The length of the vector.
         */
        inline T Norm(void) const {
            return this->Length();
        }

        /**
         * Normalise the vector.
         *
         * @return The OLD length of the vector.
         */
        T Normalise(void);

        /**
         * Directly access the internal pointer holding the vector components.
         * The object remains owner of the memory returned.
         *
         * @return The vector components in an array.
         */
        inline T *PeekComponents(void) {
            return this->components;
        }

        /**
         * Directly access the internal pointer holding the vector components. 
         * The object remains owner of the memory returned.
         *
         * @return The vector components in an array.
         */
        inline const T *PeekComponents(void) const {
            return this->components;
        }

        /**
         * Make this vector a null vector.
         */
        void SetNull(void);

        /**
         * Make this vector having a length of 'newLength'.
         *
         * Note that a null vector cannot be scaled and will remain unchanged.
         *
         * @param newLength The new length of the vector.
         *
         * @return The old length of the vector.
         */
        T ScaleToLength(const T newLength);

        /**
         * Answer the square of the length of the vector.
         *
         * @return The square of the length of the vector.
         */
        T SquareLength(void) const;

        /**
         * Answer the taxicab or Manhattan norm.
         *
         * @return The taxicab norm of the vector.
         */
        T TaxicabNorm(void) const;

        /**
         * Assignment.
         *
         * This operation does <b>not</b> create aliases.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        AbstractVectorImpl<T, D, S, C>& operator =(const C<T, D, S>& rhs);

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
         * Subclasses must ensure that sufficient memory for the 'components' 
         * member has been allocated before calling this operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        template<class Tp, unsigned int Dp, class Sp>
        AbstractVectorImpl<T, D, S, C>& operator =(const C<Tp, Dp, Sp>& rhs);

        /**
         * Test for equality. This operation uses the E function which the 
         * template has been instantiated for.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this vector are equal, false otherwise.
         */
        bool operator ==(const C<T, D, S>& rhs) const;

        /**
         * Test for equality of arbitrary vector types. This operation uses the
         * IsEqual function of the left hand side operand. Note that vectors 
         * with different dimensions are never equal.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this vector are equal, false otherwise.
         */
        template<class Tp, unsigned int Dp, class Sp>
        bool operator ==(const C<Tp, Dp, Sp>& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this vector are not equal, false otherwise.
         */
        inline bool operator !=(const C<T, D, S>& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Test for inequality of arbitrary vectors. See operator == for further
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
         * Negate the vector.
         *
         * @return The negated version of this vector.
         */
        C<T, D, S> operator -(void) const;

        /**
         * Answer the sum of this vector and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The sum of this and 'rhs'.
         */
        template<class Tp, class Sp>
        C<T, D, T[D]> operator +(const C<Tp, D, Sp>& rhs) const;

        /**
         * Add 'rhs' to this vector and answer the sum.
         *
         * @param rhs The right hand side operand.
         *
         * @return The sum of this and 'rhs'.
         */
        template<class Tp, class Sp>
        AbstractVectorImpl<T, D, S, C>& operator +=(const C<Tp, D, Sp>& rhs);

        /**
         * Answer the difference between this vector and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The difference between this and 'rhs'.
         */
        template<class Tp, class Sp>
        C<T, D, T[D]> operator -(const C<Tp, D, Sp>& rhs) const;

        /**
         * Subtract 'rhs' from this vector and answer the difference.
         *
         * @param rhs The right hand side operand.
         *
         * @return The difference between this and 'rhs'.
         */
        template<class Tp, class Sp>
        AbstractVectorImpl<T, D, S, C>& operator -=(const C<Tp, D, Sp>& rhs);

        /**
         * Scalar multiplication.
         *
         * @param rhs The right hand side operand.
         *
         * @return The result of the scalar multiplication.
         */
        C<T, D, T[D]> operator *(const T rhs) const;

        /**
         * Scalar multiplication assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return The result of the scalar multiplication.
         */
        AbstractVectorImpl<T, D, S, C>& operator *=(const T rhs);

        /**
         * Scalar division operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return The result of the scalar division.
         */
        C<T, D, T[D]> operator /(const T rhs) const;

        /**
         * Scalar division assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return The result of the scalar division.
         */
        AbstractVectorImpl<T, D, S, C>& operator /=(const T rhs);

        /**
         * Performs a component-wise multiplication.
         *
         * @param rhs The right hand side operand.
         *
         * @return The product of this and rhs.
         */
        template<class Tp, class Sp>
        C<T, D, T[D]> operator *(const C<Tp, D, Sp>& rhs) const;

        /**
         * Multiplies 'rhs' component-wise with this vector and returns
         * the result.
         *
         * @param rhs The right hand side operand.
         *
         * @return The product of this and rhs
         */
        template<class Tp, class Sp>
        AbstractVectorImpl<T, D, S, C>& operator *=(const C<Tp, D, Sp>& rhs);

        /**
         * Component access.
         *
         * @param i The index of the requested component, which must be within
         *          [0, D - 1].
         *
         * @return A reference on the 'i'th component.
         *
         * @throws OutOfRangeException, If 'i' is not within [0, D[.
         */
        T& operator [](const int i);

        /**
         * Component access.
         *
         * @param i The index of the requested component, which must be within
         *          [0, D - 1].
         *
         * @return A reference on the 'i'th component.
         *
         * @throws OutOfRangeException, If 'i' is not within [0, D[.
         */
        const T& operator [](const int i) const;

    protected:

        /**
         * Disallow instances of this class. This ctor does nothing!
         */
        inline AbstractVectorImpl(void) {};

        /** 
         * The vector components. This can be a T * pointer or a T[D] static
         * array.
         */
        S components;
    };


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::~AbstractVectorImpl
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    AbstractVectorImpl<T, D, S, C>::~AbstractVectorImpl(void) {
    }


    /*
     * AbstractVectorImpl<T, D, S, C>::Angle
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    AngleRad AbstractVectorImpl<T, D, S, C>::Angle(const C<T, D, S>& rhs) const {
        C<T, D, T[D]> v1;
        C<T, D, T[D]> v2;

        for (unsigned int d = 0; d < D; d++) {
            v1.components[d] = this->components[d];
            v2.components[d] = rhs.components[d];
        }

        v1.Normalise();
        v2.Normalise();

        return static_cast<AngleRad>(::acos(v1.Dot(v2)));
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::Dot
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    T AbstractVectorImpl<T, D, S, C>::Dot(const C<T, D, S>& rhs) const {
        T retval = static_cast<T>(0);

        for (unsigned int d = 0; d < D; d++) {
            retval += this->components[d] * rhs.components[d];
        }

        return retval;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::IsNull
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    bool AbstractVectorImpl<T, D, S, C>::IsNull(void) const {
        for (unsigned int d = 0; d < D; d++) {
            if (!IsEqual<T>(this->components[d], static_cast<T>(0))) {
                return false;
            }
        }
        /* No non-null value found. */

        return true;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::IsParallel
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    template<class Tp, class Sp>
    bool AbstractVectorImpl<T, D, S, C>::IsParallel(
            const C<Tp, D, Sp>& rhs) const {
        T factor = static_cast<T>(0); // this = factor * rhs
        bool inited = false; // if factor is initialized

        for (unsigned int d = 0; d < D; d++) {

            if (IsEqual<T>(this->components[d], static_cast<T>(0))) {
                // compare component of rhs in type of lhs
                if (!IsEqual<T>(rhs.components[d], static_cast<T>(0))) {
                    return false; // would yield to a factor of zero
                }
                // both zero, so go on.

            } else {
                // compare component of rhs in type of lhs
                if (IsEqual<T>(rhs.components[d], static_cast<T>(0))) {
                    return false; // would yield to a factor of infinity
                } else {
                    // both not zero, check if factor is const over all components
                    if (inited) {
                        if (!IsEqual<T>(factor, this->components[d] 
                                / static_cast<T>(rhs.components[d]))) {
                            return false;
                        }
                    } else {
                        factor = this->components[d] 
                            / static_cast<T>(rhs.components[d]);
                        inited = true;
                    }
                }
            }
        }

        return inited;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::Length
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    T AbstractVectorImpl<T, D, S, C>::Length(void) const {
        T retval = static_cast<T>(0);

        for (unsigned int d = 0; d < D; d++) {
            retval += Sqr(this->components[d]);
        }

        return Sqrt(retval);
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::MaxNorm
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    T AbstractVectorImpl<T, D, S, C>::MaxNorm(void) const {
#ifdef _MSC_VER
#pragma push_macro("min")
#undef min
#pragma push_macro("max")
#undef max
#endif /* _MSC_VER */
        T retval = std::numeric_limits<T>::is_integer 
            ? std::numeric_limits<T>::min() : -std::numeric_limits<T>::max();
#ifdef _MSC_VER
#pragma pop_macro("min")
#pragma pop_macro("max")
#endif /* _MSC_VER */

        for (unsigned int d = 0; d < D; d++) {
            if (Abs(this->components[d]) > retval) {
                retval = Abs(this->components[d]);
            }
        }

        return retval;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::Normalise
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    T AbstractVectorImpl<T, D, S, C>::Normalise(void) {
        T length = this->Length();

        if (length != static_cast<T>(0)) {
            for (unsigned int d = 0; d < D; d++) {
                this->components[d] /= length;
            }

        } else {
            for (unsigned int d = 0; d < D; d++) {
                this->components[d] = static_cast<T>(0);
            }
        }

        return length;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::SetNull
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    void AbstractVectorImpl<T, D, S, C>::SetNull(void) {
        for (unsigned int d = 0; d < D; d++) {
            this->components[d] = static_cast<T>(0);
        }
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::ScaleToLength
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    T AbstractVectorImpl<T, D, S, C>::ScaleToLength(const T newLength) {
        T retval = this->Length();

        if (retval != static_cast<T>(0)) {
            T scaleFactor = newLength / retval;

            for (unsigned int d = 0; d < D; d++) {
                this->components[d] *= scaleFactor;
            }
        }

        return retval;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::SquareLength
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    T AbstractVectorImpl<T, D, S, C>::SquareLength(void) const {
        T retval = static_cast<T>(0);

        for (unsigned int d = 0; d < D; d++) {
            retval += Sqr(this->components[d]);
        }

        return retval;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::TaxicabNorm
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    T AbstractVectorImpl<T, D, S, C>::TaxicabNorm(void) const {
        T retval = static_cast<T>(0);

        for (unsigned int d = 0; d < D; d++) {
            retval += this->components[d];
        }

        return retval;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator =
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    AbstractVectorImpl<T, D, S, C>& AbstractVectorImpl<T, D, S, C>::operator =(
            const C<T, D, S>& rhs) {

        if (this != &rhs) {
            ::memcpy(this->components, rhs.components, D * sizeof(T));
        }

        return *this;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator =
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    template<class Tp, unsigned int Dp, class Sp>
    AbstractVectorImpl<T, D, S, C>& AbstractVectorImpl<T, D, S, C>::operator =(
            const C<Tp, Dp, Sp>& rhs) {

        if (static_cast<void *>(this) != static_cast<const void *>(&rhs)) {
            for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
                this->components[d] = static_cast<T>(rhs[d]);
            }
            for (unsigned int d = Dp; d < D; d++) {
                this->components[d] = static_cast<T>(0);
            }            
        }

        return *this;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator ==
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    bool AbstractVectorImpl<T, D, S, C>::operator ==(
            const C<T, D, S>& rhs) const {

        for (unsigned int d = 0; d < D; d++) {
            if (!IsEqual<T>(this->components[d], rhs.components[d])) {
                return false;
            }
        }

        return true;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator ==
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    template<class Tp, unsigned int Dp, class Sp>
    bool AbstractVectorImpl<T, D, S, C>::operator ==(
            const C<Tp, Dp, Sp>& rhs) const {
        if (D != Dp) {
            return false;
        }

        for (unsigned int d = 0; d < D; d++) {
            if (!IsEqual<T>(this->components[d], rhs[d])) {
                return false;
            }
        }

        return true;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator -
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    C<T, D, S> AbstractVectorImpl<T, D, S, C>::operator -(void) const {
        C<T, D, S> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] = -this->components[d];
        }

        return retval;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator +
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    template<class Tp, class Sp>
    C<T, D, T[D]> AbstractVectorImpl<T, D, S, C>::operator +(
            const C<Tp, D, Sp>& rhs) const {
        C<T, D, T[D]> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] = this->components[d] + rhs.components[d];
        }

        return retval;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator +=
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    template<class Tp, class Sp>
    AbstractVectorImpl<T, D, S, C>& AbstractVectorImpl<T, D, S, C>::operator +=(
            const C<Tp, D, Sp>& rhs) {

        for (unsigned int d = 0; d < D; d++) {
            this->components[d] += rhs.components[d];
        }

        return *this;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator -
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    template<class Tp, class Sp>
    C<T, D, T[D]> AbstractVectorImpl<T, D, S, C>::operator -(
            const C<Tp, D, Sp>& rhs) const {
        C<T, D, T[D]> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] = this->components[d] - rhs.components[d];
        }

        return retval;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator -=
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    template<class Tp, class Sp>
    AbstractVectorImpl<T, D, S, C>& AbstractVectorImpl<T, D, S, C>::operator -=(
            const C<Tp, D, Sp>& rhs) {

        for (unsigned int d = 0; d < D; d++) {
            this->components[d] -= rhs.components[d];
        }
       
        return *this;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator *
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    C<T, D, T[D]> AbstractVectorImpl<T, D, S, C>::operator *(
            const T rhs) const {
        C<T, D, T[D]> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] = this->components[d] * rhs;
        }

        return retval;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator *=
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    AbstractVectorImpl<T, D, S, C>& AbstractVectorImpl<T, D, S, C>::operator *=(
            const T rhs) {

        for (unsigned int d = 0; d < D; d++) {
            this->components[d] *= rhs;
        }

        return *this;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator /
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    C<T, D, T[D]> AbstractVectorImpl<T, D, S, C>::operator /(
            const T rhs) const {
        C<T, D, T[D]> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] = this->components[d] / rhs;
        }

        return retval;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator /=
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    AbstractVectorImpl<T, D, S, C>& AbstractVectorImpl<T, D, S, C>::operator /=(
            const T rhs) {

        for (unsigned int d = 0; d < D; d++) {
            this->components[d] /= rhs;
        }

        return *this;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator *
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    template<class Tp, class Sp>
    C<T, D, T[D]> AbstractVectorImpl<T, D, S, C>::operator *(
            const C<Tp, D, Sp>& rhs) const {
        C<T, D, T[D]> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] = this->components[d] * rhs.components[d];
        }

        return retval;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator *=
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    template<class Tp, class Sp>
    AbstractVectorImpl<T, D, S, C>& AbstractVectorImpl<T, D, S, C>::operator *=(
            const C<Tp, D, Sp>& rhs) {

        for (unsigned int d = 0; d < D; d++) {
            this->components[d] *= rhs.components[d];
        }

        return *this;
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator []
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    T& AbstractVectorImpl<T, D, S, C>::operator [](const int i) {
        if ((i >= 0) && (i < static_cast<int>(D))) {
            return this->components[i];
        } else {
            throw OutOfRangeException(i, 0, D - 1, __FILE__, __LINE__);
        }
    }


    /*
     * vislib::math::AbstractVectorImpl<T, D, S, C>::operator []
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C>
    const T& AbstractVectorImpl<T, D, S, C>::operator [](const int i) const {
        if ((i >= 0) && (i < static_cast<int>(D))) {
            return this->components[i];
        } else {
            throw OutOfRangeException(i, 0, D - 1, __FILE__, __LINE__);
        }
    }
    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTVECTORIMPL_H_INCLUDED */
