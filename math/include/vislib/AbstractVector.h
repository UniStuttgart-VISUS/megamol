/*
 * AbstractVector.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTVECTOR_H_INCLUDED
#define VISLIB_ABSTRACTVECTOR_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include <limits>

#include "vislib/assert.h"
#include "vislib/mathfunctions.h"
#include "vislib/memutils.h"
#include "vislib/OutOfRangeException.h"


namespace vislib {
namespace math {

    /**
     * This is the abstract super class for all vectors used in the vislib. It
     * serves primarily two purposes: Firstly, it prevents code duplication 
     * between vector classes. Secondly, it permits polymorphism between vector
     * classes.
     * 
     * The AbstractVector class has the following template parameters:
     * T is the type of scalars used to build the vector of.
     * D is the dimension of the vector.
     * S is the component storage. For vectors that do not have their own storge
     *   (we calle these "shallow vectors"), this must be a T * pointer. For
     *   normal (deep) vectors, this must be a static array T[D].
     *
     * This class cannot be instantiated as it would be unsafe lettings the user
     * decide on the storage class.
     *
     * Note, that there is no virtual method in order to prevent overhead for
     * dynamic dispatching. This is no problem as long as no method must be
     * overridden and as long as the dtor does nothing. This is the case for 
     * this class and its derived classes. 
     */
    template<class T, unsigned int D, class S> class AbstractVector {

    public:

        /** Dtor. */
        ~AbstractVector(void);

        /**
         * Answer the dot product of this vector and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The dot product of this vector and 'rhs'.
         */
        T Dot(const AbstractVector& rhs) const;

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
        bool IsParallel(const AbstractVector<Tp, D, Sp>& rhs) const;

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
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        AbstractVector& operator =(const AbstractVector& rhs);

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
        AbstractVector& operator =(const AbstractVector<Tp, Dp, Sp>& rhs);

        /**
         * Test for equality. This operation uses the E function which the 
         * template has been instantiated for.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this vector are equal, false otherwise.
         */
        bool operator ==(const AbstractVector& rhs) const;

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
        bool operator ==(const AbstractVector<Tp, Dp, Sp>& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this vector are not equal, false otherwise.
         */
        inline bool operator !=(const AbstractVector& rhs) const {
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
        inline bool operator !=(const AbstractVector<Tp, Dp, Sp>& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Negate the vector.
         *
         * @return The negated version of this vector.
         */
        AbstractVector operator -(void) const;

        /**
         * Answer the sum of this vector and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The sum of this and 'rhs'.
         */
        AbstractVector<T, D, T[D]> operator +(const AbstractVector& rhs) const;

        /**
         * Add 'rhs' to this vector and answer the sum.
         *
         * @param rhs The right hand side operand.
         *
         * @return The sum of this and 'rhs'.
         */
        AbstractVector& operator +=(const AbstractVector& rhs);

        /**
         * Answer the difference between this vector and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The difference between this and 'rhs'.
         */
        AbstractVector<T, D, T[D]> operator -(const AbstractVector& rhs) const;

        /**
         * Subtract 'rhs' from this vector and answer the difference.
         *
         * @param rhs The right hand side operand.
         *
         * @return The difference between this and 'rhs'.
         */
        AbstractVector& operator -=(const AbstractVector& rhs);

        /**
         * Scalar multiplication.
         *
         * @param rhs The right hand side operand.
         *
         * @return The result of the scalar multiplication.
         */
        AbstractVector<T, D, T[D]> operator *(const T rhs) const;

        /**
         * Scalar multiplication assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return The result of the scalar multiplication.
         */
        AbstractVector& operator *=(const T rhs);

        /**
         * Scalar division operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return The result of the scalar division.
         */
        AbstractVector<T, D, T[D]> operator /(const T rhs) const;

        /**
         * Scalar division assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return The result of the scalar division.
         */
        AbstractVector& operator /=(const T rhs);

        /**
         * Performs a component-wise multiplication.
         *
         * @param rhs The right hand side operand.
         *
         * @return The product of this and rhs.
         */
        AbstractVector<T, D, T[D]> operator *(const AbstractVector& rhs) const;

        /**
         * Multiplies 'rhs' component-wise with this vector and returns
         * the result.
         *
         * @param rhs The right hand side operand.
         *
         * @return The product of this and rhs
         */
        AbstractVector& operator *=(const AbstractVector& rhs);

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
        inline AbstractVector(void) {};

        /** 
         * The vector components. This can be a T * pointer or a T[D] static
         * array.
         */
        S components;
    };

    /*
     * AbstractVector<T, D, S>::~AbstractVector
     */
    template<class T, unsigned int D, class S>
    AbstractVector<T, D, S>::~AbstractVector(void) {
    }


    /*
     * AbstractVector<T, D, S>::Dot
     */
    template<class T, unsigned int D, class S>
    T AbstractVector<T, D, S>::Dot(const AbstractVector& rhs) const {
        T retval = static_cast<T>(0);

        for (unsigned int d = 0; d < D; d++) {
            retval += this->components[d] * rhs.components[d];
        }

        return retval;
    }


    /*
     * AbstractVector<T, D, S>::IsNull
     */
    template<class T, unsigned int D, class S>
    bool AbstractVector<T, D, S>::IsNull(void) const {
        for (unsigned int d = 0; d < D; d++) {
            if (!IsEqual<T>(this->components[d], static_cast<T>(0))) {
                return false;
            }
        }
        /* No non-null value found. */

        return true;
    }


    /*
     * template<T, D, S>::IsParallel
     */
    template<class T, unsigned int D, class S>
    template<class Tp, class Sp>
    bool AbstractVector<T, D, S>::IsParallel(const AbstractVector<Tp, D, Sp>& rhs) const {
        T factor; // this = factor * rhs
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
                        if (!IsEqual<T>(factor, this->components[d] / static_cast<T>(rhs.components[d]))) {
                            return false;
                        }
                    } else {
                        factor = this->components[d] / static_cast<T>(rhs.components[d]);
                    }

                }
            }
        }

        return inited;
    }


    /*
     * AbstractVector<T, D, S>::Length
     */
    template<class T, unsigned int D, class S>
    T AbstractVector<T, D, S>::Length(void) const {
        T retval = static_cast<T>(0);

        for (unsigned int d = 0; d < D; d++) {
            retval += Sqr(this->components[d]);
        }

        return Sqrt(retval);
    }


    /*
     * AbstractVector<T, D, S>::MaxNorm
     */
    template<class T, unsigned int D, class S>
    T AbstractVector<T, D, S>::MaxNorm(void) const {
#if defined(_MSC_VER) && defined(min)
#define POP_MIN_CROWBAR 1
#pragma push_macro("min")
#undef min
#endif /* defined(_MSC_VER) && defined(min) */
#if defined(_MSC_VER) && defined(max)
#define POP_MAX_CROWBAR 1
#pragma push_macro("max")
#undef max
#endif /* defined(_MSC_VER) && defined(max) */
        T retval = std::numeric_limits<T>::is_integer 
            ? std::numeric_limits<T>::min() : -std::numeric_limits<T>::max();
#ifdef POP_MIN_CROWBAR
#pragma pop_macro("min")
#undef POP_MIN_CROWBAR
#endif /* POP_MIN_CROWBAR */
#ifdef POP_MAX_CROWBAR
#pragma pop_macro("max")
#undef POP_MAX_CROWBAR
#endif /* POP_MAX_CROWBAR */

        for (unsigned int d = 0; d < D; d++) {
            if (this->components[d] > retval) {
                retval = this->components[d];
            }
        }

        return retval;
    }


    /*
     * AbstractVector<T, D, S>::Normalise
     */
    template<class T, unsigned int D, class S>
    T AbstractVector<T, D, S>::Normalise(void) {
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
     * AbstractVector<T, D, S>::operator =
     */
    template<class T, unsigned int D, class S>
    AbstractVector<T, D, S>& AbstractVector<T, D, S>::operator =(
            const AbstractVector& rhs) {

        if (this != &rhs) {
            ::memcpy(this->components, rhs.components, D * sizeof(T));
        }

        return *this;
    }


    /*
     * AbstractVector<T, D, S>::operator =
     */
    template<class T, unsigned int D, class S>
    template<class Tp, unsigned int Dp, class Sp>
    AbstractVector<T, D, S>& AbstractVector<T, D, S>::operator =(
            const AbstractVector<Tp, Dp, Sp>& rhs) {

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
     * AbstractVector<T, D, S>::operator ==
     */
    template<class T, unsigned int D, class S>
    bool AbstractVector<T, D, S>::operator ==(
            const AbstractVector& rhs) const {

        for (unsigned int d = 0; d < D; d++) {
            if (!IsEqual<T>(this->components[d], rhs.components[d])) {
                return false;
            }
        }

        return true;
    }


    /*
     * vislib::math::AbstractVector<T, D, S>::operator ==
     */
    template<class T, unsigned int D, class S>
    template<class Tp, unsigned int Dp, class Sp>
    bool AbstractVector<T, D, S>::operator ==(
            const AbstractVector<Tp, Dp, Sp>& rhs) const {
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
     * AbstractVector<T, D, S>::operator -
     */
    template<class T, unsigned int D, class S>
    AbstractVector<T, D, S> AbstractVector<T, D, S>::operator -(
            void) const {
        AbstractVector<T, D, S> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] = -this->components[d];
        }

        return retval;
    }


    /*
     * AbstractVector<T, D, S>::operator +
     */
    template<class T, unsigned int D, class S>
    AbstractVector<T, D, T[D]> AbstractVector<T, D, S>::operator +(
            const AbstractVector& rhs) const {
        AbstractVector<T, D, T[D]> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] = this->components[d] + rhs.components[d];
        }

        return retval;
    }


    /*
     * AbstractVector<T, D, S>::operator +=
     */
    template<class T, unsigned int D, class S>
    AbstractVector<T, D, S>& AbstractVector<T, D, S>::operator +=(
            const AbstractVector& rhs) {

        for (unsigned int d = 0; d < D; d++) {
            this->components[d] += rhs.components[d];
        }

        return *this;
    }


    /*
     * AbstractVector<T, D, S>::operator -
     */
    template<class T, unsigned int D, class S>
    AbstractVector<T, D, T[D]> AbstractVector<T, D, S>::operator -(
            const AbstractVector& rhs) const {
        AbstractVector<T, D, T[D]> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] = this->components[d] - rhs.components[d];
        }

        return retval;
    }


    /*
     * AbstractVector<T, D, S>::operator -=
     */
    template<class T, unsigned int D, class S>
    AbstractVector<T, D, S>& AbstractVector<T, D, S>::operator -=(
            const AbstractVector& rhs) {

        for (unsigned int d = 0; d < D; d++) {
            this->components[d] -= rhs.components[d];
        }
       
        return *this;
    }


    /*
     * AbstractVector<T, D, S>::operator *
     */
    template<class T, unsigned int D, class S>
    AbstractVector<T, D, T[D]> AbstractVector<T, D, S>::operator *(
            const T rhs) const {
        AbstractVector<T, D, T[D]> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] = this->components[d] * rhs;
        }

        return retval;
    }


    /*
     * AbstractVector<T, D, S>::operator *=
     */
    template<class T, unsigned int D, class S>
    AbstractVector<T, D, S>& AbstractVector<T, D, S>::operator *=(
            const T rhs) {
        for (unsigned int d = 0; d < D; d++) {
            this->components[d] *= rhs;
        }

        return *this;
    }


    /*
     * AbstractVector<T, D, S>::operator /
     */
    template<class T, unsigned int D, class S>
    AbstractVector<T, D, T[D]> AbstractVector<T, D, S>::operator /(
            const T rhs) const {
        AbstractVector<T, D, T[D]> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] = this->components[d] / rhs;
        }

        return retval;
    }


    /*
     * AbstractVector<T, D, S>::operator /=
     */
    template<class T, unsigned int D, class S>
    AbstractVector<T, D, S>& AbstractVector<T, D, S>::operator /=(
            const T rhs) {
        for (unsigned int d = 0; d < D; d++) {
            this->components[d] /= rhs;
        }

        return *this;
    }


    /*
     * AbstractVector<T, D, S>::operator *
     */
    template<class T, unsigned int D, class S>
    AbstractVector<T, D, T[D]> AbstractVector<T, D, S>::operator *(
            const AbstractVector& rhs) const {
        AbstractVector<T, D, T[D]> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] = this->components[d] * rhs.components[d];
        }

        return retval;
    }


    /*
     * AbstractVector<T, D, S>::operator *=
     */
    template<class T, unsigned int D, class S>
    AbstractVector<T, D, S>& AbstractVector<T, D, S>::operator *=(
            const AbstractVector& rhs) {

        for (unsigned int d = 0; d < D; d++) {
            this->components[d] *= rhs.components[i];
        }

        return *this;
    }


    /*
     * AbstractVector<T, D, S>::operator []
     */
    template<class T, unsigned int D, class S>
    T& AbstractVector<T, D, S>::operator [](const int i) {
        if ((i >= 0) && (i < static_cast<int>(D))) {
            return this->components[i];
        } else {
            throw OutOfRangeException(i, 0, D - 1, __FILE__, __LINE__);
        }
    }


    /*
     * AbstractVector<T, D, S>::operator []
     */
    template<class T, unsigned int D, class S>
    const T& AbstractVector<T, D, S>::operator [](const int i) const {
        if ((i >= 0) && (i < static_cast<int>(D))) {
            return this->components[i];
        } else {
            throw OutOfRangeException(i, 0, D - 1, __FILE__, __LINE__);
        }
    }


    /**
     * Scalar multiplication.
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

#endif /* VISLIB_ABSTRACTVECTOR_H_INCLUDED */
