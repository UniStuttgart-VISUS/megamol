/*
 * Vector.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_VECTOR_H_INCLUDED
#define VISLIB_VECTOR_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include <limits>

#include "vislib/EqualFunc.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/StorageClass.h"


namespace vislib {
namespace math {

    /**

     *
     * The template parameters are:
     * T is the type of the elements that the vector consists of.
     * D is the dimension of the vector.
     * E is the equal functor for determining the equality of two scalars of 
     *   type T
     * S is the storage class used for storing the elements of the vector.
     */
    template<class T, unsigned int D, class E = EqualFunc<T>, 
            template<class, unsigned int> class S = DeepStorageClass> 
    class Vector {

    public:

        /**
         * Create a new vector. 
         * 
         * This operation is not supported for ShallowStorageClass as template 
         * parameter S as it is inherently unsafe and dangerous to public 
         * safety.
         */
        inline Vector(void) {}

        /**
         * Create a new vector initialised with 'components'. 'components' must
         * not be a NULL pointer. 
         *
         * If the template parameter S is ShallowStorageClass, the new object 
         * will alias the memory designated by 'components'.
         *
         * @param components The initial vector components.
         */
        explicit inline Vector(const T *components) : components(components) {}

        /**
         * Clone 'rhs'.
         *
         * If the template parameter S is ShallowStorageClass, the new object
         * will alias the vector memory of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Vector(const Vector& rhs) : components(rhs.components) {}

        /**
         * Create a copy of 'vector'. This ctor allows for arbitrary vector to
         * vector conversions.
         *
         * This operation is not supported for ShallowStorageClass as template 
         * parameter S as it is inherently unsafe and dangerous to public 
         * safety.
         *
         * @param vector The vector to be cloned
         */
        template<class Tp, unsigned int Dp, class Ep,
            template<class, unsigned int> class Sp>
        Vector(const Vector<Tp, Dp, Ep, Sp>& vector);

        /** Dtor. */
        virtual ~Vector(void);

        /**
         * Answer the dot product of this vector and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The dot product of this vector and 'rhs'.
         */
        T Dot(const Vector& rhs) const;

        /**
         * Answer whether the vector is a null vector.
         *
         * @return true, if the vector is a null vector, false otherwise.
         */
        bool IsNull(void) const;

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
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        Vector& operator =(const Vector& rhs);

        /**
         * Assigment for arbitrary vectors. A valid static_cast between T and Tp
         * is a precondition for instantiating this template.
         *
         * This operation does <b>not</b> create aliases, even for 
         * ShallowStorageClass as storage class template parameter. For shallow
         * vectors, the components of 'rhs' are copied to the shallow storage
         * of the object.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        template<class Tp, unsigned int Dp, class Ep, 
            template<class, unsigned int> class Sp>
        Vector& operator =(const Vector<Tp, Dp, Ep, Sp>& rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this vector are equal, false otherwise.
         */
        bool operator ==(const Vector& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this vector are not equal, false otherwise.
         */
        inline bool operator !=(const Vector& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Negate the vector.
         *
         * @return The negated version of this vector.
         */
        Vector operator -(void) const;

        /**
         * Answer the sum of this vector and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The sum of this and 'rhs'.
         */
        Vector operator +(const Vector& rhs) const;

        /**
         * Add 'rhs' to this vector and answer the sum.
         *
         * @param rhs The right hand side operand.
         *
         * @return The sum of this and 'rhs'.
         */
        Vector& operator +=(const Vector& rhs);

        /**
         * Answer the difference between this vector and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The difference between this and 'rhs'.
         */
        Vector operator -(const Vector& rhs) const;

        /**
         * Subtract 'rhs' from this vector and answer the difference.
         *
         * @param rhs The right hand side operand.
         *
         * @return The difference between this and 'rhs'.
         */
        Vector& operator -=(const Vector& rhs);

        /**
         * Scalar multiplication.
         *
         * @param rhs The right hand side operand.
         *
         * @return The result of the scalar multiplication.
         */
        Vector operator *(const T rhs) const;

        /**
         * Scalar multiplication assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return The result of the scalar multiplication.
         */
        Vector& operator *=(const T rhs);

        /**
         * Scalar division operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return The result of the scalar division.
         */
        Vector operator /(const T rhs) const;

        /**
         * Scalar division assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return The result of the scalar division.
         */
        Vector& operator /=(const T rhs);

        /**
         * Performs a component-wise multiplication.
         *
         * @param rhs The right hand side operand.
         *
         * @return The product of this and rhs.
         */
        Vector operator *(const Vector& rhs) const;

        /**
         * Multiplies 'rhs' component-wise with this vector and returns
         * the result.
         *
         * @param rhs The right hand side operand.
         *
         * @return The product of this and rhs
         */
        Vector& operator *=(const Vector& rhs);

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

        /**
         * Cast to a T array. This operator exposes the internal vector
         * holding the vector components to the caller. The object remains
         * owner of the memory returned.
         *
         * @return The vector components in a three component T array.
         */
        inline operator T *(void) {
            return static_cast<T *>(this->components);
        }

        /**
         * Cast to a T array. This operator exposes the internal vector
         * holding the vector components to the caller. The object remains
         * owner of the memory returned.
         *
         * @return The vector components in a three component T array.
         */
        inline operator const T *(void) const {
            return static_cast<const T *>(this->components);
        }

    protected:

        /** 
         * The vector components. This can be a ShallowVectorStorage or
         * DeepVectorStorage instantiation.
         */
        S<T, D> components;
    };


    /*
     * Vector<T, D, E, S>::Vector
     */
    template<class T, unsigned int D, class E, 
        template<class, unsigned int> class S>
    template<class Tp, unsigned int Dp, class Ep, 
        template<class, unsigned int> class Sp>
    Vector<T, D, E, S>::Vector(const Vector<Tp, Dp, Ep, Sp>& rhs) {
        for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
            this->components[d] = static_cast<T>(rhs[d]);
        }
        for (unsigned int d = Dp; d < D; d++) {
            this->components[d] = static_cast<T>(0);
        }
    }


    /*
     * Vector<T, D, E, S>::~Vector
     */
    template<class T, unsigned int D, class E, 
        template<class, unsigned int> class S>
    Vector<T, D, E, S>::~Vector(void) {
    }


    /*
     * Vector<T, D, E, S>::Dot
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    T Vector<T, D, E, S>::Dot(const Vector& rhs) const {
        T retval = static_cast<T>(0);

        for (unsigned int d = 0; d < D; d++) {
            retval += this->components[d] * rhs.components[d];
        }

        return retval;
    }


    /*
     * Vector<T, D, E, S>::IsNull
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    bool Vector<T, D, E, S>::IsNull(void) const {
        for (unsigned int d = 0; d < D; d++) {
            if (!E()(this->components[d], static_cast<T>(0))) {
                return false;
            }
        }
        /* No non-null value found. */

        return true;
    }


    /*
     * Vector<T, D, E, S>::Length
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    T Vector<T, D, E, S>::Length(void) const {
        T retval = static_cast<T>(0);

        for (unsigned int d = 0; d < D; d++) {
            retval += Sqr(this->components[d]);
        }

        return Sqrt(retval);
    }


    /*
     * Vector<T, D, E, S>::MaxNorm
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    T Vector<T, D, E, S>::MaxNorm(void) const {
#ifdef _MSC_VER
#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max
#endif /* _MSC_VER */
        T retval = std::numeric_limits<T>::is_integer 
            ? std::numeric_limits<T>::min() : -std::numeric_limits<T>::max();
#ifdef _MSC_VER
#define min
#define max
#pragma pop_macro("min")
#pragma pop_macro("max")
#endif /* _MSC_VER */

        for (unsigned int d = 0; d < D; d++) {
            if (this->components[d] > retval) {
                retval = this->components[d];
            }
        }

        return retval;
    }


    /*
     * Vector<T, D, E, S>::Normalise
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    T Vector<T, D, E, S>::Normalise(void) {
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
     * Vector<T, D, E, S>::operator =
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    Vector<T, D, E, S>& Vector<T, D, E, S>::operator =(const Vector& rhs) {
        if (this != &rhs) {
            ::memcpy(static_cast<T *>(this->components), 
                static_cast<const T *>(this->components), D * sizeof(T));
        }

        return *this;
    }


    /*
     * Vector<T, D, E, S>::operator =
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    template<class Tp, unsigned int Dp, class Ep,
        template<class, unsigned int> class Sp>
    Vector<T, D, E, S>& Vector<T, D, E, S>::operator =(
            const Vector<Tp, Dp, Ep, Sp>& rhs) {
        if (this != &rhs) {
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
     * Vector<T, D, E, S>::operator ==
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    bool Vector<T, D, E, S>::operator ==(const Vector& rhs) const {
        for (unsigned int d = 0; d < D; d++) {
            if (!E()(this->components[d], rhs.components[d])) {
                return false;
            }
        }

        return true;
    }


    /*
     * Vector<T, D, E, S>::operator -
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    Vector<T, D, E, S> Vector<T, D, E, S>::operator -(void) const {
        Vector<T, D, E, S> retval;

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] = -this->components[d];
        }

        return retval;
    }


    /*
     * Vector<T, D, E, S>::operator +
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    Vector<T, D, E, S> Vector<T, D, E, S>::operator +(const Vector& rhs) const {
        Vector<T, D, E, S> retval(*this);

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] += rhs.components[d];
        }

        return retval;
    }


    /*
     * Vector<T, D, E, S>::operator +=
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    Vector<T, D, E, S>& Vector<T, D, E, S>::operator +=(const Vector& rhs) {

        for (unsigned int d = 0; d < D; d++) {
            this->components[d] += rhs.components[d];
        }

        return *this;
    }


    /*
     * Vector<T, D, E, S>::operator -
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    Vector<T, D, E, S> Vector<T, D, E, S>::operator -(const Vector& rhs) const {
        Vector<T, D, E, S> retval(*this);

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] -= rhs.components[d];
        }

        return retval;
    }


    /*
     * Vector<T, D, E, S>::operator -=
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    Vector<T, D, E, S>& Vector<T, D, E, S>::operator -=(const Vector& rhs) {

        for (unsigned int d = 0; d < D; d++) {
            this->components[d] -= rhs.components[d];
        }
       
        return *this;
    }


    /*
     * Vector<T, D, E, S>::operator *
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    Vector<T, D, E, S> Vector<T, D, E, S>::operator *(const T rhs) const {
        Vector<T, D, E, S> retval(*this);

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] *= rhs;
        }

        return retval;
    }


    /*
     * Vector<T, D, E, S>::operator *=
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    Vector<T, D, E, S>& Vector<T, D, E, S>::operator *=(const T rhs) {
        for (unsigned int d = 0; d < D; d++) {
            this->components[d] *= rhs;
        }

        return *this;
    }


    /*
     * Vector<T, D, E, S>::operator /
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    Vector<T, D, E, S> Vector<T, D, E, S>::operator /(const T rhs) const {
        Vector<T, D, E, S> retval(*this);

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] /= rhs;
        }

        return retval;
    }


    /*
     * Vector<T, D, E, S>::operator /=
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    Vector<T, D, E, S>& Vector<T, D, E, S>::operator /=(const T rhs) {
        for (unsigned int d = 0; d < D; d++) {
            this->components[d] /= rhs;
        }

        return *this;
    }


    /*
     * Vector<T, D, E, S>::operator *
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    Vector<T, D, E, S> Vector<T, D, E, S>::operator *(const Vector& rhs) const {
        Vector<T, D, E, S> retval(*this);

        for (unsigned int d = 0; d < D; d++) {
            retval.components[d] *= rhs.components[d];
        }

        return retval;
    }


    /*
     * Vector<T, D, E, S>::operator *=
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    Vector<T, D, E, S>& Vector<T, D, E, S>::operator *=(const Vector& rhs) {

        for (unsigned int d = 0; d < D; d++) {
            this->components[d] *= rhs.components[i];
        }

        return *this;
    }


    /*
     * Vector<T, D, E, S>::operator []
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    T& Vector<T, D, E, S>::operator [](const int i) {
        if ((i >= 0) && (i < static_cast<int>(D))) {
            return this->components[i];
        } else {
            throw OutOfRangeException(i, 0, D - 1, __FILE__, __LINE__);
        }
    }


    /*
     * Vector<T, D, E, S>::operator []
     */
    template<class T, unsigned int D, class E,
        template<class, unsigned int> class S>
    const T& Vector<T, D, E, S>::operator [](const int i) const {
        if ((i >= 0) && (i < static_cast<int>(D))) {
            return this->components[i];
        } else {
            throw OutOfRangeException(i, 0, D - 1, __FILE__, __LINE__);
        }
    }


} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_VECTOR_H_INCLUDED */
