/*
 * AbstractDimensionImpl.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTDIMENSIONIMPL_H_INCLUDED
#define VISLIB_ABSTRACTDIMENSIONIMPL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/assert.h"
#include "vislib/mathfunctions.h"
#include "vislib/OutOfRangeException.h"
#include <memory.h>


namespace vislib {
namespace math {


    /**
     * This template implements the most parts of the dimension behaviour. Its
     * only intended use is being the super class of the AbstractDimension 
     * template and its partial template specialisation. The same reuse of code
     * as in AbstractVectorImpl is the reason for this class.
     *
     * The following template parameters are used:
     *
     * T: The type used for the components of the dimension.
     * D: The dimensionality of the dimension, e. g. 2 for representing 2D sizes.
     * S: The "storage class". This can be either T[D] for a "deep dimension" or
     *    T * for a "shallow dimensions". Other instantiations are inherently 
     *    dangerous and should never be used.
     * C: The direct subclass, i. e. AbstractDimension. This allows the 
     *    implementation to create the required return values. Other 
     *    instantiations are inherently dangerous and should never be used.
     */
    template<class T, unsigned int D, class S, 
            template<class T, unsigned int D, class S> class C> 
            class AbstractDimensionImpl {

    public:

        /** Dtor. */
        ~AbstractDimensionImpl(void);

        /**
         * Access the internal dimension data directly.
         *
         * @return A pointer to the dimension data.
         */
        inline const T *PeekDimension(void) const {
            return this->dimension;
        }

        /**
         * Access the internal dimension data directly.
         *
         * @return A pointer to the dimension data.
         */
        inline T *PeekDimension(void) {
            return this->dimension;
        }

        /**
         * Perform a uniform scale of the dimension.
         *
         * @param factor The scaling factor.
         */
        void Scale(const double factor);

        /**
         * Assignment operator. This operator never creates an alias.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractDimensionImpl& operator =(const C<T, D, S>& rhs);

        /** 
         * This operator allows for arbitrary dimension to dimension 
         * assignments. If the left hand side operand has a smaller dimension,
         * the values will be truncated, if it has larger dimensionality, it 
         * will be padded with zeroes.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Tp, unsigned int Dp, class Sp>
        AbstractDimensionImpl& operator =(const C<Tp, Dp, Sp>& rhs);

        /**
         * Answer, whether this Dimension and rhs are equal. The IsEqual 
         * function is used for comparing the components.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal.
         */
        bool operator ==(const C<T, D, S>& rhs) const;

        /**
         * Answer, whether this Dimension and rhs are equal. The IsEqual 
         * function of the left hand operand is used for comparing the 
         * components. If D and Dp do not match, the dimensions are
         * never equal.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal.
         */
        template<class Tp, unsigned int Dp, class Sp>
        bool operator ==(const C<Tp, Dp, Sp>& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal.
         */
        inline bool operator !=(const C<T, D, S>& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal.
         */
        template<class Tp, unsigned int Dp, class Sp>
        inline bool operator !=(const C<Tp, Dp, Sp>& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Component access. 'i' must be within [0, D[.
         *
         * @param i The component to be accessed.
         *
         * @return The component's value.
         *
         * @throws OutOfRangeException If 'i' is out of range.
         */
        T& operator [](const int i);

        /**
         * Component access. 'i' must be within [0, D[.
         *
         * @param i The component to be accessed.
         *
         * @return The component's value.
         *
         * @throws OutOfRangeException If 'i' is out of range.
         */
        const T& operator [](const int i) const;

    protected:

        /**
         * Disallow instances of this class.
         */
        inline AbstractDimensionImpl(void) {}

        /** The extents wrapped by this class. */
        S dimension;
    };


    /*
     * vislib::math::AbstractDimensionImpl<T, D, S, C>::~AbstractDimensionImpl
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    AbstractDimensionImpl<T, D, S, C>::~AbstractDimensionImpl(void) {
    }


    /*
     * vislib::math::AbstractDimensionImpl<T, D, S, C>::Scale
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    void AbstractDimensionImpl<T, D, S, C>::Scale(const double factor) {
        for (unsigned int d = 0; d < D; d++) {
            this->dimension[d] = static_cast<T>(factor 
                * static_cast<double>(this->dimension[d]));
        }
    }


    /* 
     * vislib::math::AbstractDimensionImpl<T, D, S, C>::operator =
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    AbstractDimensionImpl<T, D, S, C>& 
    AbstractDimensionImpl<T, D, S, C>::operator =(const C<T, D, S>& rhs) {

        if (this != &rhs) {
            ::memcpy(this->dimension, rhs.dimension, D * sizeof(T));
        }

        return *this;
    }


    /* 
     * vislib::math::AbstractDimensionImpl<T, D, S, C>::operator =
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    template<class Tp, unsigned int Dp, class Sp> 
    AbstractDimensionImpl<T, D, S, C>& 
    AbstractDimensionImpl<T, D, S, C>::operator =(const C<Tp, Dp, Sp>& rhs) {
        
        if (static_cast<void *>(this) != static_cast<const void *>(&rhs)) {
            for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
                this->dimension[d] = static_cast<T>(rhs[d]);
            }
            for (unsigned int d = Dp; d < D; d++) {
                this->dimension[d] = static_cast<T>(0);
            }   
        }

        return *this;
    }


    /*
     * vislib::math::AbstractDimensionImpl<T, D, S, C>::operator ==
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    bool AbstractDimensionImpl<T, D, S, C>::operator ==(
            const C<T, D, S>& rhs) const {
        for (unsigned int d = 0; d < D; d++) {
            if (!IsEqual(this->dimension[d], rhs.dimension[d])) {
                return false;
            }
        }
        /* No difference found. */

        return true;
    }


    /*
     * vislib::math::AbstractDimensionImpl<T, D, S, C>::operator ==
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    template<class Tp, unsigned int Dp, class Sp>
    bool AbstractDimensionImpl<T, D, S, C>::operator ==(
            const C<Tp, Dp, Sp>& rhs) const {
        if (D != Dp) {
            /* Cannot be equal. */
            return false;
        }

        for (unsigned int d = 0; d < D; d++) {
            if (!IsEqual<T>(this->dimension[d], static_cast<T>(rhs[d]))) {
                return false;
            }
        }
        /* No difference found. */

        return true;
    }


    /*
     * vislib::math::AbstractDimensionImpl<T, D, S, C>::operator []
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    T& AbstractDimensionImpl<T, D, S, C>::operator [](const int i) {
        ASSERT(0 <= i);
        ASSERT(i < static_cast<int>(D));

        if ((0 <= i) && (i < static_cast<int>(D))) {
            return this->dimension[i];
        } else {
            throw OutOfRangeException(i, 0, D - 1, __FILE__, __LINE__);
        }
    }


    /*
     * vislib::math::AbstractDimensionImpl<T, D, S, C>::operator []
     */
    template<class T, unsigned int D, class S, 
        template<class T, unsigned int D, class S> class C> 
    const T& AbstractDimensionImpl<T, D, S, C>::operator [](const int i) const {
        ASSERT(0 <= i);
        ASSERT(i < static_cast<int>(D));

        if ((0 <= i) && (i < static_cast<int>(D))) {
            return this->dimension[i];
        } else {
            throw OutOfRangeException(i, 0, D - 1, __FILE__, __LINE__);
        }
    }
    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTDIMENSIONIMPL_H_INCLUDED */
