/*
 * AbstractDimension.h  28.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTDIMENSION_H_INCLUDED
#define VISLIB_ABSTRACTDIMENSION_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* _MSC_VER > 1000 */


#include "vislib/assert.h"
#include "vislib/OutOfRangeException.h"


namespace vislib {
namespace math {

    /**
     * This class represents extents in D-dimensional space. 
     */
    template<class T, unsigned int D, class S> class AbstractDimension {

    public:

        /** Dtor. */
        ~AbstractDimension(void);

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
         * Assignment operator. This operator never creates an alias.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractDimension& operator =(const AbstractDimension& rhs);

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
        AbstractDimension& operator =(const AbstractDimension<Tp, Dp, Sp>& rhs);

        /**
         * Answer, whether this Dimension and rhs are equal. The IsEqual 
         * function is used for comparing the components.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal.
         */
        bool operator ==(const AbstractDimension& rhs) const;

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
        bool operator ==(const AbstractDimension<Tp, Dp, Sp>& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal.
         */
        inline bool operator !=(const AbstractDimension& rhs) const {
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
        inline bool operator !=(
                const AbstractDimension<Tp, Dp, Sp>& rhs) const {
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
        const T& operator [](const int i ) const;

    protected:

        /**
         * Disallow instances of this class.
         */
        inline AbstractDimension(void) {}

        /** The extents wrapped by this class. */
        S dimension;
    };


    /*
     * vislib::math::AbstractDimension<T, D, S>::~AbstractDimension
     */
    template<class T, unsigned int D, class S> 
    AbstractDimension<T, D, S>::~AbstractDimension(void) {
    }


    /* 
     * vislib::math::AbstractDimension<T, D, S>::operator =
     */
    template<class T, unsigned int D, class S> 
    AbstractDimension<T, D, S>& AbstractDimension<T, D, S>::operator =(
            const AbstractDimension& rhs) {

        if (this != &rhs) {
            ::memcpy(this->dimension, rhs.dimension, D * sizeof(T));
        }

        return *this;
    }


    /* 
     * vislib::math::AbstractDimension<T, D, S>::operator =
     */
    template<class T, unsigned int D, class S> 
    template<class Tp, unsigned int Dp, class Sp> 
    AbstractDimension<T, D, S>& AbstractDimension<T, D, S>::operator =(
            const AbstractDimension<Tp, Dp, Sp>& rhs) {
        
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
     * vislib::math::AbstractDimension<T, D, S>::operator ==
     */
    template<class T, unsigned int D, class S> 
    bool AbstractDimension<T, D, S>::operator ==(
            const AbstractDimension& rhs) const {
        for (unsigned int d = 0; d < D; d++) {
            if (!IsEqual(this->dimension[d], rhs.dimension[d])) {
                return false;
            }
        }
        /* No difference found. */

        return true;
    }


    /*
     * vislib::math::AbstractDimension<T, D, S>::operator ==
     */
    template<class T, unsigned int D, class S> 
    template<class Tp, unsigned int Dp, class Sp>
    bool AbstractDimension<T, D, S>::operator ==(
            const AbstractDimension<Tp, Dp, Sp>& rhs) const {
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
     * vislib::math::AbstractDimension<T, D, S>::operator []
     */
    template<class T, unsigned int D, class S> 
    T& AbstractDimension<T, D, S>::operator [](const int i) {
        ASSERT(0 <= i);
        ASSERT(i < static_cast<int>(D));

        if ((0 <= i) && (i < static_cast<int>(D))) {
            return this->dimension[i];
        } else {
            throw OutOfRangeException(i, 0, D - 1, __FILE__, __LINE__);
        }
    }


    /*
     * vislib::math::AbstractDimension<T, D, S>::operator []
     */
    template<class T, unsigned int D, class S> 
    const T& AbstractDimension<T, D, S>::operator [](const int i) const {
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

#endif /* VISLIB_ABSTRACTDIMENSION_H_INCLUDED */
