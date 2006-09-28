/*
 * Dimension.h  28.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_DIMENSION_H_INCLUDED
#define VISLIB_DIMENSION_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* _MSC_VER > 1000 */


#include "vislib/assert.h"
#include "vislib/AbstractDimension.h"


namespace vislib {
namespace math {

    /**
     * This class represents extents in D-dimensional space. 
     */
    template<class T, unsigned int D> class Dimension 
            : public AbstractDimension<T, D, T[D]> {

    public:

        /**
         * Create a zero sized dimension.
         */
        Dimension(void);

        /**
         * Create a dimension using the data from the 'dimension' array.
         *
         * @param dimension The initial value in an array. This pointer must not
         *                  be NULL.
         */
        inline Dimension(const T *dimension) {
            ASSERT(dimension != NULL);
            ::memcpy(this->dimension, dimension, D * sizeof(T));
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Dimension(const Dimension& rhs) {
            ::memcpy(this->dimension, rhs.dimension, D * sizeof(T));
        }

        /**
         * Allows arbitrary Dimension to dimension conversion. If the new
         * object has a smaller dimensionality, the values will be truncated,
         * if it has larger dimensionality, it will be padded with zeroes.
         *
         * @param rhs The object to copy.
         */
        template<class Tp, unsigned int Dp, class Sp>
        Dimension(const AbstractDimension<Tp, Dp, Sp>& rhs);

        /** Dtor. */
        ~Dimension(void);

        /**
         * Assignment operator. This operator never creates an alias.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline Dimension& operator =(const Dimension& rhs) {
            Super::operator =(rhs);
            return *this;
        }

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
        inline Dimension& operator =(const AbstractDimension<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** A typedef for the super class. */
        typedef AbstractDimension<T, D, T[D]> Super;

    };


    /*
     * vislib::math::Dimension<T, D>::Dimension
     */
    template<class T, unsigned int D> 
    Dimension<T, D>::Dimension(void) {
        for (unsigned int d = 0; d < D; d++) {
            this->dimension[d] = static_cast<T>(0);
        }
    }


    /*
     * vislib::math::Dimension<T, D>::Dimension
     */
    template<class T, unsigned int D> 
    template<class Tp, unsigned int Dp, class Sp>
    Dimension<T, D>::Dimension(const AbstractDimension<Tp, Dp, Sp>& rhs) {
        for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
            this->dimension[d] = static_cast<T>(rhs[d]);
        }
        for (unsigned int d = Dp; d < D; d++) {
            this->dimension[d] = static_cast<T>(0);
        }  
    }


    /*
     * vislib::math::Dimension<T, D>::~Dimension
     */
    template<class T, unsigned int D> 
    Dimension<T, D>::~Dimension(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_DIMENSION_H_INCLUDED */
