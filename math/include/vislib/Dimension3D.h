/*
 * Dimension3D.h  28.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_DIMENSION3D_H_INCLUDED
#define VISLIB_DIMENSION3D_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* _MSC_VER > 1000 */


#include "vislib/assert.h"
#include "vislib/AbstractDimension3D.h"


namespace vislib {
namespace math {

    /**
     * This class represents extents in three-dimensional space. 
     */
    template<class T> class Dimension3D : public AbstractDimension3D<T, T[3]> {

    public:

        /**
         * Create a zero sized dimension.
         */
        Dimension3D(void);

        /**
         * Create a new dimension.
         *
         * @param width  The width.
         * @param height The height.
         * @param depth The depth.
         */
        inline Dimension3D(const T& width, const T& height, const T& depth) {
            this->dimension[0] = width;
            this->dimension[1] = height;
            this->dimension[2] = depth;
        }

        /**
         * Create a dimension using the data from the 'dimension' array.
         *
         * @param dimension The initial value in an array. This pointer must not
         *                  be NULL.
         */
        inline Dimension3D(const T *dimension) {
            ASSERT(dimension != NULL);
            this->dimension[0] = dimension[0];
            this->dimension[1] = dimension[1];
            this->dimension[2] = dimension[2];
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Dimension3D(const Dimension3D& rhs) {
            this->dimension[0] = rhs.dimension[0];
            this->dimension[1] = rhs.dimension[1];
            this->dimension[2] = rhs.dimension[2];
        }

        /**
         * Allows arbitrary Dimension3D to dimension conversion. If the new
         * object has a smaller dimensionality, the values will be truncated,
         * if it has larger dimensionality, it will be padded with zeroes.
         *
         * @param rhs The object to copy.
         */
        template<class Tp, unsigned int Dp, class Sp>
        Dimension3D(const AbstractDimension<Tp, Dp, Sp>& rhs);

        /** Dtor. */
        ~Dimension3D(void);

        /**
         * Assignment operator. This operator never creates an alias.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline Dimension3D& operator =(const Dimension3D& rhs) {
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
        inline Dimension3D& operator =(
                    const AbstractDimension<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** A typedef for the super class. */
        typedef AbstractDimension3D<T, T[3]> Super;

    };


    /*
     * vislib::math::Dimension3D<T>::Dimension3D
     */
    template<class T> 
    Dimension3D<T>::Dimension3D(void) {
        this->dimension[0] = this->dimension[1] = this->dimension[2] 
            = static_cast<T>(0);
    }


    /*
     * vislib::math::Dimension3D<T>::Dimension3D
     */
    template<class T> 
    template<class Tp, unsigned int Dp, class Sp>
    Dimension3D<T>::Dimension3D(const AbstractDimension<Tp, Dp, Sp>& rhs) {
            this->dimension[0] = (Dp < 1) ? static_cast<T>(0) 
                                          : static_cast<T>(rhs[0]);
            this->dimension[1] = (Dp < 2) ? static_cast<T>(0) 
                                          : static_cast<T>(rhs[1]);
            this->dimension[2] = (Dp < 3) ? static_cast<T>(0) 
                                          : static_cast<T>(rhs[2]);
    }


    /*
     * vislib::math::Dimension3D<T>::~Dimension3D
     */
    template<class T> 
    Dimension3D<T>::~Dimension3D(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_DIMENSION3D_H_INCLUDED */
