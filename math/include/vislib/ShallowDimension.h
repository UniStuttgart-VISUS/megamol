/*
 * ShallowDimension.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHALLOWDIMENSION_H_INCLUDED
#define VISLIB_SHALLOWDIMENSION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractDimension.h"


namespace vislib {
namespace math {


    /**
     * This class represents a shallow dimension, that uses memory provided
     * by the caller.
     */
    template<class T, unsigned int D> 
    class ShallowDimension : public AbstractDimension<T, D, T *> {

    public:

        /**
         * Construct a dimension from an array holding its values. The array
         * 'dimension' holds the size in the following order: width, height,
         * higher dimensions ...
         *
         * @param dimension The dimension values.
         */
        explicit inline ShallowDimension(T *dimension) {
            ASSERT(dimension != NULL);
            this->dimension = dimension;
        }

        /**
         * Copy ctor. This ctor creates an alias!
         *
         * @param rhs The object to clone.
         */
        inline ShallowDimension(const ShallowDimension& rhs) {
            this->dimension = rhs.dimension;
        }

        /** Dtor. */
        ~ShallowDimension(void);

        /**
         * Assigment operator. This operator never creates an alias, even for
         * shallow dimensions!
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline ShallowDimension& operator =(const ShallowDimension& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Assigment operator. This operator never creates an alias, even for
         * shallow dimensions!
         *
         * This assignment allows for arbitrary dimension to dimension
         * conversions.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Tp, unsigned int Dp, class Sp>
        inline ShallowDimension& operator =(
                const AbstractDimension<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    public:

        /** Typedef for the super class. */
        typedef AbstractDimension<T, D, T *> Super;
    };


    /*
     * vislib::math::ShallowDimension<T, D>::~ShallowDimension
     */
    template<class T, unsigned int D> 
    ShallowDimension<T, D>::~ShallowDimension(void) {
    }


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHALLOWDIMENSION_H_INCLUDED */
