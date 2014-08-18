/*
 * ShallowPoint.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHALLOWPOINT_H_INCLUDED
#define VISLIB_SHALLOWPOINT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractPoint.h"


namespace vislib {
namespace math {

    /**
     * TODO: Documentation
     */
    template<class T, unsigned int D> 
    class ShallowPoint : public AbstractPoint<T, D, T *> {

    public:

        /**
         * Create a new Point initialised using 'coordinates' as data. The
         * Point will operate on these data. The caller is responsible that
         * the memory designated by 'components' lives as long as the object
         * and all its aliases exist.
         *
         * @param coordinates The initial Point memory. This must not be a NULL
         *                    pointer.
         */
        explicit inline ShallowPoint(T *coordinates) {
            ASSERT(coordinates != NULL);
            this->coordinates = coordinates;
        }

        /**
         * Clone 'rhs'. This operation will create an alias of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline ShallowPoint(const ShallowPoint& rhs) {
            this->coordinates = rhs.coordinates;
        }

        /** Dtor. */
        ~ShallowPoint(void);

        /**
         * Replace the coordinate pointer with a new memory location.
         * The original memory is left untouched.
         * 
         * @param coordinates The new Point memory. This must not be a NULL
         *                    pointer.
         */
        inline void SetPointer(T *coordinates) {
            ASSERT(coordinates != NULL);
            this->coordinates = coordinates;
        }

        /**
         * Assignment.
         *
         * This operation does <b>not</b> create aliases. 
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline ShallowPoint& operator =(const ShallowPoint& rhs) {
            AbstractPoint<T, D, T *>::operator =(rhs);
            return *this;
        }

        /**
         * Assignment for arbitrary Points. A valid static_cast between T and Tp
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
        inline ShallowPoint& operator =(
                const AbstractPoint<Tp, Dp, Sp>& rhs) {
            AbstractPoint<T, D, T *>::operator =(rhs);
            return *this;
        }

    private:

        /** 
         * Forbidden ctor. A default ctor would be inherently unsafe for
         * shallow Points.
         */
        inline ShallowPoint(void) {}
    };


    /*
     * ShallowPoint<T, D>::~ShallowPoint
     */
    template<class T, unsigned int D>
    ShallowPoint<T, D>::~ShallowPoint(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHALLOWPOINT_H_INCLUDED */
