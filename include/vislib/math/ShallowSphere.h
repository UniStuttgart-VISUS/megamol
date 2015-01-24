/*
 * ShallowSphere.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHALLOWSPHERE_H_INCLUDED
#define VISLIB_SHALLOWSPHERE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractSphere.h"


namespace vislib {
namespace math {


    /**
     * This class is the shallow version of a sphere which does not own the
     * memory of the center point and radius.
     */
    template<class T> class ShallowSphere : public AbstractSphere<T, T *> {

    public:

        /**
         * Create a new sphere using the array 'xyzr' as storage. 'xyzr'
         * must be an array of at least four T, where the first three element
         * represent the center point and the fourth the radius. The user must 
         * guarantee that it exists as long as this object and all its clones.
         *
         * @param parameters The storage for the sphere parameters.
         */
        inline ShallowSphere(T *xyzr) {
            ASSERT(xyzr != NULL);
            this->xyzr = xyzr;
        }

        /**
         * Clone 'rhs'. This operation will alias 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline ShallowSphere(const ShallowSphere& rhs) {
            this->xyzr = rhs.xyzr;
        }

        /** Dtor. */
        ~ShallowSphere(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline ShallowSphere& operator =(const ShallowSphere& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Assignment. This operator allows arbitrary sphere to sphere 
         * conversions.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        inline ShallowSphere& operator =(const AbstractSphere<Tp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:
    
        /** The superclass type. */
        typedef AbstractSphere<T, T *> Super;

    };


    /*
     * vislib::math::ShallowSphere<T>::~ShallowSphere
     */
    template<class T> ShallowSphere<T>::~ShallowSphere(void) {
    }
    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHALLOWSPHERE_H_INCLUDED */

