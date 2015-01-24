/*
 * ShallowPlane.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHALLOWPLANE_H_INCLUDED
#define VISLIB_SHALLOWPLANE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractPlane.h"


namespace vislib {
namespace math {


    /**
     * This class implements a plane that uses user specified memory for its
     * data - we call this a shallow plane, as the user remains owner of the
     * memory he provides.
     */
    template<class T> class ShallowPlane : public AbstractPlane<T, T *>{

    public:

        /**
         * Create a new plane using 'parameters' as storage. 'parameters'
         * must be an array of at least four T. The user must guarantee that
         * it exists as long as this object and all its clones.
         *
         * @param parameters The storage for the plane parameters.
         */
        inline ShallowPlane(T *parameters) {
            ASSERT(parameters != NULL);
            this->parameters = parameters;
        }

        /**
         * Clone 'rhs'. This operation will alias 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline ShallowPlane(const ShallowPlane& rhs) {
            this->parameters = rhs.parameters;
        }

        /** Dtor. */
        ~ShallowPlane(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline ShallowPlane& operator =(const ShallowPlane& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Assignment. This operator allows arbitrary plane to plane 
         * conversions.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        inline ShallowPlane& operator =(const AbstractPlane<Tp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** Typedef of the super class. */
        typedef AbstractPlane<T, T *> Super;

    };


    /*
     * ShallowPlane<T>::~ShallowPlane
     */
    template<class T> ShallowPlane<T>::~ShallowPlane(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHALLOWPLANE_H_INCLUDED */
