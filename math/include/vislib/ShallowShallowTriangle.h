/*
* ShallowShallowTriangle.h
*
* Copyright (C) 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
* Alle Rechte vorbehalten.
*/

#ifndef VISLIB_SHALLOWSHALLOWTRIANGLE_H_INCLUDED
#define VISLIB_SHALLOWSHALLOWTRIANGLE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractTriangle.h"
#include "vislib/ShallowPoint.h"
#include "vislib/memutils.h"

namespace vislib {
namespace math {


    /**
    * Note: T MUST basically be float/double. or else
    */
    template <class T, unsigned int D> class ShallowShallowTriangle :
        public AbstractTriangle<ShallowPoint<T, D>, ShallowPoint<T, D>*> {

    public:

        /** Ctor. */
        ShallowShallowTriangle(T *memory);

        /** Dtor. */
        ~ShallowShallowTriangle(void);

        /**
         * Answer the pointer to the internal memory
         *
         * @return The vertex pointer
         */
        inline T *GetPointer(void);

        /**
        * Replace the vertex pointer with a new memory location.
        * The original memory is left untouched.
        * 
        * @param memory The vertex memory. This must not be a NULL
        *                    pointer.
        */
        inline void SetPointer(T *memory);

        /**
        * Assignment.
        *
        * @param rhs The right hand side operand.
        *
        * @return *this
        */
        inline ShallowShallowTriangle& operator =(const ShallowShallowTriangle& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /**
        * Assignment.
        *
        * @param rhs The right hand side operand.
        *
        * @return *this
        */
        template<class Tp, class Sp>
        inline ShallowShallowTriangle& operator =(const AbstractTriangle<Tp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** Memory the shallow points will be placed in */
        unsigned char b[3 * sizeof(ShallowPoint<T, D>)];

    private:

        /** definition of super-class type */
        typedef AbstractTriangle<ShallowPoint<T, D>, ShallowPoint<T, D>*> Super;

    };


    /*
     * vislib::math::ShallowShallowTriangle<T>::ShallowShallowTriangle
     */
    template<class T, unsigned int D>
    ShallowShallowTriangle<T, D>::ShallowShallowTriangle(T *memory) {
        this->vertices = reinterpret_cast<ShallowPoint<T, D>*>(&b);
        for (unsigned int d = 0; d < 3; d++) {
            new (&this->vertices[d]) ShallowPoint<T, D>(memory + d * D);
        }
    }


    /*
     * vislib::math::ShallowShallowTriangle<T>::~ShallowShallowTriangle
     */
    template<class T, unsigned int D>
    ShallowShallowTriangle<T, D>::~ShallowShallowTriangle(void) {
        for (unsigned int d = 0; d < 3; d++) {
            this->vertices[d].~ShallowPoint<T, D>();
        }
    }


    /*
     * vislib::math::ShallowShallowTriangle<T>::GetPointer
     */
    template<class T, unsigned int D>
    T *ShallowShallowTriangle<T, D>::GetPointer(void) {
        return this->vertices[0].PeekCoordinates();
    }


    /*
     * vislib::math::ShallowShallowTriangle<T>::SetPointer
     */
    template<class T, unsigned int D>
    void ShallowShallowTriangle<T, D>::SetPointer(T *memory) {
        for (unsigned int d = 0; d < 3; d++) {
            this->vertices[d].SetPointer(memory + d * D);
        }
    }


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHALLOWSHALLOWTRIANGLE_H_INCLUDED */

