/*
 * ShallowTriangle.h
 *
 * Copyright (C) 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHALLOWTRIANGLE_H_INCLUDED
#define VISLIB_SHALLOWTRIANGLE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractTriangle.h"

namespace vislib {
namespace math {


	template <class T> class ShallowTriangle : public AbstractTriangle<T, T*> {

    public:

        /**
         * Create a new triangle initialised using 'vertices' as data. The
         * vector will operate on these data. The caller is responsible that
         * the memory designated by 'vertices' lives as long as the object
         * and all its aliases exist.
         *
         * @param vertices The initial triangle memory. This must not be a NULL
         *                   pointer.
         */
        explicit inline ShallowTriangle(T *vertices) {
            ASSERT(vertices != NULL);
            this->vertices = vertices;
        }


        /** Dtor. */
        ~ShallowTriangle(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline ShallowTriangle& operator =(const ShallowTriangle& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        template<class Tp, class Sp>
        inline ShallowTriangle& operator =(const AbstractTriangle<Tp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

    private:
		typedef AbstractTriangle<T, T*> Super;
    };

 
    /*
     * vislib::math::Triangle<T>::~Triangle
     */
    template<class T>
	ShallowTriangle<T>::~ShallowTriangle(void) {
	}
    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHALLOWTRIANGLE_H_INCLUDED */

