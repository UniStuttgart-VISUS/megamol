/*
 * Triangle.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_TRIANGLE_H_INCLUDED
#define VISLIB_TRIANGLE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractTriangle.h"

namespace vislib {
namespace math {


    /**
     * HURZ!
	 */
	template <class T> class Triangle : public AbstractTriangle<T, T[3]> {

    public:

        /** Ctor. */
        Triangle(void);

        /** Dtor. */
        ~Triangle(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         */
        inline Triangle& operator =(const Triangle& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        template<class Tp, class Sp>
        inline Triangle& operator =(const AbstractTriangle<Tp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

    private:
		typedef AbstractTriangle<T, T[3]> Super;
    };

 
	/*
     * vislib::math::Triangle<T>::Triangle
     */
    template<class T>
	Triangle<T>::Triangle(void) : Super() {
	}


    /*
     * vislib::math::Triangle<T>::~Triangle
     */
    template<class T>
	Triangle<T>::~Triangle(void) {
	}
    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_TRIANGLE_H_INCLUDED */

