/*
 * Polynom.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_POLYNOM_H_INCLUDED
#define VISLIB_POLYNOM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractPolynom.h"


namespace vislib {
namespace math {


    /**
     * One-dimensional polynom of degree d.
     *
     * The one-dimensional polynom is defined by its coefficients a_0 ... a_d
     * as:
     *  f(x) := a_d * x^d + a_{d-1} * x^{d-1} + ... + a_1 * x + a_0
     *
     * T scalar type
     * D Degree of the polynom
     */
    template<class T, unsigned int D>
    class Polynom : public AbstractPolynom<T, D, T[D + 1]> {
    public:

        /** Ctor. */
        Polynom(void) : Super() {
            for (unsigned int i = 0; i <= D; i++) {
                this->coefficients[i] = static_cast<T>(0);
            }
        }

        /** Dtor. */
        ~Polynom(void);

    protected:

        /** A typedef for the super class. */
        typedef AbstractPolynom<T, D, T[D + 1]> Super;

    };


    /*
     * Polynom<T, D>::~Polynom
     */
    template<class T, unsigned int D>
    Polynom<T, D>::~Polynom(void) {
        // intentionally empty
    }


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_POLYNOM_H_INCLUDED */

