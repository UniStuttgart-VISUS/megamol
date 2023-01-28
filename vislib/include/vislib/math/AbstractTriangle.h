/*
 * AbstractTriangle.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTTRIANGLE_H_INCLUDED
#define VISLIB_ABSTRACTTRIANGLE_H_INCLUDED
#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/math/AbstractTriangleImpl.h"


namespace vislib::math {


/**
 * This class represents a triangle.
 *
 * The triangle can be instantiated for different built-in number
 * types T and storage classes S. For a triangle that holds its own
 * vertices, use T[3] as storage class (this is the default). For a
 * triangle that does not own the storage (we call this "shallow
 * triangle"), use T * for S.
 */
template<class T, class S>
class AbstractTriangle : public AbstractTriangleImpl<T, S, AbstractTriangle> {

public:
    template<class Tf1, class Sf1, template<class Tf2, class Sf2> class Cf>
    friend class AbstractTriangleImpl;

    /** Dtor. */
    ~AbstractTriangle();

protected:
    /** Ctor. */
    AbstractTriangle();

    AbstractTriangle& operator=(const AbstractTriangle& rhs) {
        Super::operator=(rhs);
        return *this;
    }

    template<class Tp, class Sp>
    inline AbstractTriangle& operator=(const AbstractTriangle<Tp, Sp>& rhs) {
        Super::operator=(rhs);
        return *this;
    }

private:
    typedef AbstractTriangleImpl<T, S, vislib::math::AbstractTriangle> Super;
};


/*
 * vislib::math::AbstractTriangle<T>::AbstractTriangle
 */
template<class T, class S>
AbstractTriangle<T, S>::AbstractTriangle() : Super() {}


/*
 * vislib::math::AbstractTriangle<T>::~AbstractTriangle
 */
template<class T, class S>
AbstractTriangle<T, S>::~AbstractTriangle() {}

} // namespace vislib::math

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTTRIANGLE_H_INCLUDED */
