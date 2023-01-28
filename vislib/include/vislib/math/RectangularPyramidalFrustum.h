/*
 * RectangularPyramidalFrustum.h
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_RECTANGULARPYRAMIDALFRUSTUM_H_INCLUDED
#define VISLIB_RECTANGULARPYRAMIDALFRUSTUM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/IllegalParamException.h"
#include "vislib/math/AbstractRectangularPyramidalFrustum.h"


namespace vislib::math {


/**
 * TODO: comment class
 */
template<class T>
class RectangularPyramidalFrustum : public AbstractRectangularPyramidalFrustum<T, T[15]> {

public:
    /** Ctor. */
    inline RectangularPyramidalFrustum() : Super() {
        this->values[Super::IDX_BOTTOM] = static_cast<T>(0);
        this->values[Super::IDX_TOP] = static_cast<T>(0);
        this->values[Super::IDX_LEFT] = static_cast<T>(0);
        this->values[Super::IDX_RIGHT] = static_cast<T>(0);
        this->values[Super::IDX_NEAR] = static_cast<T>(0);
        this->values[Super::IDX_FAR] = static_cast<T>(0);
        this->values[Super::IDX_APEX_X] = static_cast<T>(0);
        this->values[Super::IDX_APEX_Y] = static_cast<T>(0);
        this->values[Super::IDX_APEX_Z] = static_cast<T>(0);
        this->values[Super::IDX_NORMAL_X] = static_cast<T>(0);
        this->values[Super::IDX_NORMAL_Y] = static_cast<T>(0);
        this->values[Super::IDX_NORMAL_Z] = static_cast<T>(1);
        this->values[Super::IDX_UP_X] = static_cast<T>(0);
        this->values[Super::IDX_UP_Y] = static_cast<T>(1);
        this->values[Super::IDX_UP_Z] = static_cast<T>(0);
        ShallowVector<T, 3> tmp(this->values + Super::IDX_UP_X);
        this->safeUpVector(tmp);
    }

    template<class Sp1, class Sp2, class Sp3, class Sp4>
    RectangularPyramidalFrustum(const AbstractViewFrustum<T, Sp1>& frustum, const AbstractPoint<T, 3, Sp2>& apex,
        const AbstractVector<T, 3, Sp3>& baseNormal, const AbstractVector<T, 3, Sp4>& baseUp);

    /** Dtor. */
    virtual ~RectangularPyramidalFrustum();

protected:
    /** Superclass typedef. */
    typedef AbstractRectangularPyramidalFrustum<T, T[15]> Super;
};


/*
 * ...RectangularPyramidalFrustum<T>::RectangularPyramidalFrustum
 */
template<class T>
template<class Sp1, class Sp2, class Sp3, class Sp4>
RectangularPyramidalFrustum<T>::RectangularPyramidalFrustum(const AbstractViewFrustum<T, Sp1>& frustum,
    const AbstractPoint<T, 3, Sp2>& apex, const AbstractVector<T, 3, Sp3>& baseNormal,
    const AbstractVector<T, 3, Sp4>& baseUp)
        : Super() {
    this->Set(frustum, apex, baseNormal, baseUp);
}


/*
 * ...RectangularPyramidalFrustum<T>::~RectangularPyramidalFrustum
 */
template<class T>
RectangularPyramidalFrustum<T>::~RectangularPyramidalFrustum() {}

} // namespace vislib::math

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_RECTANGULARPYRAMIDALFRUSTUM_H_INCLUDED */
