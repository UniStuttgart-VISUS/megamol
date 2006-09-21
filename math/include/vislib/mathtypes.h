/*
 * mathtypes.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MATHTYPES_H_INCLUDED
#define VISLIB_MATHTYPES_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


namespace vislib {
namespace math {

    /** Value of constant PI in type double */
    extern const double PI_DOUBLE;

    /** Epsilon value for floating point comparison. */
    extern const float FLOAT_EPSILON;

    /** Epsilon value for double precision floating point comparison. */
    extern const double DOUBLE_EPSILON;

    /** type for angles in degrees */
    typedef float AngleDeg;

    /** type for angles in radians */
    typedef float AngleRad;


} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_MATHTYPES_H_INCLUDED */
