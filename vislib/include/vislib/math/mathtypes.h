/*
 * mathtypes.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


namespace vislib::math {

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

/** Enumeration for halfspaces. */
enum HalfSpace {
    HALFSPACE_NEGATIVE = -1, //< Halfspace opposite of normal.
    HALFSPACE_IN_PLANE = 0,  //< No halfspace, but plane itself.
    HALFSPACE_POSITIVE = 1   //< Halfspace in normal direction.
};

/** Enumeration for coordinate system types. */
enum CoordSystemType {
    COORD_SYS_LEFT_HANDED = 1, //< Identify a left-handed system
                               //  (Direct3D default).
    COORD_SYS_RIGHT_HANDED     //< Identify a right-handed system (OpenGL).
};


} // namespace vislib::math

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
