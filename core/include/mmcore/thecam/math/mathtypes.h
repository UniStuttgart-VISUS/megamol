/*
 * thecam/math/types.h
 *
 * Copyright (c) 2012, TheLib Team (http://www.thelib.org/license)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of TheLib, TheLib Team, nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THELIB TEAM AS IS AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THELIB TEAM BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*
 * mathtypes.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef THE_MATH_TYPES_H_INCLUDED
#define THE_MATH_TYPES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include <cfloat>

#include "mmcore/thecam/utility/config.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/thecam/utility/not_instantiable.h"
#include "mmcore/thecam/utility/types.h"


namespace megamol {
namespace core {
namespace thecam {
namespace math {

/** type for angles in degrees */
typedef float angle_deg_t;


/** type for angles in radians */
typedef float angle_rad_t;

/** Enumeration for halfspaces. */
enum class half_space {
    negative = -1, //< Halfspace opposite of normal.
    in_plane = 0,  //< No halfspace, but plane itself.
    positive = 1   //< Halfspace in normal direction.
};


/** Possible types of memory layouts for matrices */
enum class matrix_layout {

    /** The memory layout is unknown. */
    unknown = 0,

    /**
     * Elements are arranged such that consecutive elements of the columns
     * are contiguous in memory.
     *
     * In column-major layout, the rows vary faster.
     *
     * The element (row, column) of an (R, C) matrix can be found at
     * column * R + row, provided that indices start at zero.
     *
     * Column-major layout is the layout used for multi-dimensional arrays
     * in Fortran and for matrices in OpenGL.
     */
    column_major,

    /**
     * Elements are arranged such that consecutive elements of the rows
     * are contiguous in memory.
     *
     * In row-major layout, the columns vary faster.
     *
     * The element (row, column) of an (R, C) matrix can be found at
     * row * C + column, provided that indices start at zero.
     *
     * Row-major layout is the layout used for multi-dimensional arrays in
     * C/C++ and C#.
     */
    row_major
};


/** Utility class providing an epsilon value. */
template <class T> class MEGAMOLCORE_API epsilon : public utility::not_instantiable {
public:
    /** The epsilon value. */
    static constexpr const T value = static_cast<T>(DBL_EPSILON);
};


/** Utility class providing an epsilon value. */
template <> class MEGAMOLCORE_API epsilon<float> : public utility::not_instantiable {
public:
    /** The epsilon value. */
    static constexpr const float value = FLT_EPSILON;
};


/** Utility class providing an epsilon value. */
template <> class MEGAMOLCORE_API epsilon<double> : public utility::not_instantiable {
public:
    /** The epsilon value. */
    static constexpr const double value = DBL_EPSILON;
};


/** Utility class for providing the value of Euler's number. */
template <class T> class MEGAMOLCORE_API euler_e : public utility::not_instantiable {
public:
    /* The value of Euler's number. */
    static constexpr const T value = static_cast<T>(2.71828182845904523536);
};

/** Utility class for providing the value of Euler's number. */
template <> class MEGAMOLCORE_API euler_e<double> : public utility::not_instantiable {
public:
    /* The value of Euler's number. */
    static constexpr const double value = 2.71828182845904523536;
};

/** Utility class providing the value of Pi. */
template <class T> class MEGAMOLCORE_API pi : public utility::not_instantiable {
public:
    /* The value of Pi. */
    static constexpr const T value = static_cast<T>(3.14159265358979323846);
};


/** Utility class providing the value of Pi. */
template <> class MEGAMOLCORE_API pi<double> : public utility::not_instantiable {
public:
    /* The value of Pi. */
    static constexpr const double value = 3.14159265358979323846;
};

} /* end namespace math */
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_MATH_TYPES_H_INCLUDED */
