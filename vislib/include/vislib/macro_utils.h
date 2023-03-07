/*
 * macro_utils.h
 * Copyright (C) 2015 by Sebastian Grottel
 * All rights reserved. Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

/*
 * Helper macro to suppress compiler warnings
 *
 * @example: Use a template member variable internlly in a class exported by
 *           a dll.
 *
 *   VISLIB_MSVC_SUPPRESS_WARNING(4261)
 *   std::vector<my_type> my_array;
 *
 */
#ifdef _MSC_VER
#define VISLIB_MSVC_SUPPRESS_WARNING(A) __pragma(warning(suppress : A))
#else /* _MSC_VER */
#define VISLIB_MSVC_SUPPRESS_WARNING(A)
#endif /* _MSC_VER */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
