/*
 * unreferenced.h
 *
 * Copyright (C) 2009 by Visualisierungsinstitut der Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_UNREFERENCED_H_INCLUDED
#define VISLIB_UNREFERENCED_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


/**
 * Define the parameter 'p' as unreferenced on purpose. Using this macro
 * prevents compiler warnings.
 *
 * @param p The parameter that is not used.
 */
#ifdef UNREFERENCED_PARAMETER
#define VL_UNREFERENCED_PARAMETER(p) UNREFERENCED_PARAMETER(p)
#else /* UNREFERENCED_PARAMETER */
#define VL_UNREFERENCED_PARAMETER(p) ((void) (p))
#endif /* UNREFERENCED_PARAMETER */


/**
 * Define the variable 'v' as unreferenced on purpose. Using this macro
 * prevents compiler warnings.
 *
 * @param v The local variable that is not used.
 */
#define VL_UNREFERENCED_LOCAL_VARIABLE(v) VL_UNREFERENCED_PARAMETER(v)


/**
 * Define the parameter 'p' as only referenced in the debug build. Using this
 * macro prevents compiler warnings.
 *
 * @param p The parameter that is only referenced in debug builds.
 */
#if (!defined(DEBUG) && !defined(_DEBUG))
#define VL_DBGONLY_REFERENCED_PARAMETER(p) VL_UNREFERENCED_PARAMETER(p)
#else /* (!defined(DEBUG) && !defined(_DEBUG)) */
#define VL_DBGONLY_REFERENCED_PARAMETER(p)
#endif /* (!defined(DEBUG) && !defined(_DEBUG)) */


/**
 * Define the variable 'v' as only referenced in the debug build. Using this 
 * macro prevents compiler warnings.
 *
 * @param v The local that is only referenced in debug builds.
 */
#define VL_DBGONLY_REFERENCED_LOCAL_VARIABLE(v) \
    VL_DBGONLY_REFERENCED_PARAMETER(v)

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_UNREFERENCED_H_INCLUDED */
