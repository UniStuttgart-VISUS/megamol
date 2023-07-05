/*
 * assert.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <cassert>


#ifndef ASSERT
#if (defined(DEBUG) || defined(_DEBUG))
#define ASSERT(exp) assert(exp)
#else
#define ASSERT(exp) ((void)0)
#endif /* (defined(DEBUG) || defined(_DEBUG)) */
#endif /* ASSERT */


#define BECAUSE_I_KNOW(exp) ASSERT(exp)


// If DEBUG and NDEBUG are defined at the same time, the VERIFY macro goes mad!
#if (defined(DEBUG) || defined(_DEBUG)) && defined(NDEBUG)
#error "{ DEBUG | _DEBUG } and NDEBUG must not be defined at the same time!"
#endif /* (defined(DEBUG) || defined(_DEBUG)) && defined(NDEBUG) */

#ifndef VERIFY
#if (defined(DEBUG) || defined(_DEBUG))
#define VERIFY(exp) ASSERT(exp)
#else
#define VERIFY(exp) exp
#endif /* (defined(DEBUG) || defined(_DEBUG)) */
#endif /* VERIFY */


#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
