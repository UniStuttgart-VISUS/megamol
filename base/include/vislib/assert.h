/*
 * assert.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ASSERT_H_INCLUDED
#define VISLIB_ASSERT_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <cassert>


#ifndef ASSERT
#if (defined(DEBUG) || defined(_DEBUG))
#define ASSERT(exp) assert(exp)
#else
#define ASSERT(exp) ((void) 0)
#endif /* (defined(DEBUG) || defined(_DEBUG)) */
#endif /* ASSERT */


#define BECAUSE_I_KNOW(exp) ASSERT(exp)


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
#endif /* VISLIB_ASSERT_H_INCLUDED */
