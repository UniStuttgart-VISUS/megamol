/*
 * vislibsymbolimportexport.inl
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_VISLIBSYMBOLIMPORTEXPORT_INL_INCLUDED
#define VISLIB_VISLIBSYMBOLIMPORTEXPORT_INL_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32

#if defined(VISLIB_SYMBOL_EXPORT)
#define VISLIB_STATICSYMBOL __declspec(dllexport)
#elif defined(VISLIB_SYMBOL_IMPORT)
#define VISLIB_STATICSYMBOL __declspec(dllimport)
#else
#define VISLIB_STATICSYMBOL
#endif

#else /* _WIN32 */

#if defined(VISLIB_SYMBOL_EXPORT)
#define VISLIB_STATICSYMBOL
#elif defined(VISLIB_SYMBOL_IMPORT)
#define VISLIB_STATICSYMBOL extern
#else
#define VISLIB_STATICSYMBOL
#endif

#endif /* _WIN32 */


#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_VISLIBSYMBOLIMPORTEXPORT_INL_INCLUDED */
