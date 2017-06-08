/*
 * forceinline.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FORCEINLINE_H_INCLUDED
#define VISLIB_FORCEINLINE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _MSC_VER
#define VISLIB_FORCEINLINE __forceinline
#else /* _MSC_VER */
#define VISLIB_FORCEINLINE inline
#endif /* _MSC_VER */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_FORCEINLINE_H_INCLUDED */
