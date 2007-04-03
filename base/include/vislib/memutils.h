/*
 * memutils.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MEMUTILS_H_INCLUDED
#define VISLIB_MEMUTILS_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <memory.h>
#include <stdlib.h>


#ifndef NULL
#ifdef __cplusplus
#define NULL (0)
#else
#define NULL ((void *) 0)
#endif /* __cplusplus */
#endif /* NULL */


/**
 * Delete memory designated by 'ptr' and set 'ptr' NULL.
 */
#define SAFE_DELETE(ptr) delete (ptr); (ptr) = NULL;


/**
 * Delete array designated by 'ptr' and set 'ptr' NULL.
 */
#define ARY_SAFE_DELETE(ptr) delete[] (ptr); (ptr) = NULL;


/**
 * Free memory designated by 'ptr', if 'ptr' is not NULL, and set 'ptr' NULL.
 */
#define SAFE_FREE(ptr) if ((ptr) != NULL) { ::free(ptr); (ptr) = NULL; }


/**
 * Delete memory using ::operator delete and set 'ptr' NULL.
 */
#define SAFE_OPERATOR_DELETE(ptr) if ((ptr) != NULL) {\
    ::operator delete(ptr); (ptr) = NULL; }


#ifndef _WIN32
/**
 * Set 'size' bytes to zero beginning at 'ptr'.
 */
#define ZeroMemory(ptr, size) memset((ptr), 0, (size))
#endif /* !_WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_MEMUTILS_H_INCLUDED */
