/*
 * memutils.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MEMUTILS_H_INCLUDED
#define VISLIB_MEMUTILS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
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
#ifndef SAFE_DELETE
#define SAFE_DELETE(ptr) if ((ptr) != NULL) {delete (ptr); (ptr) = NULL; }
#endif /* !SAFE_DELETE */


/**
 * Delete array designated by 'ptr' and set 'ptr' NULL.
 */
#ifndef ARY_SAFE_DELETE
#define ARY_SAFE_DELETE(ptr) if ((ptr) != NULL) { delete[] (ptr);\
    (ptr) = NULL; }
#endif /* !ARY_SAFE_DELETE */


/**
 * Free memory designated by 'ptr', if 'ptr' is not NULL, and set 'ptr' NULL.
 */
#ifndef SAFE_FREE
#define SAFE_FREE(ptr) if ((ptr) != NULL) { ::free(ptr); (ptr) = NULL; }
#endif /* !SAFE_FREE */


/**
 * Delete memory using ::operator delete and set 'ptr' NULL.
 */
#ifndef SAFE_OPERATOR_DELETE
#define SAFE_OPERATOR_DELETE(ptr) if ((ptr) != NULL) {\
    ::operator delete(ptr); (ptr) = NULL; }
#endif /* SAFE_OPERATOR_DELETE */


#ifndef _WIN32
/**
 * Set 'size' bytes to zero beginning at 'ptr'.
 */
#define ZeroMemory(ptr, size) memset((ptr), 0, (size))

#define SecureZeroMemory ZeroMemory
#endif /* !_WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_MEMUTILS_H_INCLUDED */
