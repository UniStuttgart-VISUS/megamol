/*
 * types.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <Windows.h>

typedef INT32 VL_INT32;

#else /* _WIN32 */

#include <cstddef>
#include <inttypes.h>
#include <stdint.h>

typedef char CHAR;
typedef CHAR* PCHAR;
typedef char INT8;
typedef unsigned char UCHAR;
typedef unsigned char UINT8;
typedef unsigned char BYTE;

typedef wchar_t WCHAR;

typedef int16_t SHORT;
typedef int16_t INT16;
typedef uint16_t USHORT;
typedef uint16_t UINT16;
typedef uint16_t WORD;

typedef int32_t INT;
typedef int32_t VL_INT32;
typedef int32_t LONG;
typedef uint32_t UINT;
typedef uint32_t UINT32;
typedef uint32_t ULONG;
typedef uint32_t DWORD;

typedef int64_t LONGLONG;
typedef int64_t INT64;
typedef uint64_t ULONGLONG;
typedef uint64_t UINT64;


// TODO: Remove float/double or add bool etc.?

typedef float FLOAT;
typedef double DOUBLE;

#ifdef _LIN64
typedef INT64 INT_PTR;
typedef UINT64 UINT_PTR;
typedef UINT64 ULONG_PTR;
#else  /* _LIN64 */
typedef VL_INT32 INT_PTR;
typedef UINT32 UINT_PTR;
typedef UINT32 ULONG_PTR;
#endif /* _LIN64 */

typedef size_t SIZE_T;

#ifndef SIZE_MAX
#define SIZE_MAX (static_cast<size_t>(-1))
#endif /* !SIZE_MAX */

#endif /* _WIN32 */

typedef UINT64 EXTENT;

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
