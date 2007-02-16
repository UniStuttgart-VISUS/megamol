/*
 * tchar.h  09.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_TCHAR_H_INCLUDED
#define VISLIB_TCHAR_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32
#include <tchar.h>
#else /* _WIN32 */

#if defined(UNICODE) || defined(_UNICODE)

#include <wchar.h>

typedef wchar_t TCHAR;

#define _T(x) L##x

#define _tprintf wprintf
#define _ftprintf fwprintf
#define _vftprintf vfwprintf
#define _tcslen wcslen
#define _vsntprintf vswprintf
#define _tcscpy wcscpy
#define _tsystem wsystem

#else /* defined(UNICODE) || defined(_UNICODE) */

typedef char TCHAR;

#define _T(x) x

#define _tprintf printf
#define _ftprintf fprintf
#define _vftprintf vfprintf
#define _tcslen strlen
#define _vsntprintf vsnprintf
#define _tcscpy strcpy
#define _tsystem system

#endif /* defined(UNICODE) || defined(_UNICODE) */

#endif /* _WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_TCHAR_H_INCLUDED */
