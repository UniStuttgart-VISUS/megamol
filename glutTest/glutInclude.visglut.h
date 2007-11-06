/*
 * glutInclude.h
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 *
 * Include this file for the visglut library.
 */

#ifndef VISLIBTEST_GLUTINCLUDE_H_INCLUDED
#define VISLIBTEST_GLUTINCLUDE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#ifdef _MSC_VER
#ifdef _WIN32
#ifdef _WIN64
#if defined _DEBUG || defined DEBUG 
#pragma comment(lib, "%visglutWinPath%/freeglut/lib/win64d/freeglut.lib")
#else /* DEBUG */
#pragma comment(lib, "%visglutWinPath%/freeglut/lib/win64/freeglut.lib")
#endif /* DEBUG */
#else /* _WIN64 */
#if defined _DEBUG || defined DEBUG 
#pragma comment(lib, "%visglutWinPath%/freeglut/lib/win32d/freeglut.lib")
#else /* DEBUG */
#pragma comment(lib, "%visglutWinPath%/freeglut/lib/win32/freeglut.lib")
#endif /* DEBUG */
#endif /* _WIN64 */
#endif /* _WIN32 */
#pragma comment(linker, "/NODEFAULTLIB:freeglut_static.lib")
#endif /* _MSC_VER */

#define FREEGLUT_STATIC

#ifdef _WIN32
#include "%visglutWinPath%/freeglut/include/GL/glut.h"
#else /* _WIN32 */
#include "%visglutLinPath%/freeglut/include/GL/glut.h"
#endif /* _WIN32 */

#endif /* VISLIBTEST_GLUTINCLUDE_H_INCLUDED */
