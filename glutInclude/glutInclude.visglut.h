/*
 * glutInclude.visglut.h
 *
 * Copyright (C) 2007 - 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 *
 * Include this file for the visglut library.
 */

#ifndef VISLIBTEST_GLUTINCLUDE_VISGLUT_H_INCLUDED
#define VISLIBTEST_GLUTINCLUDE_VISGLUT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#ifdef _WIN32
#ifdef _MSC_VER
#ifdef _WIN64
#if defined _DEBUG || defined DEBUG 
#pragma comment(lib, "%visglutPath%/freeglut/lib/win64d/freeglut.lib")
#else /* DEBUG */
#pragma comment(lib, "%visglutPath%/freeglut/lib/win64/freeglut.lib")
#endif /* DEBUG */
#else /* _WIN64 */
#if defined _DEBUG || defined DEBUG 
#pragma comment(lib, "%visglutPath%/freeglut/lib/win32d/freeglut.lib")
#else /* DEBUG */
#pragma comment(lib, "%visglutPath%/freeglut/lib/win32/freeglut.lib")
#endif /* DEBUG */
#endif /* _WIN64 */
#pragma comment(linker, "/NODEFAULTLIB:freeglut_static.lib")
#endif /* _MSC_VER */
#endif /* _WIN32 */

#define FREEGLUT_STATIC

#include "%visglutPath%/freeglut/include/GL/glut.h"

#endif /* VISLIBTEST_GLUTINCLUDE_VISGLUT_H_INCLUDED */
