/*
 * glutInclude.h
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 *
 * Include this file for a system glut library.
 */

#ifndef VISLIBTEST_GLUTINCLUDE_H_INCLUDED
#define VISLIBTEST_GLUTINCLUDE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#ifdef _MSC_VER
#ifdef _WIN32
#ifdef _WIN64
#pragma comment(lib, "glut64.lib")
#else /* _WIN64 */
#pragma comment(lib, "glut32.lib")
#endif /* _WIN64 */
#endif /* _WIN32 */
#endif /* _MSC_VER */

#include <GL/glut.h>

#endif /* VISLIBTEST_GLUTINCLUDE_H_INCLUDED */
