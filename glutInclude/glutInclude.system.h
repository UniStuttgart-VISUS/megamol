/*
 * glutInclude.system.h
 *
 * Copyright (C) 2007 - 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 *
 * Include this file for a system glut library.
 */

#ifndef VISLIBTEST_GLUTINCLUDE_SYSTEM_H_INCLUDED
#define VISLIBTEST_GLUTINCLUDE_SYSTEM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

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

#endif /* VISLIBTEST_GLUTINCLUDE_SYSTEM_H_INCLUDED */
