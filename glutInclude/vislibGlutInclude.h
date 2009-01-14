/*
 * vislibGlutInclude.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_VISLIBGLUTINCLUDE_H_INCLUDED
#define VISLIBTEST_VISLIBGLUTINCLUDE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32
#include "glutInclude.win.h"
#else /* _WIN32 */
#include "glutInclude.lin.h"
#endif /* _WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIBTEST_VISLIBGLUTINCLUDE_H_INCLUDED */
