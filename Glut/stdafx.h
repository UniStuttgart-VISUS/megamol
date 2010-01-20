/*
 * stdafx.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLVIEWER_STDAFX_H_INCLUDED
#define MEGAMOLVIEWER_STDAFX_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#ifdef _WIN32
/* Windows includes */

#ifndef WINVER // Windows XP or later.
#define WINVER 0x0501
#endif

#ifndef _WIN32_WINNT // Windows XP or later.
#define _WIN32_WINNT 0x0501
#endif

#ifndef _WIN32_WINDOWS // Windows 98 or later.
#define _WIN32_WINDOWS 0x0410
#endif

#ifndef _WIN32_IE // IE 6.0 or later.
#define _WIN32_IE 0x0600
#endif

#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>

#else /* _WIN32 */
/* Linux includes */

#include <memory.h>

#ifndef NULL
#   define NULL 0
#endif

#endif /* _WIN32 */

#include "vislib/types.h"

#endif /* MEGAMOLVIEWER_STDAFX_H_INCLUDED */
