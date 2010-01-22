/*
 * stdafx.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_PROTEIN_STDAFX_H_INCLUDED
#define MEGAMOL_PROTEIN_STDAFX_H_INCLUDED
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

#define MMXML_CHAR wchar_t

#else /* _WIN32 */
/* Linux includes */


#include <memory.h>
#include <pthread.h>

#ifndef NULL
#   define NULL 0
#endif

#define MMXML_CHAR char

#endif /* _WIN32 */

#include <stdlib.h>
#include "vislib/types.h"
#include "vislib/String.h"

typedef vislib::String<vislib::CharTraits<MMXML_CHAR> > MMXML_STRING;

#endif /* MEGAMOL_PROTEIN_STDAFX_H_INCLUDED */
