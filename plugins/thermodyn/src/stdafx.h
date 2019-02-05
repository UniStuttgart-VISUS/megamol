/*
 * stdafx.h
 * Copyright (C) 2006-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef _STDAFX_H_INCLUDED
#define _STDAFX_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#ifdef _WIN32
/* Windows includes */

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max

#else /* _WIN32 */
/* Linux includes */

#include <memory.h>

#ifndef NULL
#   define NULL 0
#endif

#endif /* _WIN32 */

#include "vislib/types.h"

#endif /* _STDAFX_H_INCLUDED */
