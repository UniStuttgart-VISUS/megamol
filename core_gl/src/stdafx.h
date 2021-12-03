/*
 * stdafx.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_STDAFX_H_INCLUDED
#define MEGAMOLCORE_STDAFX_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#ifdef _WIN32
/* Windows includes */

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#undef min
#undef max

#else /* _WIN32 */
/* Linux includes */

#include <memory.h>
#include <pthread.h>

#ifndef NULL
#define NULL 0
#endif

#endif /* _WIN32 */

#include "vislib/String.h"
#include "vislib/types.h"
#include <stdlib.h>

/* common includes */
// #define USE_LOG_SUFFIX true
#ifndef USE_LOG_SUFFIX
#define USE_LOG_SUFFIX false
#endif /* USE_LOG_SUFFIX */

//#ifndef _mmc_stringize
//#define _mmc_stringize2(x) #x
//#define _mmc_stringize(x) _mmc_stringize2(x)
//#endif /* _mmc_stringize */
//
//#define MEGAMOL_TRACE_LEVEL vislib::Trace::LEVEL_ALL
//#define MEGAMOL_TRACE_LEVEL vislib::Trace::LEVEL_VL -1

#endif /* MEGAMOLCORE_STDAFX_H_INCLUDED */
