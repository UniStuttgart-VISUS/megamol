/*
 * MegaMolCore.std.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_STD_H_INCLUDED
#define MEGAMOLCORE_STD_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#include <stdlib.h>

/**
 * Struct to be returned by a plugin api holding compatibility informations
 */
typedef struct _mmplg_compatibilityvalues_t {
    size_t size; // number of bytes of this struct
    const char* mmcoreRev; // MegaMol� core revision
    const char* vislibRev; // VISlib revision or zero if no vislib is used
} mmplgCompatibilityValues;


#ifdef _WIN32
/**
 * defines to control the export and import of functions
 */
#   ifdef MEGAMOLCORE_EXPORTS
#       define MEGAMOLCORE_API /*__declspec(dllexport)*/
#       define MEGAMOLCORE_APIEXT
#   else /* MEGAMOLCORE_EXPORTS */
#       define MEGAMOLCORE_API /*__declspec(dllimport)*/
#       define MEGAMOLCORE_APIEXT extern
#   endif /* MEGAMOLCORE_EXPORTS */
#   define MEGAMOLCORE_CALL __stdcall
#   define MEGAMOLCORE_CALLBACK __stdcall
#else /* _WIN32 */
#   define MEGAMOLCORE_API
#   ifdef MEGAMOLCORE_EXPORTS
#       define MEGAMOLCORE_APIEXT
#   else /* MEGAMOLCORE_EXPORTS */
#       define MEGAMOLCORE_APIEXT extern
#   endif /* MEGAMOLCORE_EXPORTS */
#   define MEGAMOLCORE_CALL
#   define MEGAMOLCORE_CALLBACK
#endif /* _WIN32 */


#endif /* MEGAMOLCORE_STD_H_INCLUDED */
