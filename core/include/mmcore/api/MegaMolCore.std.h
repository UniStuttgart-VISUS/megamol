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

#ifdef _WIN32
#ifndef GL_LOAD_DLL
#error All MegaMol projects must define GL_LOAD_DLL (Preprocessor option)
#endif /* GL_LOAD_DLL */
#endif /* _WIN32 */

/**
 * Struct to be returned by a plugin api holding compatibility informations
 */
typedef struct _mmplg_compatibilityvalues_t {
    SIZE_T size; // number of bytes of this struct
    const char* mmcoreRev; // MegaMol™ core revision
    const char* vislibRev; // VISlib revision or zero if no vislib is used
} mmplgCompatibilityValues;


#ifdef _WIN32
/**
 * defines to control the export and import of functions
 */
#   ifdef MEGAMOLCORE_EXPORTS
#       define MEGAMOLCORE_API __declspec(dllexport)
#       define MEGAMOLCORE_APIEXT
#   else /* MEGAMOLCORE_EXPORTS */
#       define MEGAMOLCORE_API __declspec(dllimport)
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
