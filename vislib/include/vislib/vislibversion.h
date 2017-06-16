/*
 * vislibversion.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_VISLIBVERSION_H_INCLUDED
#define VISLIB_VISLIBVERSION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

 /*
 * VISLIB_STRINGIZING_HELPERS for macro argument expansion
 */
#ifndef VISLIB_STRINGIZING_HELPERS
#define VISLIB_STRINGIZING_HELPERS

#define VISLIB_STRINGIZING_HELPER_INTERNAL(arg) #arg
#define VISLIB_STRINGIZING_HELPER(arg) VISLIB_STRINGIZING_HELPER_INTERNAL(arg)
#define VISLIB_STRINGIZING_W_HELPER_INTERNAL(arg) L ## #arg
#define VISLIB_STRINGIZING_W_HELPER(arg) VISLIB_STRINGIZING_W_HELPER_INTERNAL(arg)

#endif /* VISLIB_STRINGIZING_HELPERS */


/**
 * The vislib version number ANSI string.
 * Implementation note: Add no spaces after new line!!!
 */
#define VISLIB_VERSION_STR VISLIB_STRINGIZING_HELPER(VISLIB_VERSION_MAJOR.\
VISLIB_VERSION_MINOR.VISLIB_VERSION_REVISION)


/**
 * The vislib version number ANSI string.
 */
#define VISLIB_VERSION_STRA VISLIB_VERSION_STR


/**
 * The vislib version number unicode string.
 * Implementation note: Add no spaces after new line!!!
 */
#define VISLIB_VERSION_STRW VISLIB_STRINGIZING_W_HELPER(VISLIB_VERSION_MAJOR.\
VISLIB_VERSION_MINOR.VISLIB_VERSION_REVISION)


#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_VISLIBVERSION_H_INCLUDED */
