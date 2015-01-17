/*
 * productversion.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Copyright (C) 2009 by SGrottel (http://www.sgrottel.de)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_PRODUCTVERSION_H_INCLUDED
#define MEGAMOL_PRODUCTVERSION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#ifndef MEGAMOL_STRINGIZE_DEFINED
#define MEGAMOL_STRINGIZE_DEFINED 1

#define MEGAMOL_STRINGIZE_DO(arg) #arg
#define MEGAMOL_STRINGIZE(arg) MEGAMOL_STRINGIZE_DO(arg)

#endif /* MEGAMOL_STRINGIZE_DEFINED */


#if defined(_WIN64) || defined(_LIN64)
#if defined(DEBUG) || defined(_DEBUG)
#define MEGAMOL_FILENAME_BITSD "64d"
#else /* defined(DEBUG) || defined(_DEBUG) */
#define MEGAMOL_FILENAME_BITSD "64"
#endif /* defined(DEBUG) || defined(_DEBUG) */
#else /* defined(_WIN64) || defined(_LIN64) */
#if defined(DEBUG) || defined(_DEBUG)
#define MEGAMOL_FILENAME_BITSD "32d"
#else /* defined(DEBUG) || defined(_DEBUG) */
#define MEGAMOL_FILENAME_BITSD "32"
#endif /* defined(DEBUG) || defined(_DEBUG) */
#endif /* defined(_WIN64) || defined(_LIN64) */

#ifdef _WIN32
#define MEGAMOL_DLL_FILENAME_EXT ".dll"
#define MEGAMOL_EXE_FILENAME_EXT ".exe"
#else /* _WIN32 */
#define MEGAMOL_DLL_FILENAME_EXT ".so"
#define MEGAMOL_EXE_FILENAME_EXT
#endif /* _WIN32 */


#define MEGAMOL_PRODUCT_MAJOR_VER 1
#define MEGAMOL_PRODUCT_MINOR_VER 1
#define MEGAMOL_PRODUCT_MAJOR_REV 0
#define MEGAMOL_PRODUCT_MINOR_REV 0


#define MEGAMOL_DIRTYWARNING "UNCLEAN PRERELEASE VERSION. ONLY USE FOR DEVELOPMENT.\n"

#include "productversion.gen.h"


#define MEGAMOL_PRODUCT_VERSION MEGAMOL_PRODUCT_MAJOR_VER, MEGAMOL_PRODUCT_MINOR_VER, MEGAMOL_PRODUCT_MAJOR_REV, MEGAMOL_PRODUCT_MINOR_REV
#define MEGAMOL_PRODUCT_VERSION_STR MEGAMOL_STRINGIZE(MEGAMOL_PRODUCT_VERSION)

#define MEGAMOL_PRODUCT_NAME "MegaMol"
#define MEGAMOL_PRODUCT_COMPANY "MegaMol Consortium: VISUS (Universitaet Stuttgart, Germany), TU Dresden (Dresden, Germany)"


#endif /* MEGAMOL_PRODUCTVERSION_H_INCLUDED */
