/*
 * versioninfo.h
 * (WGL)
 *
 * Copyright (C) 2009-2011 by Universitaet Stuttgart (VISUS)
 * Copyright (C) 2009 by SGrottel (http://www.sgrottel.de)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_WGL_VERSIONINFO_H_INCLUDED
#define MEGAMOL_WGL_VERSIONINFO_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "MegaMolViewer.version.gen.h"  // megamol console viewer version
#include "version.gen.h"                // megamol console glut version
#include "mmcore/productversion.h"          // megamol core version

// from megamol core version
#define MEGAMOL_WGL_MAJOR_VER MEGAMOL_PRODUCT_MAJOR_VER
#define MEGAMOL_WGL_MINOR_VER MEGAMOL_PRODUCT_MINOR_VER
// from megamol console viewer version
#define MEGAMOL_WGL_MAJOR_REV MEGAMOL_CONSOLE_VIEWER_LCREV
// from megamol console glut version
#define MEGAMOL_WGL_MINOR_REV MEGAMOL_CONSOLE_WGL_LCREV


#if (MEGAMOL_CONSOLE_WGL_DIRTY == 1) || (MEGAMOL_CONSOLE_VIEWER_DIRTY == 1)
#define MEGAMOL_WGL_ISDIRTY 1
#define MEGAMOL_WGL_DIRTYTEXT MEGAMOL_DIRTYWARNING
#else
#undef MEGAMOL_WGL_ISDIRTY
#define MEGAMOL_WGL_DIRTYTEXT ""
#endif


#if (VISGLUT_DIRTY == 1)
#define VISGLUT_DIRTYTEXT " (UNCLEAN)"
#else
#define VISGLUT_DIRTYTEXT
#endif


#define MEGAMOL_WGL_VERSION MEGAMOL_WGL_MAJOR_VER, MEGAMOL_WGL_MINOR_VER, MEGAMOL_WGL_MAJOR_REV, MEGAMOL_WGL_MINOR_REV
#define MEGAMOL_WGL_VERSION_STR MEGAMOL_STRINGIZE(MEGAMOL_WGL_VERSION)

#define MEGAMOL_WGL_COPYRIGHT "Copyright (c) 2006 - " MEGAMOL_STRINGIZE(MEGAMOL_CONSOLE_WGL_LCYEAR) " by " MEGAMOL_PRODUCT_COMPANY "\n" \
    "Alle Rechte vorbehalten.\n" \
    "All rights reserved.\n" MEGAMOL_WGL_DIRTYTEXT

#define MEGAMOL_WGL_NAME "MegaMolWGL"
#define MEGAMOL_WGL_FILENAME "MegaMolWGL" MEGAMOL_FILENAME_BITSD MEGAMOL_DLL_FILENAME_EXT

#define MEGAMOL_WGL_COMMENTS "MegaMol Console Viewer API " MEGAMOL_STRINGIZE(MEGAMOL_VIEWER_API_LCREV) "\n"\
    MEGAMOL_WGL_DIRTYTEXT

#endif /* MEGAMOL_WGL_VERSIONINFO_H_INCLUDED */
