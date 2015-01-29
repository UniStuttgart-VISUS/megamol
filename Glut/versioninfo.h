/*
 * versioninfo.h
 * (Glut)
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS)
 * Copyright (C) 2009 by SGrottel (http://www.sgrottel.de)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GLUT_VERSIONINFO_H_INCLUDED
#define MEGAMOL_GLUT_VERSIONINFO_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "MegaMolViewer.version.gen.h"  // megamol console viewer version
#include "version.gen.h"                // megamol console glut version
#include "mmcore/productversion.h"          // megamol core version
#include "visglutversion.h"             // visglut version

// from megamol core version
#define MEGAMOL_GLUT_MAJOR_VER MEGAMOL_PRODUCT_MAJOR_VER
#define MEGAMOL_GLUT_MINOR_VER MEGAMOL_PRODUCT_MINOR_VER
// from megamol console viewer version
#define MEGAMOL_GLUT_MAJOR_REV MEGAMOL_CONSOLE_VIEWER_LCREV
// from megamol console glut version
#define MEGAMOL_GLUT_MINOR_REV MEGAMOL_CONSOLE_GLUT_LCREV


#if (MEGAMOL_CONSOLE_GLUT_DIRTY == 1) || (MEGAMOL_CONSOLE_VIEWER_DIRTY == 1)
#define MEGAMOL_GLUT_ISDIRTY 1
#define MEGAMOL_GLUT_DIRTYTEXT MEGAMOL_DIRTYWARNING
#else
#undef MEGAMOL_GLUT_ISDIRTY
#define MEGAMOL_GLUT_DIRTYTEXT ""
#endif


#if (VISGLUT_DIRTY == 1)
#define VISGLUT_DIRTYTEXT " (UNCLEAN)"
#else
#define VISGLUT_DIRTYTEXT
#endif


#define MEGAMOL_GLUT_VERSION MEGAMOL_GLUT_MAJOR_VER, MEGAMOL_GLUT_MINOR_VER, MEGAMOL_GLUT_MAJOR_REV, MEGAMOL_GLUT_MINOR_REV
#define MEGAMOL_GLUT_VERSION_STR MEGAMOL_STRINGIZE(MEGAMOL_GLUT_VERSION)

#define MEGAMOL_GLUT_COPYRIGHT "Copyright (c) 2006 - " MEGAMOL_STRINGIZE(MEGAMOL_CONSOLE_GLUT_LCYEAR) " by " MEGAMOL_PRODUCT_COMPANY "\n" \
    "Alle Rechte vorbehalten.\n" \
    "All rights reserved.\n" MEGAMOL_GLUT_DIRTYTEXT

#define MEGAMOL_GLUT_NAME "VISGlut"
#define MEGAMOL_GLUT_FILENAME "MegaMolGlut" MEGAMOL_FILENAME_BITSD MEGAMOL_DLL_FILENAME_EXT

#define MEGAMOL_GLUT_COMMENTS "MegaMol Console Viewer API " MEGAMOL_STRINGIZE(MEGAMOL_VIEWER_API_LCREV) "\n"\
    "VISglut " MEGAMOL_STRINGIZE(VISGLUT_REVISION) VISGLUT_DIRTYTEXT "\n" MEGAMOL_GLUT_DIRTYTEXT

#endif /* MEGAMOL_GLUT_VERSIONINFO_H_INCLUDED */
