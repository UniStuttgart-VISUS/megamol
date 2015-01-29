/*
 * versioninfo.h
 * (Console)
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Copyright (C) 2009 by SGrottel (http://www.sgrottel.de)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_CONSOLE_VERSIONINFO_H_INCLUDED
#define MEGAMOL_CONSOLE_VERSIONINFO_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "MegaMolViewer.version.gen.h"  // megamol console viewer version
#include "version.gen.h"                // megamol console version
#include "mmcore/productversion.gen.h"      // core file
#include "mmcore/productversion.h"          // core file

#define MEGAMOL_CONSOLE_MAJOR_VER MEGAMOL_PRODUCT_MAJOR_VER
#define MEGAMOL_CONSOLE_MINOR_VER MEGAMOL_PRODUCT_MINOR_VER
#define MEGAMOL_CONSOLE_MAJOR_REV MEGAMOL_CORE_API_LCREV
#define MEGAMOL_CONSOLE_MINOR_REV MEGAMOL_CONSOLE_LCREV


#if (MEGAMOL_CORE_API_DIRTY == 1) || (MEGAMOL_CONSOLE_DIRTY == 1) || (MEGAMOL_CONSOLE_VIEWER_DIRTY == 1)
#define MEGAMOL_CONSOLE_ISDIRTY 1
#define MEGAMOL_CONSOLE_DIRTYTEXT MEGAMOL_DIRTYWARNING
#else
#undef MEGAMOL_CONSOLE_ISDIRTY
#define MEGAMOL_CONSOLE_DIRTYTEXT ""
#endif


#define MEGAMOL_CONSOLE_VERSION MEGAMOL_CONSOLE_MAJOR_VER, MEGAMOL_CONSOLE_MINOR_VER, MEGAMOL_CONSOLE_MAJOR_REV, MEGAMOL_CONSOLE_MINOR_REV
#define MEGAMOL_CONSOLE_VERSION_STR MEGAMOL_STRINGIZE(MEGAMOL_CONSOLE_VERSION)

#define MEGAMOL_CONSOLE_COPYRIGHT "Copyright (c) 2006 - " MEGAMOL_STRINGIZE(MEGAMOL_CONSOLE_LCYEAR) " by " MEGAMOL_PRODUCT_COMPANY "\n" \
    "Alle Rechte vorbehalten.\n" \
    "All rights reserved.\n" MEGAMOL_CONSOLE_DIRTYTEXT

#define MEGAMOL_CONSOLE_NAME "MegaMolCon" MEGAMOL_FILENAME_BITSD
#define MEGAMOL_CONSOLE_FILENAME MEGAMOL_CONSOLE_NAME MEGAMOL_EXE_FILENAME_EXT

#define MEGAMOL_CONSOLE_COMMENTS "MegaMol Core API " MEGAMOL_STRINGIZE(MEGAMOL_CORE_API_LCREV) "\n" \
    "MegaMol Console Viewer API " MEGAMOL_STRINGIZE(MEGAMOL_CONSOLE_VIEWER_LCREV) "\n" MEGAMOL_CONSOLE_DIRTYTEXT


#endif /* MEGAMOL_CONSOLE_VERSIONINFO_H_INCLUDED */
