/*
 * versioninfo.h
 * (Core)
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS)
 * Copyright (C) 2009 by SGrottel (http://www.sgrottel.de)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_CORE_VERSIONINFO_H_INCLUDED
#define MEGAMOL_CORE_VERSIONINFO_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/productversion.h"


#define MEGAMOL_CORE_MAJOR_VER MEGAMOL_VERSION_MAJOR
#define MEGAMOL_CORE_MINOR_VER MEGAMOL_VERSION_MINOR
#define MEGAMOL_CORE_MAJOR_REV ""
#define MEGAMOL_CORE_MINOR_REV ""

#if (MEGAMOL_CORE_DIRTY == 1) || (MEGAMOL_CORE_API_DIRTY == 1)
#define MEGAMOL_CORE_ISDIRTY 1
#define MEGAMOL_CORE_DIRTYTEXT MEGAMOL_DIRTYWARNING
#else
#undef MEGAMOL_CORE_ISDIRTY
#define MEGAMOL_CORE_DIRTYTEXT ""
#endif


#define MEGAMOL_CORE_VERSION MEGAMOL_CORE_MAJOR_VER, MEGAMOL_CORE_MINOR_VER, MEGAMOL_CORE_COMP_REV
#define MEGAMOL_CORE_VERSION_STR MEGAMOL_STRINGIZE(MEGAMOL_CORE_VERSION)

#define MEGAMOL_CORE_COPYRIGHT "Copyright (c) 2006 - " MEGAMOL_STRINGIZE(MEGAMOL_CORE_LCYEAR) " by " MEGAMOL_PRODUCT_COMPANY "\n" \
    "Alle Rechte vorbehalten.\n" \
    "All rights reserved.\n" MEGAMOL_CORE_DIRTYTEXT

#if (MEGAMOL_CORE_MAJOR_VER == 1)
#if (MEGAMOL_CORE_MINOR_VER == 1) // v1.1
#define MEGAMOL_CORE_NAME MEGAMOL_PRODUCT_NAME " Core (Post-Orphan)"
#elif (MEGAMOL_CORE_MINOR_VER == 2) // v1.2
#define MEGAMOL_CORE_NAME MEGAMOL_PRODUCT_NAME " Core (Evolution Chamber)"
#endif

#elif (MEGAMOL_CORE_MAJOR_VER == 0)
#if (MEGAMOL_CORE_MINOR_VER == 4) // v0.4
#define MEGAMOL_CORE_NAME MEGAMOL_PRODUCT_NAME " Core (Skynet)"
#elif (MEGAMOL_CORE_MINOR_VER == 3) // v0.3
#define MEGAMOL_CORE_NAME MEGAMOL_PRODUCT_NAME " Core (Redesign 03)"
#elif (MEGAMOL_CORE_MINOR_VER == 1) // v0.1
#define MEGAMOL_CORE_NAME MEGAMOL_PRODUCT_NAME " Core (Prototype)"
#endif

#endif

#ifndef MEGAMOL_CORE_NAME
#define MEGAMOL_CORE_NAME MEGAMOL_PRODUCT_NAME " Core"
#endif

#define MEGAMOL_CORE_FILENAME "MegaMolCore" MEGAMOL_FILENAME_BITSD MEGAMOL_DLL_FILENAME_EXT

#define MEGAMOL_CORE_COMMENTS MEGAMOL_CORE_DIRTYTEXT

#endif /* MEGAMOL_CORE_VERSIONINFO_H_INCLUDED */
