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

#include "megamol_build_info.h"
#include "vislib/VersionNumber.h"
#include "vislib/String.h"

#define MEGAMOL_CONSOLE_MAJOR_VER megamol::build_info::MEGAMOL_VERSION_MAJOR
#define MEGAMOL_CONSOLE_MINOR_VER megamol::build_info::MEGAMOL_VERSION_MINOR
#define MEGAMOL_CONSOLE_REV megamol::build_info::MEGAMOL_GIT_HASH


#if (MEGAMOL_CORE_API_DIRTY == 1) || (MEGAMOL_CONSOLE_DIRTY == 1)
#define MEGAMOL_CONSOLE_ISDIRTY 1
#define MEGAMOL_CONSOLE_DIRTYTEXT MEGAMOL_DIRTYWARNING
#else
#undef MEGAMOL_CONSOLE_ISDIRTY
#define MEGAMOL_CONSOLE_DIRTYTEXT ""
#endif


#define MEGAMOL_CONSOLE_VERSION MEGAMOL_CONSOLE_MAJOR_VER, MEGAMOL_CONSOLE_MINOR_VER, MEGAMOL_CONSOLE_REV
#define MEGAMOL_CONSOLE_VERSION_STR MEGAMOL_STRINGIZE(MEGAMOL_CONSOLE_VERSION)

#define MEGAMOL_CONSOLE_COPYRIGHT "Copyright (c) 2006 - 2021 by MegaMol Dev Team\n" \
    "Alle Rechte vorbehalten.\n" \
    "All rights reserved.\n" MEGAMOL_CONSOLE_DIRTYTEXT

#define MEGAMOL_CONSOLE_NAME "MegaMolCon" MEGAMOL_FILENAME_BITSD
#define MEGAMOL_CONSOLE_FILENAME MEGAMOL_CONSOLE_NAME MEGAMOL_EXE_FILENAME_EXT

#define MEGAMOL_CONSOLE_COMMENTS vislib::StringA("MegaMol Core API ") + vislib::VersionNumber(MEGAMOL_CONSOLE_VERSION).ToStringA() + vislib::StringA("\n") + vislib::StringA(MEGAMOL_CONSOLE_DIRTYTEXT)


#endif /* MEGAMOL_CONSOLE_VERSIONINFO_H_INCLUDED */
