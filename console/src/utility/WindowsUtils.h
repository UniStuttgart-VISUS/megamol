/*
 * WindowsUtils.h
 *
 * Copyright (C) 2016 by MegaMol Team, TU Dresden
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCON_UTILITY_WINDOWSUTILS_H_INCLUDED
#define MEGAMOLCON_UTILITY_WINDOWSUTILS_H_INCLUDED
#pragma once
#ifdef _WIN32

#include "mmcore/api/MegaMolCore.h"

namespace megamol {
namespace console {
namespace utility {

    void windowsConsoleWindowSetup(void* hCore);
    void MEGAMOLCORE_CALLBACK windowsConsoleLogEcho(unsigned int level, const char* message);

} /* end namespace utility */
} /* end namespace console */
} /* end namespace megamol */

#endif /* _WIN32 */
#endif /* MEGAMOLCON_UTILITY_WINDOWSUTILS_H_INCLUDED */
