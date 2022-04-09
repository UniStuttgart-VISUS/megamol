/*
 * KHR.h
 *
 * Copyright (C) 2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/CoreInstance.h"

#include <sstream>
#include <string>

#ifndef _WIN32
#include <execinfo.h>
#else
#include <DbgHelp.h>
#include <Windows.h>
#pragma comment(lib, "DbgHelp")
#endif


namespace megamol {
namespace core {
namespace utility {

class KHR {
public:
    static void DebugCallback(unsigned int source, unsigned int type, unsigned int id, unsigned int severity,
        int length, const char* message, void* userParam);

    static int startDebug();
    static std::string getStack();
};

} // namespace utility
} // namespace core
} // namespace megamol
