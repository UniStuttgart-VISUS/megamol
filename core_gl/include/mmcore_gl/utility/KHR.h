/*
* KHR.h
*
* Copyright (C) 2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#pragma once

#include "mmcore/CoreInstance.h"

#include <string>
#include <sstream>

#ifndef _WIN32 
#include <execinfo.h>
#else
#include <Windows.h>
#include <DbgHelp.h>
#pragma comment(lib,"DbgHelp")
#endif


namespace megamol {
namespace core {
namespace utility {

    class MEGAMOLCORE_API KHR {
        public:
            static void DebugCallback(unsigned int source, unsigned int type, unsigned int id,
                unsigned int severity, int length, const char* message, void* userParam);

            static int startDebug();
            static std::string getStack();
    };

} // namespace utility
} // namespace core
} // namespace megamol

