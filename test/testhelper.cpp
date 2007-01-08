/*
 * testhelper.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testhelper.h"

#include <vislib/Console.h>


bool AssertTrue(const char *desc, const bool cond) {
    if (cond) {
        AssertOutput(desc);
        AssertOutputSuccess();
    } else {
        std::cout << std::endl;
        AssertOutput(desc);
        AssertOutputFail();
        std::cout << std::endl;
    }

    return cond;
}


bool AssertFalse(const char *desc, const bool cond) {
    return ::AssertTrue(desc, !cond);
}

void AssertOutput(const char *desc) {
    std::cout << "\"" << desc << "\" ";
}

void AssertOutputSuccess(void) {
    vislib::sys::Console::SetForegroundColor(vislib::sys::Console::GREEN);
    std::cout << "succeeded.";
    vislib::sys::Console::RestoreDefaultColors();
    std::cout << std::endl;
}

void AssertOutputFail(void) {
    vislib::sys::Console::SetForegroundColor(vislib::sys::Console::RED);
    std::cout << "FAILED.";
    vislib::sys::Console::RestoreDefaultColors();
    std::cout << std::endl;
}
