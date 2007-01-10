/*
 * testhelper.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testhelper.h"

#include <iomanip>

#include <vislib/Console.h>


static unsigned int testhelp_testSuccess = 0;

static unsigned int testhelp_testFail = 0;


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
    vislib::sys::Console::SetForegroundColor(vislib::sys::Console::DARK_GRAY);
    std::cout << "[" << std::setw(3) << std::setfill('0') << (testhelp_testSuccess + testhelp_testFail + 1) << std::setw(0) << "] ";
    vislib::sys::Console::RestoreDefaultColors();
    std::cout << "\"" << desc << "\" ";
}

void AssertOutputSuccess(void) {
    testhelp_testSuccess++;
    vislib::sys::Console::SetForegroundColor(vislib::sys::Console::GREEN);
    std::cout << "succeeded.";
    vislib::sys::Console::RestoreDefaultColors();
    std::cout << std::endl;
}

void AssertOutputFail(void) {
    testhelp_testFail++;
    vislib::sys::Console::SetForegroundColor(vislib::sys::Console::RED);
    std::cout << "FAILED.";
    vislib::sys::Console::RestoreDefaultColors();
    std::cout << std::endl;
}

void OutputAssertTestSummary(void) {
    std::cout << std::endl << "Assert Test Summary:" << std::endl;
    unsigned int testAll = testhelp_testFail + testhelp_testSuccess;
    double persFail = double(testhelp_testFail) / double(testAll) * 100.0;
    double persSuccess = double(testhelp_testSuccess) / double(testAll) * 100.0;
    std::cout << "  " << testAll << " Assert Tests." << std::endl;

    if (testAll > 0) {
        if (testhelp_testSuccess == testAll) {
            vislib::sys::Console::SetForegroundColor(vislib::sys::Console::GREEN);
        } else {
            vislib::sys::Console::SetForegroundColor(vislib::sys::Console::GRAY);
        }
    }
    std::cout << "  " << testhelp_testSuccess << " Tests succeeded (" << persSuccess << "%)" << std::endl;
    vislib::sys::Console::RestoreDefaultColors();

    if ((testAll > 0) && (testhelp_testFail > 0)) {
        vislib::sys::Console::SetForegroundColor(vislib::sys::Console::RED);
    }
    std::cout << "  " << testhelp_testFail << " Tests failed (" << persFail << "%)" << std::endl << std::endl;
    vislib::sys::Console::RestoreDefaultColors();
}
