/*
 * testhelper.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testhelper.h"

#include <iomanip>

#include <vislib/Console.h>


static bool _assertTrueShowSuccess = true;

static bool _assertTrueShowFailure = true;

static unsigned int testhelp_testSuccess = 0;

static unsigned int testhelp_testFail = 0;


bool AssertTrue(const char *desc, const bool cond) {
    if (cond) {
        if (::_assertTrueShowSuccess) {
            AssertOutput(desc);
            AssertOutputSuccess();
        } else {    // CROWBAR
            ::testhelp_testSuccess++;
        }
    } else {
        if (::_assertTrueShowFailure) {
            std::cout << std::endl;
            AssertOutput(desc);
            AssertOutputFail();
            std::cout << std::endl;
        } else {    // CROWBAR
            ::testhelp_testFail++;
        }
    }

    return cond;
}


bool AssertFalse(const char *desc, const bool cond) {
    return ::AssertTrue(desc, !cond);
}

void AssertOutput(const char *desc) {
    using namespace std;
    using namespace vislib::sys;

    Console::SetForegroundColor(Console::DARK_GRAY);
    cout << "[" << setw(4) << setfill('0') 
        << (testhelp_testSuccess + testhelp_testFail + 1) 
        << setw(0) << "] ";
    Console::RestoreDefaultColors();
    cout << "\"" << desc << "\" ";
}

void AssertOutputSuccess(void) {
    using namespace std;
    using namespace vislib::sys;

    testhelp_testSuccess++;
    Console::SetForegroundColor(Console::GREEN);
    cout << "succeeded.";
    Console::RestoreDefaultColors();
    cout << endl;
    cout << flush;
}

void AssertOutputFail(void) {
    using namespace std;
    using namespace vislib::sys;

    testhelp_testFail++;
    Console::SetForegroundColor(Console::RED);
    cout << "FAILED.";
    Console::RestoreDefaultColors();
    cout << endl;
    cout << flush;
}

void OutputAssertTestSummary(void) {
    std::cout << std::endl << "Assert Test Summary:" << std::endl;
    unsigned int testAll = testhelp_testFail + testhelp_testSuccess;

    if (testAll == 0) {
        std::cout << "  No Test conducted." << std::endl << std::endl;
        return;
    }

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
    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(1);
    std::cout << "  " << testhelp_testSuccess << " Tests succeeded (" << persSuccess << "%)" << std::endl;
    vislib::sys::Console::RestoreDefaultColors();

    if ((testAll > 0) && (testhelp_testFail > 0)) {
        vislib::sys::Console::SetForegroundColor(vislib::sys::Console::RED);
    }
    std::cout << "  " << testhelp_testFail << " Tests failed (" << persFail << "%)" << std::endl << std::endl;
    std::cout << std::resetiosflags(std::ios::fixed);

    vislib::sys::Console::RestoreDefaultColors();
}


void EnableAssertSuccessOutput(const bool isEnabled) {
    ::_assertTrueShowSuccess = isEnabled;
}

void EnableAssertFailureOutput(const bool isEnabled) {
    ::_assertTrueShowFailure = isEnabled;
}
