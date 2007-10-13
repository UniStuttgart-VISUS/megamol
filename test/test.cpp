/*
 * test.cpp  09.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "vislib/Console.h"

/* include test implementations */
#include "testhelper.h"
#include "teststring.h"
#include "testfloat16.h"
#include "testthread.h"
#include "testfile.h"
#include "testvector.h"
#include "testdimandrect.h"
#include "testsysinfo.h"
#include "testprocess.h"
#include "testmatrix.h"
#include "testmisc.h"
#include "testcmdlineparser.h"
#include "testdate.h"
#include "testcollection.h"
#include "testpointers.h"
#include "testdiscovery.h"
#include "testthelog.h"
#include "testdirectoryiterator.h"
#include "testhash.h"
#include "testtrayicon.h"
#include "testenvironment.h"
#include "testnamedpipe.h"
#include "testsockets.h"
#include "testinterlocked.h"
#include "testtcpserver.h"


/* type for test functions */
typedef void (*VislibTestFunction)(void);

/* type for test manager structure */
typedef struct _VislibTest_t {
    TCHAR *testName; // the tests name. Used as command line argument to select this test.
    VislibTestFunction testFunc; // the function called when this test is selected.
    char *testDesc; // the description of this test. Used for the online help.
} VislibTest;


/* all available tests:
 * Add your tests here
 */
VislibTest tests[] = {
    // base
    {_T("Array"), ::TestArray, "Tests vislib::Array"},
    {_T("Heap"), ::TestHeap, "Tests vislib::Heap"},
    {_T("ColumnFormatter"), ::TestColumnFormatter, "Tests vislib::ColumnFormatter"},
    {_T("SmartPtr"), ::TestSmartPtr, "Tests vislib::SmartPtr"},
    {_T("String"), ::TestString, "Tests vislib::String and string utility classes"},
    {_T("Trace"), ::TestTrace, "Tests vislib tracing"},
    {_T("Hash"), ::TestHash, "Tests vislib hash providers"},
    // math
    {_T("Dimension"), ::TestDimension, "Tests vislib::math::Dimension"},
    {_T("Float16"), ::TestFloat16, "Tests vislib::math::Float16"},
    {_T("Matrix"), ::TestMatrix, "Tests vislib::math::Matrix"},
    {_T("Rectangle"), ::TestRectangle, "Tests vislib::math::Rectangle"},
    {_T("Vector"), ::TestVector, "Tests vislib::math::Vector"},
    // net
    {_T("ClusterDiscovery"), ::TestClusterDiscoveryService, "Tests vislib::net::ClusterDiscoveryService and utility classes"},
    {_T("ClusterDiscovery2"), ::TestClusterDiscoveryObserver, "Tests vislib::net::ClusterDiscoveryService in observer mode"},
    {_T("NetInfo"), ::TestNetworkInformation, "Tests vislib::net::NetworkInformation"},
    {_T("Sockets"), ::TestSockets, "Tests the sockets"},
    {_T("TcpServer"), ::TestTcpServer, "Tests the TCP server"},
    // sys
    {_T("CmdLineParser"), ::TestCmdLineParser, "Tests vislib::sys::CmdLineParser"},
    {_T("ConColors"), ::TestConsoleColours, "Tests colored console output using vislib::sys::Console"},
    {_T("DateTime"), ::TestDateTime, "Tests vislib::sys::DateTime"},
    {_T("DirIterator"), ::TestDirectoryIterator, "Test vislib::sys::DirectoryIterator"},
    {_T("File"), ::TestFile, "Tests vislib::sys::File and derived classes"},
    {_T("Log"), ::TestTheLogWithPhun, "Tests vislib::sys::Log"},
    {_T("Path"), ::TestPath, "Tests vislib::sys::Path"},
    {_T("Process"), ::TestProcess, "Tests vislib::sys::Process"},
    {_T("SysInfo"), ::TestSysInfo, "Tests vislib::sys::SystemInformation"},
    {_T("Thread"), ::TestThread, "Tests vislib::sys::Thread"},
    {_T("TrayIcon"), ::TestTrayIcon, "Tests vislib::sys::TrayIcon"},
    {_T("Environment"), ::TestEnvironment, "Tests vislib::sys::Environment"},
    {_T("NamedPipe"), ::TestNamedPipe, "Tests vislib::sys::NamedPipe (also requires 'vislib::sys::Thread' and 'vislib::sys::Mutex' to work correctly)"},
    {_T("Interlocked"), ::TestInterlocked, "Tests interlocked operations."},
    // end guard. Do not remove. Must be last entry
    {NULL, NULL, NULL}
};


#ifdef _WIN32
int _tmain(int argc, TCHAR **argv) {
#else
int main(int argc, char **argv) {
#endif
    unsigned int countTests = (sizeof(tests) / sizeof(VislibTest)) - 1;

    printf("VISlib Test Application\n\n");

    /* setting console title and icon. */
    /* could be in testmisc, but i like it here, running always. ;-) */
#ifdef _WIN32
    vislib::sys::Console::SetTitle(L"VISlib™ Test Application");
#else  /* _WIN32 */
    vislib::sys::Console::SetTitle("VISlib Test Application");
#endif /* _WIN32 */
    vislib::sys::Console::SetIcon(101);

    /* check command line arguments*/
    if (argc <= 1) {
        fprintf(stderr, "You must specify at least on test to perform.\n\n");
        fprintf(stderr, "Syntax:\n\t%s testname [testname] ...\n\n", argv[0]);
        fprintf(stderr, "Available Testnames are:\n");
        for (unsigned int i = 0; i < countTests; i++) {
            fprintf(stderr, "\t");
#ifdef _WIN32
            _ftprintf(stderr, _T("%s"), tests[i].testName);
#else /* _WIN32 */
            fprintf(stderr, "%s", tests[i].testName);
#endif /* _WIN32 */
            fprintf(stderr, "\n\t\t%s\n", tests[i].testDesc);
        }
        fprintf(stderr, "\n");

    } else {
        /* checking command line arguments for unknown tests */
        for (int i = 1; i < argc; i++) {
            bool found = false;

            for (unsigned int j = 0; j < countTests; j++) {
#ifdef _WIN32
#pragma warning(disable: 4996)
                if (_tcsicmp(argv[i], tests[j].testName) == 0) {
#pragma warning(default: 4996)
#else /* _WIN32 */
                if (strcasecmp(argv[i], tests[j].testName) == 0) {
#endif /* _WIN32 */
                    found = true;
                }

            }

            if (!found) {
                fprintf(stderr, "Warning: No Test named ");
#ifdef _WIN32
                _ftprintf(stderr, _T("%s"), argv[i]);
#else /* _WIN32 */
                fprintf(stderr, "%s", argv[i]);
#endif /* _WIN32 */                
                fprintf(stderr, " found. Ignoring this argument.\n\n");
            }
        }

        /* run selected tests */
        for (int i = 1; i < argc; i++) {
            for (unsigned int j = 0; j < countTests; j++) {
#ifdef _WIN32
#pragma warning(disable: 4996)
                if (_tcsicmp(argv[i], tests[j].testName) == 0) {
#pragma warning(default: 4996)
#else /* _WIN32 */
                if (strcasecmp(argv[i], tests[j].testName) == 0) {
#endif /* _WIN32 */

                    printf("Performing Test %d: ", i);
#ifdef _WIN32
                    _tprintf(_T("%s\n"), tests[j].testName);
#else /* _WIN32 */
                    printf("%s\n", tests[j].testName);
#endif /* _WIN32 */

                    tests[j].testFunc();

                    printf("\n");
                    break; // next argument
                }
            }
        }

        /* output test summary */
        ::OutputAssertTestSummary();
    }

#ifdef _WIN32
    ::_tsystem(_T("pause"));
#endif
    return 0;
}

