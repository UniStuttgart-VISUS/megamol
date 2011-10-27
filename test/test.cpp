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
#include "vislib/MessageBox.h"
#include "vislib/Trace.h"

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
#include "testdimred.h"
#include "testipc.h"
#include "testVIPCStrTab.h"
#include "testserialiser.h"
#include "testipv6.h"
#include "testthreadpool.h"
#include "testrefcount.h"
#include "testpoolallocator.h"
#include "testpoint.h"
#include "teststacktrace.h"
#include "testasyncsocket.h"
#include "testnetinfo.h"
#include "testfrustum.h"
#include "testbezier.h"
#include "testcomm.h"
#include "testmsgdisp.h"
#include "testpolynom.h"
#include "testquaternion.h"
#include "testtriangle.h"
#include "testBitmapImage.h"
#include "testReaderWriterLock.h"
#ifdef _WIN32
#include "testwinreg.h"
#endif /* _WIN32 */


/* type for test functions */
typedef void (*VislibTestFunction)(void);

/* type for test manager structure */
typedef struct _VislibTest_t {
    const TCHAR *testName; // the tests name. Used as command line argument to select this test.
    VislibTestFunction testFunc; // the function called when this test is selected.
    const char *testDesc; // the description of this test. Used for the online help.
} VislibTest;


/* all available tests:
 * Add your tests here
 */
VislibTest tests[] = {
    // base
    {_T("Array"), ::TestArray, "Tests vislib::Array"},
    {_T("ArraySort"), ::TestArraySort, "Tests vislib::Array::Sort"}, 
    {_T("ColumnFormatter"), ::TestColumnFormatter, "Tests vislib::ColumnFormatter"},
    {_T("Hash"), ::TestHash, "Tests vislib hash providers"},
    {_T("Heap"), ::TestHeap, "Tests vislib::Heap"},
    {_T("List"), ::TestSingleLinkedList, "Tests the single linked list"},
    {_T("ListSort"), ::TestSingleLinkedListSort, "Tests the 'Sort' Method of vislib::SingleLinkedList"},
    {_T("Map"), ::TestMap, "Tests vislib::Map"},
    {_T("Serialiser"), ::TestSerialiser, "Tests VISlib serialisers."},
    {_T("SmartPtr"), ::TestSmartPtr, "Tests vislib::SmartPtr"},
    {_T("StackTrace"), ::TestStackTrace, "Tests vislib::StackTrace"},
    {_T("String"), ::TestString, "Tests vislib::String and string utility classes"},
    {_T("Trace"), ::TestTrace, "Tests vislib tracing"},
    {_T("RefCount"), ::TestRefCount, "Tests VISlib ReferenceCounted and SmartRef"},
    {_T("RLEUINT"), ::TestRLEUInt, "Tests UINT RLE Encoding"},
    // graphics
    {_T("BitmapCodecSimple"), ::TestBitmapCodecSimple, "Performs very simple tests of vislib::graphics::*BitmapCodec"},
    {_T("NamedColours"), ::TestNamedColours, "Tests NamedColours"},
    {_T("BitmapPainter"), ::TestBitmapPainter, "Test BitmapPainter"},
    // math
    {_T("Dimension"), ::TestDimension, "Tests vislib::math::Dimension"},
    {_T("FastMap"), ::TestFastMap, "Tests vislib::math::FastMap"},
    {_T("Float16"), ::TestFloat16, "Tests vislib::math::Float16"},
    {_T("ForceDirected"), ::TestForceDirected, "Tests vislib::math::ForceDirected"},
    {_T("Matrix"), ::TestMatrix, "Tests vislib::math::Matrix"},
    {_T("Rectangle"), ::TestRectangle, "Tests vislib::math::Rectangle"},
    {_T("Vector"), ::TestVector, "Tests vislib::math::Vector"},
    {_T("Point"), ::TestPoint, "Tests vislib::math::Point"},
    {_T("Polynom"), ::TestPolynom, "Tests vislib::math::Polynom"},
    {_T("Quaternion"), ::TestQuaternion, "Test vislib::math::Quaternion"},
    {_T("Frustum"), ::TestFrustum, "Tests vislib::math::WorldSpaceFrustum"},
    {_T("Bezier"), ::TestBezier, "Tests vislib::math::BezierCurve"},
    {_T("Triangle"), ::TestTriangle, "Tests vislib::math::Triangle and ShallowTriangle"},
    {_T("CovarianceMatrix"), ::TestCovarianceMatrix, "Tests CalcCovarianceMatrix"},
    // net
    {_T("ClusterDiscovery"), ::TestClusterDiscoveryService, "Tests vislib::net::ClusterDiscoveryService and utility classes"},
    {_T("ClusterDiscovery2"), ::TestClusterDiscoveryObserver, "Tests vislib::net::ClusterDiscoveryService in observer mode"},
    {_T("IPv6"), ::TestIPv6, "Tests VISlib IPv6 support"},
    {_T("NetInfo"), ::TestNetworkInformation, "Tests vislib::net::NetworkInformation"},
    {_T("Sockets"), ::TestSockets, "Tests the sockets"},
    {_T("TcpServer"), ::TestTcpServer, "Tests the TCP server"},
    {_T("AsyncSockets"), ::TestAsyncSocket, "Tests the asynchronous socket extension"},
    {_T("Comm"), ::TestComm, "Tests the communication abstraction layer"},
    {_T("MsgDisp"), ::TestMsgDisp, "Tests the message dispatching facility on TCP/IP"},
    // sys
    {_T("ASCIIFileBuffer"), ::TestAsciiFile, "Tests vislib::sys::ASCIIFileBuffer (ON KLASSIK)"},
    {_T("CmdLineParser"), ::TestCmdLineParser, "Tests vislib::sys::CmdLineParser"},
    {_T("ConColors"), ::TestConsoleColours, "Tests colored console output using vislib::sys::Console"},
    {_T("DateTime"), ::TestDateTime, "Tests vislib::sys::DateTime"},
    {_T("DirIterator"), ::TestDirectoryIterator, "Test vislib::sys::DirectoryIterator"},
    {_T("Environment"), ::TestEnvironment, "Tests vislib::sys::Environment"},
    {_T("File"), ::TestFile, "Tests vislib::sys::File and derived classes"},
    {_T("FileNameSequence"), ::TestFileNameSequence, "Tests FileNameSequence."},
    {_T("Interlocked"), ::TestInterlocked, "Tests interlocked operations."},
    {_T("IPC"), ::TestIpc, "Tests inter-process communication"},
    {_T("IPC2"), ::TestIpc2, "For internal use only. Do not call."},
    {_T("Log"), ::TestTheLogWithPhun, "Tests vislib::sys::Log"},
    {_T("MTStackTrace"), ::TestMTStackTrace, "Tests vislib::sys::ThreadSafeStackTrace"},
    {_T("NamedPipe"), ::TestNamedPipe, "Tests vislib::sys::NamedPipe (also requires 'vislib::sys::Thread' and 'vislib::sys::Mutex' to work correctly)"},
    {_T("Path"), ::TestPath, "Tests vislib::sys::Path"},
    {_T("PoolAllocator"), ::TestPoolAllocator, "Tests vislib::sys::PoolAllocator"},
    {_T("Process"), ::TestProcess, "Tests vislib::sys::Process"},
    {_T("SysInfo"), ::TestSysInfo, "Tests vislib::sys::SystemInformation"},
    {_T("Thread"), ::TestThread, "Tests vislib::sys::Thread"},
    {_T("ThreadPool"), ::TestThreadPool, "Tests the thread pool"},
    {_T("TrayIcon"), ::TestTrayIcon, "Tests vislib::sys::TrayIcon"},
    {_T("VIPCStrTabGet"), ::TestVIPCStrTabGet, "Tests the getter functions of vislib::sys::VolatileIPCStringTable"},
    {_T("VIPCStrTabSet"), ::TestVIPCStrTabSet, "Tests the setter functions of vislib::sys::VolatileIPCStringTable"},
    {_T("PerfCounter"), ::TestPerformanceCounter, "Tests the performance counter"},
    {_T("FatRWLock"), ::TestFatReaderWriterLock, "Tests the FatReaderWriterLock"},
#ifdef _WIN32
    {_T("WinReg"), ::TestWinReg, "Tests the windows RegistryKey class"},
#endif /* _WIN32 */
    // end guard. Do not remove. Must be last entry.
    {NULL, NULL, NULL}
};


#ifdef _WIN32
int _tmain(int argc, TCHAR **argv) {
#if defined(DEBUG) | defined(_DEBUG)
    ::_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif /* defined(DEBUG) | defined(_DEBUG) */
#else
int main(int argc, char **argv) {
#endif
    unsigned int countTests = (sizeof(tests) / sizeof(VislibTest)) - 1;

    printf("VISlib Test Application\n\n");

    /* setting console title and icon. */
    /* could be in testmisc, but i like it here, running always. ;-) */
    vislib::sys::Console::SetTitle("VISlib Test Application");
    /* note: unicode title does not even work under windows, ... so */
    vislib::sys::Console::SetIcon(101);

    /* check command line arguments*/
    if (argc <= 1) {

        //try {
        //    vislib::sys::MessageBox::ReturnValue rv 
        //        = vislib::sys::MessageBox::Show(
        //        "You must specify at least one test to be performed.",
        //        "VISlib Test",
        //        vislib::sys::MessageBox::BTNS_OK,
        //        //vislib::sys::MessageBox::BTNS_OKCANCEL,
        //        //vislib::sys::MessageBox::BTNS_CANCELRETRYCONTINUE,
        //        //vislib::sys::MessageBox::ICON_NONE);
        //        //vislib::sys::MessageBox::ICON_ERROR);
        //        //vislib::sys::MessageBox::ICON_WARNING);
        //        //vislib::sys::MessageBox::ICON_INFO);
        //        vislib::sys::MessageBox::ICON_QUESTION);
        //    exit(0);
        //
        //    if (rv == vislib::sys::MessageBox::RET_CANCEL) {
        //        printf("Canceled\n");
        //        exit(0);
        //    } else if (rv == vislib::sys::MessageBox::RET_RETRY) {
        //        printf("There is nothing to be retried!\n");
        //    }
        //} catch(vislib::Exception e) {
        //    fprintf(stderr, "Exception: %s\n", e.GetMsgA());
        //}

        fprintf(stderr, "You must specify at least one test to be performed.\n\n");
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

                    try {

                        tests[j].testFunc();

                    } catch (vislib::Exception& e) {
#ifdef _WIN32
                        _tprintf(_T("\nUnexpected vislib::Exception: %s "), e.GetMsg());
#else /* _WIN32 */
                        printf("\nUnexpected vislib::Exception: %s ", e.GetMsgA());
#endif /* _WIN32 */
                        AssertOutputFail(); // add a generic fail
                    } catch (...) {
#ifdef _WIN32
                        _tprintf(_T("\nUnexpected Exception "));
#else /* _WIN32 */
                        printf("\nUnexpected Exception ");
#endif /* _WIN32 */
                        AssertOutputFail(); // add a generic fail
                    }

                    printf("\n");
                    break; // next argument
                }
            }
        }

        /* output test summary */
        ::OutputAssertTestSummary();
    }

#if defined(_WIN32) && defined(_DEBUG) // VC Debugger Halt on Stop Crowbar
#pragma warning(disable: 4996)
    if (getenv("_MSVC_STOP_AFTER_DEBUG_") != NULL) system("pause");
#pragma warning(default: 4996)
#endif
    return 0;
}

