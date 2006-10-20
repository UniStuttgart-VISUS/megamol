/*
 * test.cpp  09.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "teststring.h"
#include "testfloat16.h"
#include "testthread.h"
#include "testfile.h"
#include "testvector.h"
#include "testdimandrect.h"

#include "vislib/Exception.h"
#include "vislib/SystemException.h"
#include "vislib/PerformanceCounter.h"
#include "vislib/StringConverter.h"
#include "vislib/SystemMessage.h"
#include "vislib/Trace.h"
#include "vislib/SystemInformation.h"


#ifdef _WIN32
int _tmain(int argc, TCHAR **argv) {
#else
int main(int argc, char **argv) {
#endif
    using namespace vislib;
    using namespace vislib::sys;
    
    try {
        vislib::TString machName;
        vislib::sys::SystemInformation::GetMachineName(machName);
        ::_tprintf(_T("Running on %s (%u Processor Machine)\n"), machName.PeekBuffer(), vislib::sys::SystemInformation::GetProcessorCount());

        std::cout << "Page Size: " << vislib::sys::SystemInformation::GetPageSize() << " Bytes." << std::endl;
        std::cout << "Total Memory: " << vislib::sys::SystemInformation::GetPhysicalMemorySize() << " Bytes." << std::endl;
        std::cout << "Free Memory:  " << vislib::sys::SystemInformation::GetAvailableMemorySize() << " Bytes." << std::endl;

        std::cout << "System Type: " << vislib::sys::SystemInformation::GetSystemType() << std::endl;
        std::cout << "System Word Size: " << vislib::sys::SystemInformation::GetSystemWordSize() << std::endl;
        std::cout << "Self System Type: " << vislib::sys::SystemInformation::GetSelfSystemType() << std::endl;
        std::cout << "Self Word Size: " << vislib::sys::SystemInformation::GetSelfWordSize() << std::endl;

    } catch(Exception e) {
        std::cout << "Exception catched: " << e.GetMsg() << std::endl;
    } catch(...) {
        std::cout << "Unexpected exception catched." << std::endl;
    }

    //vislib::Trace::GetInstance().EnableFileOutput("trace.txt");
    //vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_ALL);
    //TRACE(1, "HORST!\n");

    //::TestFile();
    //::TestString();
    //::TestVector();
    //::TestDimension();
    //::TestRectangle();
    //::TestVector();
    //::TestFloat16();
    //::TestThread();

    //::printf("%s", W2A(L"Hugo"));

    //SystemException e1(2, __FILE__, __LINE__);
    //::_tprintf(_T("%s\n"), e1.GetMsg());

    //Exception e2(__FILE__, __LINE__);
    //::_tprintf(_T("%s\n"), e2.GetMsg());
    
    //for (int i = 0; i < 100; i++) {
    //    ::_tprintf(_T("%lu\n"), PerformanceCounter::Query());
    //}
    
    //SystemMessage sysMsg(4);
    //::_tprintf(_T("%s\n"), static_cast<const TCHAR *>(sysMsg));

#ifdef _WIN32
    ::_tsystem(_T("pause"));
#endif
    return 0;
}

