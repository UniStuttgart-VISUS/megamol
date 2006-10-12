/*
 * test.cpp  09.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include <cstdio>
#include <cstdlib>

#include "teststring.h"
#include "testfloat16.h"
#include "testthread.h"
#include "testfile.h"
#include "testvector.h"
#include "testdimandrect.h"

#include "vislib/SystemException.h"
#include "vislib/PerformanceCounter.h"
#include "vislib/StringConverter.h"
#include "vislib/SystemMessage.h"
#include "vislib/Trace.h"


#ifdef _WIN32
int _tmain(int argc, TCHAR **argv) {
#else
int main(int argc, char **argv) {
#endif
    using namespace vislib;
    using namespace vislib::sys;

    //vislib::Trace::GetInstance().EnableFileOutput("trace.txt");
    //vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_ALL);
    //TRACE(1, "HORST!\n");

    //::TestFile();
    //::TestString();
    ::TestVector();
    ::TestDimension();
    ::TestRectangle();
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

