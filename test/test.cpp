/*
 * test.cpp  09.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include <cstdio>
#include <cstdlib>

#include "testhelper.h"

#include "vislib/File.h"
#include "vislib/SystemException.h"
#include "vislib/PerformanceCounter.h"


#ifdef _WIN32
int _tmain(int argc, TCHAR **argv) {
#else
int main(int argc, char **argv) {
#endif
    using namespace vislib;
    using namespace vislib::sys;

    SystemException e1(2, __FILE__, __LINE__);
    ::_tprintf(_T("%s\n"), e1.GetMsg());

    Exception e2(__FILE__, __LINE__);
    ::_tprintf(_T("%s\n"), e2.GetMsg());
    
    //for (int i = 0; i < 100; i++) {
    //    ::_tprintf(_T("%lu\n"), PerformanceCounter::Query());
    //}
    
    
#ifdef _WIN32
    ::_tsystem(_T("pause"));
#endif
    return 0;
}

