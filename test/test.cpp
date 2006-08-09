/*
 * test.cpp  09.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include <cstdio>
#include <cstdlib>

#include "vislib/File.h"
#include "vislib/SystemException.h"


#ifdef _WIN32
int _tmain(int argc, TCHAR **argv) {
#else
int main(int argc, char **argv) {
#endif

    vislib::sys::SystemException e(2, __FILE__, __LINE__);

    ::_tprintf(e.GetMsg());
    
#ifdef _WIN32
    ::_tsystem(_T("pause"));
#endif
    return 0;
}

