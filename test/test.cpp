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


int _tmain(int argc, TCHAR **argv) {
    vislib::sys::SystemException e(2, __FILE__, __LINE__);

    ::_tprintf(e.GetMsg());
    
    ::_tsystem(_T("pause"));
    return 0;
}

