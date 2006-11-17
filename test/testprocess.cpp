/*
 * testprocess.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testprocess.h"

#include "testhelper.h"
#include "vislib/Process.h"


void TestProcess(void) {
    using namespace vislib::sys;

    Process::Environment envi1;

    ::AssertTrue("New environment is empty", envi1.IsEmpty());
    ::AssertTrue("EMPTY_ENVIRONMENT is empty", Process::EMPTY_ENVIRONMENT.IsEmpty());
#ifdef _WIN32
    ::AssertEqual("Empty environment is NULL", static_cast<const void *>(envi1), static_cast<const void *>(NULL));
#else
    ::AssertEqual("Empty environment is NULL", static_cast<const char **>(envi1), static_cast<const char **>(NULL));
#endif 

    Process::Environment envi2("test=hugo", NULL);
    ::AssertFalse("Initialised environment is not empty", envi2.IsEmpty());

    Process::Environment envi3("test=hugo", "hugo=horst", NULL);
    ::AssertFalse("Initialised environment is not empty", envi3.IsEmpty());

}
