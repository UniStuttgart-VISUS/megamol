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

    Process::Environment envi;

    ::AssertTrue("New environment is empty", envi.IsEmpty());
    ::AssertTrue("EMPTY_ENVIRONMENT is empty", Process::EMPTY_ENVIRONMENT.IsEmpty());
#ifdef _WIN32
    ::AssertEqual("Empty environment is NULL", static_cast<const void *>(envi), static_cast<const void *>(NULL));
#else
    ::AssertEqual("Empty environment is NULL", static_cast<const char **>(envi), static_cast<const char **>(NULL));
#endif 

    envi.~Environment();
    new (&envi) Process::Environment("test=hugo", NULL);
    ::AssertFalse("Initialised environment is not empty", envi.IsEmpty());

    envi.~Environment();
    new (&envi) Process::Environment("test=hugo", "hugo=horst", NULL);
    ::AssertFalse("Initialised environment is not empty", envi.IsEmpty());

}
